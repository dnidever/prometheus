import os
import errno
import numpy as np
from dlnpyutils import utils as dln, robust
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.table import Table
import astropy.units as u
from . import groupfit,allfit,models,leastsquares as lsq
from .ccddata import CCDData

# ALLFRAME-like forced photometry

# also see multifit.py

def solveone(psf,im,cat,method='qr',bounds=None,radius=None,absolute=False):

    method = str(method).lower()
  
    # Image offset for absolute X/Y coordinates
    if absolute:
        imx0 = im.bbox.xrange[0]
        imy0 = im.bbox.yrange[0]

    xc = cat['x']
    yc = cat['y']
    if absolute:  # offset
        xc -= imx0
        yc -= imy0
        if bounds is not None:
            bounds[0][1] -= imx0  # lower
            bounds[0][2] -= imy0
            bounds[1][1] -= imx0  # upper
            bounds[1][2] -= imy0
    if radius is None:
        radius = np.maximum(psf.fwhm(),1)
    bbox = psf.starbbox((xc,yc),im.shape,radius)
    X,Y = psf.bbox2xy(bbox)

    # Get subimage of pixels to fit
    # xc/yc might be offset
    flux = im.data[bbox.slices]
    err = im.error[bbox.slices]
    wt = 1.0/np.maximum(err,1)**2  # weights
    skyim = im.sky[bbox.slices]
    xc -= bbox.ixmin  # offset for the subimage
    yc -= bbox.iymin
    X -= bbox.ixmin
    Y -= bbox.iymin
    if bounds is not None:
        bounds[0][1] -= bbox.ixmin  # lower
        bounds[0][2] -= bbox.iymin
        bounds[1][1] -= bbox.ixmin  # upper
        bounds[1][2] -= bbox.iymin            
    xdata = np.vstack((X.ravel(), Y.ravel()))        
    #sky = np.median(skyim)
    sky = 0.0
    if 'amp' in cat:
        amp = cat['amp']
    else:
        amp = flux[int(np.round(yc)),int(np.round(xc))]-sky   # python images are (Y,X)
        amp = np.maximum(amp,1)  # make sure it's not negative
            
    initpar = [amp,xc,yc,sky]
 
    # Use Cholesky, QR or SVD to solve linear system of equations
    m,jac = psf.jac(xdata,*initpar,retmodel=True)
    dy = flux.ravel()-m.ravel()
    # Solve Jacobian
    dbeta = lsq.jac_solve(jac,dy,method=method,weight=wt.ravel())
    dbeta[~np.isfinite(dbeta)] = 0.0  # deal with NaNs
    chisq = np.sum(dy**2 * wt.ravel())/len(dy)

    # Output values
    newamp = np.maximum(amp+dbeta[0], 0)  # amp cannot be negative
    dx = dbeta[1]
    dy = dbeta[2]

    return newamp,dx,dy


def solve(psf,resid,tab,verbose=False):
    """
    Solve for the flux and find corrections for x and y.

    Parameters
    ----------
    psf : psf model
       The image PSF model.
    resid : image
       Residual image with initial estimate of star models subtracted.
    tab : table
       Table of stars ot fit.
    verbose : bool, optional
       Verbose output to the screen.  Default is False.

    Returns
    -------
    out : table
       Table of results

    Example
    -------

    out = solve(psf,resid,meastab)

    """

    ntab = len(tab)
    out = tab.copy()
    
    # Loop over the stars
    for i in range(ntab):
        
        # Add the previous best-fit model back in to the image
        if tab['amp'][i] > 0:
            # nocopy=True will change in place
            _ = psf.add(resid.data,tab[i:i+1],nocopy=True)

        # Solve single flux
        newamp,dx,dy = solveone(psf,resid,tab[i:i+1])

        # Save the results
        out['amp'][i] = newamp
        out['dx'][i] = dx
        out['dy'][i] = dy 
        
        # Immediately subtract the new model
        #  npcopy=True will change in place
        _ = psf.sub(resid.data,out[i:i+1],nocopy=True)

        #print(i,newamp,dx,dy)
        
    return out,resid
        

def forced(images,mastertab,fitpm=False,reftime=None,refwcs=None,verbose=True):
    """
    ALLFRAME-like forced photometry.

    Parameters
    ----------
    images : list
       List of image filenames or list of CCD objects.
    mastertab : table
       Master table of objects.
    fitpm : boolean, optional
       Fit proper motions as well as central positions.
         Default is False.
    reftime : Time object, optional
       The reference time to use.  By default, the mean
         JD of all images is used.
    refwcs : WCS object, optional
       The reference WCS delineating a region in which the
        the images are to be fit.  By default, a WCS is
        generated with a tangent plane that covers all the
        sources in the master object table (objtab).

    Returns
    -------

    Example
    -------

    out = forced()

    """

    # fit proper motions as well
    # don't load all the data at once, only what you need
    #   maybe use memory maps

    # The default behavior of fits.open() is to use memmap=True
    # and only load the data into physical memory when it is accessed
    # e.g. hdu[0].data.
    # after closing the hdu you still need to delete the data i.e.
    # del hdu[0].data, because there's still a memory map open.

    # I created a function in utils.refresh_mmap() that you can give
    # an open HDUList() and it will refresh the mmap and free up the
    # virtual memory.
    
    nimages = len(images)
    print('Running forced photometry on {:d} images'.format(nimages))

    # Load files if necessary
    if isinstance(images[0],str):
        files = images
        images = []
        for i in range(nimages):
            print('Loading image {:d}  {:s}'.format(i+1,files[i]))
            if os.path.exists(files[i]):
                im1 = ccddata.CCDData.read(files[i])
                # Load PSF model
                psffile = files[i].replace('.fits','.psf.fits')
                if os.path.exists(psffile):
                    psf1 = models.read(psffile)
                    im1.psf = psf1
                else:
                    print('PSF file not found',psffile)
                    continue
                # Put the image in the list      
                images.append(im1)
            else:
                print(files[i],' not found')
        nimages = len(images)
                
    # Check that all the images have a PSF model
    tempimages = images
    images = []
    for i in range(len(tempimages)):
        if hasattr(tempimages[i],'psf'):
            images.append(tempimages[i])
        else:
            print('Image {:d} does not have a PSF model attached'.format(i+1))
    del tempimages
    nimages = len(images)
    
    # Load the master star table if necessary
    if isinstance(mastertab,str):
        if os.path.exists(mastertab)==False:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), mastertab)
        mastertab_filename = mastertab
        mastertab = Table.read(mastertab_filename)
    # Make sure we have the necessary columns
    for c in mastertab.colnames: mastertab[c].name = c.lower()
    if 'ra' not in mastertab.colnames or 'dec' not in mastertab.colnames:
        raise ValueError('ra and dec columns must exist in mastertab')
    
    # Initialize the array of positions and proper motions
    nobj = len(mastertab)
    print('Master  list has {:d} stars'.format(nobj))
    objtab = mastertab.copy()
    objtab['objid'] = 0
    objtab['objid'] = np.arange(nobj)+1
    objtab['cenra'] = objtab['ra'].copy()
    objtab['cendec'] = objtab['dec'].copy()
    objtab['cenpmra'] = 0.0
    objtab['cenpmdec'] = 0.0
    objtab['nmeas'] = 0
    objtab['converged'] = False
    
    coo0 = SkyCoord(ra=objtab['cenra']*u.deg,dec=objtab['cendec']*u.deg,frame='icrs')
    
    # Get information about all of the images
    dt = [('dateobs',str,26),('jd',float),('cenra',float),('cendec',float),
          ('nx',int),('ny',int),('vra',float,4),
          ('vdec',float,4),('exptime',float),('startmeas',int),('nmeas',int)]
    iminfo = Table(np.zeros(nimages,dtype=np.dtype(dt)))
    for i in range(nimages):
        iminfo['dateobs'][i] = images[i].header['DATE-OBS']
        iminfo['jd'][i] = Time(iminfo['dateobs'][i]).jd
        nx = images[i].header['NAXIS1']
        ny = images[i].header['NAXIS2']
        iminfo['nx'][i] = nx
        iminfo['ny'][i] = ny
        cencoo = images[i].wcs.pixel_to_world(nx//2,ny//2)
        iminfo['cenra'][i] = cencoo.ra.deg
        iminfo['cendec'][i] = cencoo.dec.deg
        vra,vdec = images[i].wcs.wcs_pix2world([0,nx-1,nx-1,0],[0,0,ny-1,ny-1],0)
        iminfo['vra'][i] = vra
        iminfo['vdec'][i] = vdec        
        isin = coo0.contained_by(images[i].wcs,images[i].data)
        iminfo['nmeas'][i] = np.sum(isin)
        print('Image {:d} - {:d} stars overlap'.format(i+1,np.sum(isin)))
        
    # Get the reference epoch, mean epoch
    refepoch = Time(np.mean(iminfo['jd']),format='jd')
    
    # Get the reference WCS

    # Make object and measurement tables
    # with indices from one to the other
    # many objects won't appear in many images
        
    # Make the measurement table
    dt = [('objid',int),('objindex',int),('imindex',int),('jd',float),
          ('ra',float),('dec',float),('x',float),('y',float),('amp',float),
          ('flux',float),('fluxerr',float),('dflux',float),('dfluxerr',float),
          ('dx',float),('dxerr',float),('dy',float),('dyerr',float),('dra',float),
          ('ddec',float),('sky',float),('converged',bool)]
    nmeas = np.sum(iminfo['nmeas'])
    meastab = np.zeros(nmeas,dtype=np.dtype(dt))
    meascount = 0
    for i in range(nimages):
        contained = coo0.contained_by(images[i].wcs,images[i].data)
        isin, = np.where(contained)
        nisin = len(isin)
        objtab['nmeas'][isin] += 1 
        if nisin > 0:
            meastab['objid'][meascount:meascount+nisin] = objtab['objid'][isin]
            meastab['objindex'][meascount:meascount+nisin] = isin
            meastab['imindex'][meascount:meascount+nisin] = i
            meastab['jd'][meascount:meascount+nisin] = iminfo['jd'][i]
            meascount += nisin

    # Some stars have zero measurements
    zeromeas, = np.where(objtab['nmeas']==0)
    if len(zeromeas)>0:
        print('{:d} objects have zero measurements'.format(len(zeromeas)))
            
    # Create object index into measurement table
    oindex = dln.create_index(meastab['objindex'])
    objindex = nobj*[[]]  # initialize index with empty list for each object
    for i in range(len(oindex['value'])):
        ind = oindex['index'][oindex['lo'][i]:oindex['hi'][i]+1]
        objindex[oindex['value'][i]] = ind    # value is the index in objtab
    
    # Iterate until convergence has been reached
    count = 0
    flag = True
    while (flag):

        print('----- Iteration {:d} -----'.format(count+1))
        
        # Loop over the images:
        for i in range(nimages):
            residfile = 'image{:d}_resid.npy'.format(i+1)            
            im = images[i]
            imtime = Time(iminfo['jd'][i],format='jd')
            
            # Get the objects and measurements that overlap this image
            contained = coo0.contained_by(images[i].wcs,images[i].data)            
            objtab1 = objtab[contained]
            msbeg = iminfo['startmeas'][i]
            msend = msbeg + iminfo['nmeas'][i]        
            meastab1 = meastab[msbeg:msend]
            print('Image {:d}  {:d} stars'.format(i+1,iminfo['nmeas'][i]))
            
            # Calculate x/y position for each object in this image
            # using the current best overall on-the-sky position
            # and proper motion
            # Need to convert celestial values to x/y position in
            # the image using the WCS.
            coo1 = SkyCoord(ra=objtab1['cenra']*u.deg,dec=objtab1['cendec']*u.deg,
                           pm_ra_cosdec=objtab1['cenpmra']*u.mas/u.year,
                           pm_dec=objtab1['cenpmdec']*u.mas/u.year,
                           obstime=refepoch,frame='icrs')
            # Use apply_space_motion() method to get coordinates for the
            #  epoch of this image
            newcoo1 = coo1.apply_space_motion(imtime)
            meastab1['ra'] = newcoo1.ra.deg
            meastab1['dec'] = newcoo1.dec.deg            
            
            # Now convert to image X/Y coordinates
            x,y = im.wcs.world_to_pixel(newcoo1)
            meastab1['x'] = x
            meastab1['y'] = y            

            # Initialize or load the residual image
            resid = im.copy()            
            if count>0:
                resid_data = np.load(residfile)
                resid.data = resid_data
            
            # Subtract sky
            if count % 5 == 0:
                print('Recomputing and subtracting the sky')
                resid._sky = None  # force it to be recomputed
                resid.data -= resid.sky

            # Fit the fluxes while holding the positions fixed            
            out,resid = solve(im.psf,resid,meastab1,verbose=verbose)

            # Save the residual file
            np.save(residfile,resid.data)

            
            # SOMETHING IS WRONG, I THINK THE PSF MIGHT NOT BE GOOD

            #from dlnpyutils import plotting as pl 
            #import pdb; pdb.set_trace()
            
            # Stuff the information back in
            meastab['amp'][msbeg:msend] = out['amp']
            meastab['dx'][msbeg:msend] = out['dx']
            meastab['dy'][msbeg:msend] = out['dy']            
            
            # allframe operates on the residual map, with the best-fit model subtracted
            
            # allframe derived flux and centroid corrections for each object
            # the flux corrections are applied immediately while the centroid
            # corrections are saved.

            # there are no groups in allframe
            # the least-squares design matrix is completely diagonalized: the
            # incremental brightness and position corrections are derived for
            # each star separately!  this may add a few more iterations for the
            # badly blended stars, but it does *not* affect the accuracy of the
            # final results.

            # Once a star has converged, it's best-fit model is subtracted from
            # the residual map and its parameters are fixed.
            
            # when no further infinitesimal change to a star's parameters
            # produces any reduction in the robust estimate of the mean-square
            # brightness residual inside that star's fitting region, the
            # maximum-likelihood solution has been achieved.
            
            # Calculate dx/dy residuals for each source
            # convert from pixel to work offset using image wcs

            # need to save the residual image and uncertainties
            # that's all we need to solve the least-squares problem.
            
        # Calculate new coordinates and proper motions based on the x/y residuals
        # the accumulated centroid corrections are projected through the individual
        # frames' geometric transformations to the coordinate system of the master list
        # and averaged.  These net corrections are applied to the stars' positions as
        # retained in the master list.

        # Can also make modest corrections to the input geometric transformation
        # equations by evaluating and removing systematic trends in the centroid
        # corrections derived for stars in each input image.

        # Loop over object and calculate coordinate and proper motion corrections
        for i in range(nobj):
            measind = objindex[i]
            meas1 = meastab[measind]
            # robust linear fit
            racoef = robust.linefit(meas1['jd'],meas1['dra'])
            deccoef = robust.linefit(meas1['jd'],meas1['ddec'])            
            # update object cenra, cendec, cenpmra, cenpmdec
            objtab['cenra'][i] += racoef[0]
            objtab['cendec'][i] += deccoef[0]
            objtab['cenpmra'][i] += racoef[1]  # multiply by cos(dec)???
            objtab['cenpmdec'][i] += deccoef[1]    
            

        # Check for convergence
        count += 1
            
        import pdb; pdb.set_trace()

    return objtab,meastab

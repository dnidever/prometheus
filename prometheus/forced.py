import os
import numpy as np
from dlnpyutils import utils as dln, robust
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.table import Table
import astropy.units as u
from . import groupfit,allfit

# ALLFRAME-like forced photometry

# also see multifit.py

def forced(images,objtab,fitpm=False,reftime=None,refwcs=None):
    """
    ALLFRAME-like forced photometry.

    Parameters
    ----------
    images : list
       List of image filenames or list of CCD objects.
    objtab : table
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

    # Initialize the array of positions and proper motions
    nobj = len(mastertab)
    objtab = mastertab.copy()
    objtab['objid'] = 0
    objtab['objid'] = np.arange(nobj)+1
    objtab['cenra'] = 0.0
    objtab['cendec'] = 0.0
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
        iminfo['ny'][i] = nyx
        cenra,cendec = images[i].wcs.pixel_to_world(nx//2,ny//2)
        iminfo['cenra'][i] = cenra
        iminfo['cendec'][i] = cendec
        vra,vdec = wcs1.wcs_pix2world([0,nx-1,nx-1,0],[0,0,ny-1,ny-1],0)
        iminfo['vra'][i] = vra
        iminfo['vdec'][i] = vdec        
        isin = coo0.contained_by(images[i].wcs,images[i].data)
        iminfo['nmeas'][i] = np.sum(isin)
        
    # Get the reference epoch, mean epoch
    refepoch = Time(mean(iminfo['jd']),format='jd')

    # Get the reference WCS

    # Make object and measurement tables
    # with indices from one to the other
    # many objects won't appear in many images
        
    # Make the measurement table
    dt = [('objid',int),('objindex',int),('imindex',int),('jd',float),
          ('ra',float),('dec',float),('x',float),('y',float),('flux',float),
          ('fluxerr',float),('dflux',float),('dfluxerr',float),('dx',float),
          ('dxerr',float),('dy',float),('dyerr',float),('dra',float),
          ('ddec',float),('converged',bool)]
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

    # Create object index into measurement table
    oindex = dln.create_index(meastab['objindex'])
    objindex = nobj*[[]]  # initialize index with empty list for each object
    for i in range(len(ind1)):
        ind = oindex['index'][oindex['lo'][i]:oindex['hi'][i]+1]
        objindex[oindex['value'][i]] = ind    # value is the index in objtab
        
    
    # Iterate until convergence has been reached
    count = 0
    flag = True
    while (flag):

        # Loop over the images:
        for i in range(nimages):
            im = images[i]
            imtime = Time(iminfo['jd'][i],format='jd')

            # Get the objects and measurements that overlap this image
            contained = coo0.contained_by(images[i].wcs,images[i].data)            
            objtab1 = objtab[contained]
            msbeg = iminfo['startmeas'][i]
            msend = meas_beg + iminfo['nmeas'][i]        
            meastab1 = meastab[msbeg:msend+1]
            
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
            
            # Fit the fluxes while holding the positions fixed
            out,model = allfit.fit(im.psf,im,meastab1,method='qr',fitradius=None,recenter=False,
                                   maxiter=10,minpercdiff=0.5,reskyiter=2,nofreeze=False,
                                   skyfit=True,verbose=False)

            # Stuff the information back in
            meastab[msbeg:msend+1] = meastab1
            
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

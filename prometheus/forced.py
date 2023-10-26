import os
import errno
import numpy as np
from dlnpyutils import utils as dln, robust, coords
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from astropy.table import Table,vstack,hstack
import astropy.units as u
from . import groupfit,allfit,models,leastsquares as lsq
from .ccddata import CCDData

# ALLFRAME-like forced photometry

# also see multifit.py

def getimageinfo(files):
    """
    Gather information about the images.

    Parameters
    ----------
    files : list
       List of image FITS files.  There should be associated _prometheus.fits
        that include the PSF and source catalog.

    Returns
    -------
    iminfo : table
       List of dictionaries with information on the FITS files including the
        header, WCS, PSF, source catalog, but not the image itself.

    Example
    -------

    iminfo = getimageinfo(files)

    """

    # Load images headers and WCS
    iminfo = []
    nfiles = len(files)
    for i in range(nfiles):
        print('Image {:d}  {:s}'.format(i+1,files[i]))
        if os.path.exists(files[i])==False:
            print(files[i],' not found')
            continue
        prfile = files[i].replace('.fits','_prometheus.fits')            
        if os.path.exists(prfile)==False:
            print(prfile,' not found')
            continue
        # Load header
        head1 = fits.getheader(files[i],0)
        wcs1 = WCS(head1)
        # Load soure table and PSF model from prometheus output file
        tab1 = Table.read(prfile,1)
        for c in tab1.colnames: tab1[c].name = c.lower()
        psf1 = models.read(prfile,4)
        dateobs = head1['DATE-OBS']
        jd = Time(dateobs).jd
        exptime = head1.get('exptime')
        filt = head1.get('filter')
        nx = head1['NAXIS1']
        ny = head1['NAXIS2']        
        cencoo = wcs1.pixel_to_world(nx//2,ny//2)
        vra,vdec = wcs1.wcs_pix2world([0,nx-1,nx-1,0],[0,0,ny-1,ny-1],0)        
        iminfo.append({'file':files[i],'residfile':files[i].replace('.fits','_resid.npy'),
                       'header':head1,'dateobs':dateobs,'jd':jd,'exptime':exptime,'filter':filt,
                       'nx':nx,'ny':ny,'wcs':wcs1,'cenra':cencoo.ra.deg,'cendec':cencoo.dec.deg,
                       'vra':vra,'vdec':vdec,'table':tab1,'psf':psf1,'startmeas':-1,'nmeas':-1})

    return iminfo
    
def makemastertab(images,dcr=1.0,mindet=2):
    """
    Make master star list from individual image star catalogs.

    Parameters
    ----------
    images : list
       List of dictionaries containing the WCS and catalog
        information.
    dcr : float, optional
       Cross-matching radius in arcsec.  Default is 1.0 arcsec.
    mindet : int, optional
       Minimum number of detections to be counted in master list.
        Default is 2.

    Returns
    -------
    mastertab : table
       Table of unique sources in the images.

    Example
    -------

    mastertab = mkmastertab(images)

    """

    nimages = len(images)

    objdt = [('objid',int),('ra',float),('dec',float),('amp',float),
             ('flux',float),('nmeas',int)]
    
    # Loop over the images
    for i in range(nimages):
        tab1 = images[i]['table']
        tab1['objid'] = -1
        tab1['ra'].unit = None   # no units
        tab1['dec'].unit = None        
        # First catalog
        if i==0:
            tab1['objid'] = np.arange(len(tab1))+1
            meas = tab1.copy()
            obj = Table(np.zeros(len(meas),dtype=np.dtype(objdt)))
            for c in ['ra','dec']: obj[c] = meas[c]
            obj['nmeas'] = 1
        # 2nd and later catalog, need to crossmatch
        else:
            # Cross-match
            ind1,ind2,dist = coords.xmatch(obj['ra'],obj['dec'],
                                           tab1['ra'],tab1['dec'],dcr,unique=True)
            # Some matches
            if len(ind1)>0:
                tab1['objid'][ind2] = obj['objid'][ind1]
                obj['nmeas'][ind1] += 1
            # Some left, add them to object table
            if len(ind1) < len(tab1):
                leftind = np.arange(len(tab1))
                leftind = np.delete(leftind,ind2)
                tab1['objid'][leftind] = np.arange(len(leftind))+len(obj)+1
                meas = vstack((meas,tab1))
                newobj = Table(np.zeros(len(leftind),dtype=np.dtype(objdt)))
                for c in ['ra','dec']: newobj[c] = tab1[leftind][c]
                newobj['nmeas'] = 1
                obj = vstack((obj,newobj))

    # Get mean ra, dec and flux from the measurements
    measindex = dln.create_index(meas['objid'])
    obj['flux'] = 0.0
    for i in range(len(measindex['value'])):
        objid = measindex['value'][i]
        ind = measindex['index'][measindex['lo'][i]:measindex['hi'][i]+1]
        nind = len(ind)
        obj['ra'][objid-1] = np.mean(meas['ra'][ind])
        obj['dec'][objid-1] = np.mean(meas['dec'][ind])
        obj['amp'][objid-1] = np.mean(meas['psfamp'][ind])
        obj['flux'][objid-1] = np.mean(meas['psfflux'][ind])

    # Impose minimum number of detections
    if mindet is not None:
        gdobj, = np.where(obj['nmeas'] >= mindet)
        if len(gdobj)==0:
            print('No objects passed the minimum number of detections threshold of '+str(mindet))
            return []
        obj = obj[gdobj]

    return obj

def initialize_catalogs(iminfo,mastertab):
    """
    Initialize the object and measurement array.

    Parameters
    ----------
    iminfo : list
       List of information on the image files.
    mastertab : table
       Master list of sources.

    Returns
    -------
    objtab : table
       Catalog of unique objects.
    meastab : table
       Catalog of individual measurements.
    objindex : list
       List of measurement indices for each object.
    iminfo : table
       List of information on the image files.  Now
        overlap information has been added.

    Example
    -------

    objtab,meastab,objindex,iminfo = initialize_catalogs(iminfo,mastertab)

    """

    nimages = len(iminfo)
    
    nobj = len(mastertab)
    objtab = mastertab.copy()
    objtab['ra'].unit = None   # don't want any units
    objtab['dec'].unit = None    
    objtab['objid'] = 0
    objtab['objid'] = np.arange(nobj)+1
    objtab['cenra0'] = objtab['ra'].copy()  # initial coordinates
    objtab['cendec0'] = objtab['dec'].copy()
    objtab['cenra'] = objtab['ra'].copy()
    objtab['cendec'] = objtab['dec'].copy()    
    objtab['cenpmra'] = 0.0
    objtab['cenpmdec'] = 0.0
    objtab['nmeas'] = 0
    objtab['converged'] = False
    objtab['chisq'] = np.nan
    
    # Initial master list coordinates
    coo0 = SkyCoord(ra=objtab['cenra']*u.deg,dec=objtab['cendec']*u.deg,frame='icrs')
    
    # Get overlap and measurement indices for the images
    meascount = 0
    for i in range(nimages):
        isin = coo0.contained_by(iminfo[i]['wcs'])
        iminfo[i]['startmeas'] = meascount
        iminfo[i]['nmeas'] = np.sum(isin)
        meascount += iminfo[i]['nmeas']
        print('Image {:d} - {:d} stars overlap'.format(i+1,np.sum(isin)))
        
    # Initialize the measurement table
    dt = [('objid',int),('objindex',int),('imindex',int),('jd',float),
          ('ra',float),('dec',float),('x',float),('y',float),('amp',float),
          ('flux',float),('fluxerr',float),('damp',float),('dx',float),
          ('dxerr',float),('dy',float),('dyerr',float),('dra',float),
          ('ddec',float),('sky',float),('chisq',float),('converged',bool)]
    nmeas = np.sum([f['nmeas'] for f in iminfo])
    #nmeas = np.sum(iminfo['nmeas'])
    meastab = Table(np.zeros(nmeas,dtype=np.dtype(dt)))
    meascount = 0
    for i in range(nimages):
        contained = coo0.contained_by(iminfo[i]['wcs'])
        isin, = np.where(contained)
        nisin = len(isin)
        objtab['nmeas'][isin] += 1 
        if nisin > 0:
            meastab['objid'][meascount:meascount+nisin] = objtab['objid'][isin]
            meastab['objindex'][meascount:meascount+nisin] = isin
            meastab['imindex'][meascount:meascount+nisin] = i
            meastab['jd'][meascount:meascount+nisin] = iminfo[i]['jd']
            meascount += nisin

    # Create object index into measurement table
    oindex = dln.create_index(meastab['objindex'])
    objindex = nobj*[[]]  # initialize index with empty list for each object
    for i in range(len(oindex['value'])):
        ind = oindex['index'][oindex['lo'][i]:oindex['hi'][i]+1]
        objindex[oindex['value'][i]] = ind    # value is the index in objtab

    return objtab,meastab,objindex,iminfo


def solveone(psf,im,cat,method='qr',bounds=None,fitradius=None,
             recenter=True,absolute=False,verbose=False):
    """
    Fit the flux and offsets in coordinates for one source.
    
    Parameters
    ----------
    psf : PSF object
       The PSF object for the image.
    im : CCDData object
       Image to use for fitting.
    cat : table
       Initial parameters.  Should at minimum have 'x' and 'y'.
    method : str, optional
       Method to use for solving the non-linear least squares problem: "cholesky",
         "qr", "svd", and "curve_fit".  Default is "qr".
    bounds : list, optional
       Input lower and upper bounds/constraints on the fitting parameters (tuple of two
         lists (e.g., ([amp_lo,x_low,y_low],[amp_hi,x_hi,y_hi])).
    fitradius : float, optional
       Fitting radius in pixels.  Default is to use the PSF FWHM.
    recenter : boolean, optional
       Allow the centroids to be fit.  Default is True.
    absolute : boolean, optional
       Input and output coordinates are in "absolute" values using the image bounding box.
         Default is False, everything is relative.
    verbose : boolean, optional
       Verbose output to the screen.  Default is False.

    Returns
    -------
    newamp : float
       New amplitude.
    dx : float
       The offset in x coordinate.
    dy : float
       The offset in y coordinate.

    Example
    -------
    
    newamp,dx,dy = solveone(psf,im,pars)

    """
    
    method = str(method).lower()
  
    # Image offset for absolute X/Y coordinates
    if absolute:
        imx0 = im.bbox.xrange[0]
        imy0 = im.bbox.yrange[0]

    xc = cat['x'][0]
    yc = cat['y'][0]
    if absolute:  # offset
        xc -= imx0
        yc -= imy0
        if bounds is not None:
            bounds[0][1] -= imx0  # lower
            bounds[0][2] -= imy0
            bounds[1][1] -= imx0  # upper
            bounds[1][2] -= imy0
    if fitradius is None:
        fitradius = np.maximum(psf.fwhm(),1)
    bbox = psf.starbbox((xc,yc),im.shape,fitradius)
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
        #amp = flux[int(np.round(yc)),int(np.round(xc))]-sky   # python images are (Y,X)
        #amp = np.maximum(amp,1)  # make sure it's not negative
        amp = 1.0
        
    initpar = [amp,xc,yc,sky]
    
    # Make bounds
    if bounds is None:
        bounds = psf.mkbounds(initpar,flux.shape)
    # Not fitting centroids
    if recenter==False:
        bounds[0][1] = initpar[1]-1e-7
        bounds[0][2] = initpar[2]-1e-7
        bounds[1][1] = initpar[1]+1e-7
        bounds[1][2] = initpar[2]+1e-7
    # Never fitting sky
    bounds[0][3] = -1e-7
    bounds[1][3] = 1e-7    
    
    maxsteps = psf.steps(initpar,bounds,star=True)  # maximum steps
    
    # Use Cholesky, QR or SVD to solve linear system of equations
    m,jac = psf.jac(xdata,*initpar,retmodel=True)
    dy = flux.ravel()-m.ravel()
    # Solve Jacobian
    dbeta = lsq.jac_solve(jac,dy,method=method,weight=wt.ravel())
    dbeta[~np.isfinite(dbeta)] = 0.0  # deal with NaNs
    
    # Perform line search
    alpha,new_dbeta = psf.linesearch(xdata,initpar,dbeta,flux,wt,m,jac) #,allpars=allpars)
                    
    # Update parameters
    oldpar = initpar.copy()
    # limit the steps to the maximum step sizes and boundaries
    newpar = psf.newpars(initpar,new_dbeta,bounds,maxsteps)                
    if recenter==False:
        # Allow the flux to fully vary
        newpar = [np.maximum(new_dbeta[0],0),initpar[1],initpar[2],0.0]
    
    # Output values
    newamp = np.maximum(newpar[0], 0)  # amp cannot be negative
    dx = newpar[1]-initpar[1]
    dy = newpar[2]-initpar[2]

    # Model with only the new flux
    newm = psf.model(xdata,*[newamp,initpar[1],initpar[2],0.0])
    newdy = flux.ravel()-newm.ravel()
    chisq = np.sum(newdy**2 * wt.ravel())/len(newdy)
    
    return newamp,dx,dy,chisq


def solve(psf,resid,tab,fitradius=None,recenter=True,verbose=False):
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
    fitradius : float, optional
       The fitting radius in pixels.  The default is 0.5*psf.fwhm().
    recenter : boolean, optional
       Allow the centroids to be fit.  Default is True.
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

    if fitradius is None:
        fitradius = 0.5*psf.fwhm()
    
    ntab = len(tab)
    out = tab.copy()
    
    # Loop over the stars
    for i in range(ntab):
        
        # Add the previous best-fit model back in to the image
        if tab['amp'][i] > 0:
            # nocopy=True will change in place
            _ = psf.add(resid.data,tab[i:i+1],nocopy=True)

        # Solve single flux
        newamp,dx,dy,chisq = solveone(psf,resid,tab[i:i+1],fitradius=fitradius,recenter=recenter)

        # the dx/dy numbers are CRAZY LARGE!!!
        
        # Save the results
        out['amp'][i] = newamp
        out['dx'][i] = dx
        out['dy'][i] = dy
        out['chisq'][i] = chisq
        
        # Immediately subtract the new model
        #  npcopy=True will change in place
        _ = psf.sub(resid.data,out[i:i+1],nocopy=True)

        #print(i,newamp,dx,dy)
        
    return out,resid


def update_object(objtab,meastab,objindex,refepoch,fitpm=False):
    """
    Determine the mean coordinates and proper motions of objects:

    Parameters
    ----------
    objtab : table
       Catalog of unique objects.
    meastab : table
       Catalog of individual measurements.  This should contain the
        updated ra/dec and dra/ddec from the last round of fitting
        sources in the images.
    objindex : list
       List of measurement indices for each object.
    refepoch : Time object
       The reference time to use.  Should be an astropy Time object.
    fitpm : boolean, optional
       Fit proper motions as well as central positions.
         Default is False.

    Returns
    -------
    objtab : table
       Catalog of unique objects with cenra/cendec and pmra/pmdec updated.

    Example
    -------

    objtab = update_object(objtab,meastab,objindex)

    """

    # Loop over object and calculate coordinate and proper motion corrections
    nobj = len(objtab)
    for i in range(nobj):
        cosdec = np.cos(np.deg2rad(objtab['cendec'][i]))
        measind = objindex[i]
        meas1 = meastab[measind]
        jd0 = np.min(meas1['jd'])
        jd = meas1['jd']-jd0
        ra = meas1['ra']+meas1['dra']     # both in degrees
        dec = meas1['dec']+meas1['ddec']
        objtab['chisq'][i] = np.mean(meas1['chisq'])
        # Fit proper motions
        if fitpm:
            # Perform linear fit
            # USE ROBUSTED WEIGHTED LINEAR FIT
            racoef = robust.linefit(jd,ra)
            deccoef = robust.linefit(jd,dec)
            # Get coordinate at the reference epoch
            refra = np.polyval(racoef,refepoch.jd-jd0)
            refdec = np.polyval(deccoef,refepoch.jd-jd0)
            # Calculate new proper motion
            pmra = racoef[0] * (3600*1e3)*365.2425      # convert slope from deg/day to mas/yr
            pmra *= cosdec                              # multiply by cos(dec) for true angle
            pmdec = deccoef[0] * (3600*1e3)*365.2425
            # update object cenra, cendec, cenpmra, cenpmdec
            objtab['cenra'][i] = refra
            objtab['cendec'][i] = refdec
            objtab['cenpmra'][i] = pmra
            objtab['cenpmdec'][i] = pmdec
        # Only fit central coordinates
        else:
            # USE ROBUST WEIGHTED MEAN
            refra = np.mean(ra)
            refdec = np.mean(dec)
            objtab['cenra'][i] = refra
            objtab['cendec'][i] = refdec            

    return objtab


def make_magnitudes(meastab,iminfo,zeropoint=25.0):
    """
    Create measurement magnitudes corrected for exposure time.

    Parameters
    ----------
    meastab : table
       Catalog of individual measurements.  This should contain the
        final fluxes.
    iminfo : list
       List of structures with information on the images including
         exposure times and filter names.
    zeropoint : float, optional
       The zero-point to add to the magnitudes.  Default is 25.0.

    Returns
    -------
    meastab : table
       Catalog of individual measurements.  This will contain a new
        "mag" and "mag_error" magnitude columns.

    Example
    -------

    meastab = make_magnitudes(meastab,iminfo)
    
    """

    # Add the new "mag" column
    meastab['mag'] = 0.0
    meastab['mag_error'] = 0.0    
    
    # Loop over the images
    nimages = len(iminfo)
    for i in range(nimages):
        exptime = iminfo[i]['exptime']
        msbeg = iminfo[i]['startmeas']
        msend = msbeg + iminfo[i]['nmeas']
        flux = meastab[msbeg:msend]['flux']
        mag = -2.5*np.log10(np.maximum(flux,1e-10))+zeropoint
        mag = +2.5*np.log10(exptime)
        flux_error = meastab[msbeg:msend]['flux_error']
        mag_error = (2.5/np.log(10))*(flux_error/flux)
        meastab[msbeg:msend]['mag'] = mag
        meastab[msbeg:msend]['mag_error'] = mag_error        

    return meastab

        
def average_photometry(objtab,meastab,objindex,iminfo):
    """
    Average photometry in each filter for unique objects

    Parameters
    ----------
    objtab : table
       Catalog of unique objects.
    meastab : table
       Catalog of individual measurements.  This should contain the
        final fluxes and "mag" from make_magnitudes().
    objindex : list
       List of measurement indices for each object.
    iminfo : list
       List of structures with information on the images including
         exposure times and filter names.

    Returns
    -------
    objtab : table
       Catalog of unique objects with photometry updated.

    Example
    -------

    objtab = average_photometry(objtab,meastab,objindex)

    """

    # Get unique filters
    imfilters = [f['filter'] for f in iminfo]
    ufilters = np.unique(imfilters)
    nufilt = len(ufilters)
    
    # Add average photometry columns to the object table
    for f in ufilters:
        objtab['mag_'+f] = np.nan
        objtab['mag_error_'+f] = np.nan        
        objtab['ndet_'+f] = 0

    # Image loop
    nobj = len(objtab)
    totalwt = np.zeros((nobj,nufilt),float)
    totalfluxwt = np.zeros((nobj,nufilt),float)    
    ndet = np.zeros((nobj,nufilt),int)
    nimages = len(iminfo)
    for i in range(nimages):
        filt = iminfo[i]['filter']
        ufiltind = np.where(ufilters==filt)[0][0]
        # Get the measurements for this image
        msbeg = iminfo[i]['startmeas']
        msend = msbeg + iminfo[i]['nmeas']
        meas1 = meastab[msbeg:msend]
        # Add up the flux for this filter for all coverage objects
        #   meastab has "objid" which we can use as an index into objtab and totalwt/totalfluxwt/ndet
        totalwt[meas1['objid']-1,ufiltind] += 1.0/meas1['mag_error']**2
        totalfluxwt[meas1['objid']-1,ufiltind] += 2.5118864**meas1['mag'] * (1.0/meas1['mag_error']**2)
        ndet[meas1['objid']-1,ufiltind] += 1

    # Now average the photometry in each unique filter
    for i,f in enumerate(ufilters):
        ind, = np.where(ndet[:,i] > 0)
        objtab['ndet_'+f] = ndet[:,i]
        if len(ind)>0:
            newflux = totalfluxwt[ind,i]/totalwt[ind,i]
            newmag = 2.5*np.log10(newflux) 
            newerr = np.sqrt(1.0/totalwt[ind,i])
            objtab['mag_'+f][ind] = newmag
            objtab['mag_error_'+f][ind] = newerr

    return objtab

def check_convergence(objtab,meastab,objindex,last_obj,fitpm=True):
    """
    Check for convergence of individual stars or all stars.

    Parameters
    ----------
    objtab : table
       Catalog of unique objects.
    meastab : table
       Catalog of individual measurements.  This should contain the
        final fluxes and "mag" from make_magnitudes().
    objindex : list
       List of measurement indices for each object.
    last_obj : table
       Object values from the last iteration.
    fitpm : boolean, optional
       Fit proper motions as well as central positions.
         Default is True.

    Results
    -------
    objtab : table
       Catalog of unique objects.
    meastab : table
       Catalog of individual measurements.  This should contain the
        final fluxes and "mag" from make_magnitudes().
    flag : int
       Flag indicating the convergence.

    Example
    -------

    objtab,meastab,flag = check_convergence(objtab,meastab,objindex)

    """

    flag = 0
    
    # Object loop
    nobj = len(objtab)
    for i in range(nobj):
        measind = objindex[i]
        meas1 = meastab[measind]

        # Check of object cenra/cendec/cenpmra/cenpmdec changed
        #  or the measurement dx/dy values
        dra = objtab['cenra'][i]-last_obj['cenra'][i]
        ddec = objtab['cendec'][i]-last_obj['cendec'][i]
        if fitpm:
            dpmra = objtab['cenpmra'][i]-last_obj['cenpmra'][i]
            dpmdec = objtab['cenpmdec'][i]-last_obj['cenpmdec'][i]

        # Measurement differences
        dx = meas1['dx']
        dy = meas1['dy']

        # Check differences in chisq values
            
    import pdb; pdb.set_trace()

    return objtab,meastab,flag

def forced(files,mastertab=None,fitpm=True,refepoch=None,refwcs=None,verbose=True):
    """
    ALLFRAME-like forced photometry.

    Parameters
    ----------
    files : list
       List of image filenames.
    mastertab : table
       Master table of objects.
    fitpm : boolean, optional
       Fit proper motions as well as central positions.
         Default is True.
    refepoch : Time object, optional
       The reference time to use.  Should be an astropy Time object.
         By default, the mean JD of all images is used.

    Returns
    -------
    obj : table
       Table of unique objects and their mean coordinates and
         proper motions.
    meas : table
       Table of individiual measurements.

    Example
    -------

    obj,meas = forced(files)

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
    
    nfiles = len(files)
    print('Running forced photometry on {:d} images'.format(nfiles))


    # Do NOT load all of the images at once, only the headers and WCS objects
    # load the PSF and object catalog from the _prometheus.fits file
    # run prometheus if the _prometheus.fits file does not exist
    
    # Load images headers and WCS
    print('Loading image headers, WCS, and catalogs')
    iminfo = getimageinfo(files)
    nimages = len(iminfo)
    if nimages==0:
        print('No images to process')
        return
    
    # Load the master star table if necessary
    if mastertab is not None:
        if isinstance(mastertab,str):
            if os.path.exists(mastertab)==False:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), mastertab)
            mastertab_filename = mastertab
            mastertab = Table.read(mastertab_filename)
    # No master star table input, make one from the
    #  individual catalogs
    else:
        print('Creating master list')
        mastertab = makemastertab(iminfo)
    nobj = len(mastertab)
    print('Master list has {:d} stars'.format(nobj))
        
    # Make sure we have the necessary columns
    for c in mastertab.colnames: mastertab[c].name = c.lower()
    if 'ra' not in mastertab.colnames or 'dec' not in mastertab.colnames:
        raise ValueError('ra and dec columns must exist in mastertab')

    # Initialize the array of positions and proper motions
    objtab,meastab,objindex,iminfo = initialize_catalogs(iminfo,mastertab)
    
    # Initial master list coordinates
    coo0 = SkyCoord(ra=objtab['cenra']*u.deg,dec=objtab['cendec']*u.deg,frame='icrs')

    # Some stars have zero measurements
    zeromeas, = np.where(objtab['nmeas']==0)
    if len(zeromeas)>0:
        print('{:d} objects have zero measurements'.format(len(zeromeas)))

    # Reference epoch
    if refepoch is not None:
        if isinstance(refepoch,Time)==False:
            raise ValueError('refepoch must be an astropy Time object')
    else:
        # Default reference epoch is mean JD
        refepoch = Time(np.mean([f['jd'] for f in iminfo]),format='jd')
        
    # Iterate until convergence has been reached
    count = 0
    flag = 0
    #lastvalues = np.zeros(len(meastab),dtype=np.dtype([('flux',float),('dx',float),('dy',float)]))
    while (flag==0):

        print('----- Iteration {:d} -----'.format(count+1))

        if count % 5 == 0:
            print('Recomputing and subtracting the sky')

        # Only fit amplitude on first iteration
        if count==0:
            recenter = False
        else:
            recenter = True

        # Save object information
        ldt = [('cenra',float),('cendec',float),('cenpmra',float),
               ('cenpmdec',float),('chisq',float)]
        last_obj = np.zeros(len(objtab),dtype=np.dtype(ldt))
                                                        
            
        # Loop over the images:
        for i in range(nimages):
            wcs = iminfo[i]['wcs']
            psf = iminfo[i]['psf']
            residfile = iminfo[i]['residfile']
            imtime = Time(iminfo[i]['jd'],format='jd')
            
            # Get the objects and measurements that overlap this image
            contained = coo0.contained_by(iminfo[i]['wcs'])
            objtab1 = objtab[contained]
            msbeg = iminfo[i]['startmeas']
            msend = msbeg + iminfo[i]['nmeas']
            meastab1 = meastab[msbeg:msend]
            print('Image {:d}  {:d} stars'.format(i+1,iminfo[i]['nmeas']))
            
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
            x,y = wcs.world_to_pixel(newcoo1)
            meastab1['x'] = x
            meastab1['y'] = y            

            # Initialize or load the residual image
            if count==0:
                im = CCDData.read(iminfo[i]['file'])
                resid = im.copy()
            else:
                #resid_data = np.load(residfile)
                #resid.data = resid_data
                reid = dln.unpickle(residfile)
                
            # Subtract sky
            if count % 5 == 0:
                if hasattr(resid,'_sky'):
                    resid._sky = None  # force it to be recomputed
                resid.data -= resid.sky

            # only fit measurements that have not converged yet
                
            # Fit the fluxes while holding the positions fixed            
            out,resid = solve(psf,resid,meastab1,recenter=recenter,verbose=verbose)

            # I THINK THIS IS SLOW BECAUSE IT IS DOING ONE STAR AT A TIME AND
            # USING A LOOP IN PYTHON.  Try to jaxify it!
            
            # Convert dx/dy to dra/ddec
            coo2 = wcs.pixel_to_world(meastab1['x']+out['dx'],meastab1['y']+out['dy'])
            dra = coo2.ra.deg - meastab1['ra']
            ddec = coo2.dec.deg - meastab1['dec']
            out['dra'] = dra    # in degrees
            out['ddec'] = ddec  # in degrees
            
            # Save the residual file
            dln.pickle(residfile,resid)
            
            # SOMETHING IS WRONG, I THINK THE PSF MIGHT NOT BE GOOD

            #from dlnpyutils import plotting as pl 
            #import pdb; pdb.set_trace()
            
            # Stuff the information back in
            meastab['damp'][msbeg:msend] = out['amp']-meastab['amp'][msbeg:msend]
            meastab['amp'][msbeg:msend] = out['amp']
            meastab['ra'][msbeg:msend] = out['ra']
            meastab['dec'][msbeg:msend] = out['dec']
            meastab['dx'][msbeg:msend] = out['dx']
            meastab['dy'][msbeg:msend] = out['dy']
            meastab['dra'][msbeg:msend] = out['dra']
            meastab['ddec'][msbeg:msend] = out['ddec']
            meastab['chisq'][msbeg:msend] = out['chisq']            
            
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

        # Update the central coordinates and proper motions
        print('Updating object coordinates and proper motions')
        objtab = update_object(objtab,meastab,objindex,refepoch,fitpm=fitpm)

        # Check for convergence
        #   individual stars can converge
        if count > 0:
            objtab,meastab,flag = check_convergence(objtab,meastab,objindex,last_obj,fitpm=fitpm)

        # Save object values 
        last_obj['cenra'] = objtab['cenra']
        last_obj['cendec'] = objtab['cendec']
        last_obj['cenpmra'] = objtab['cenpmra']
        last_obj['cenpmdec'] = objtab['cenpmdec']
        last_obj['chisq'] = objtab['chisq']        
        
        # Check for convergence
        count += 1
            
        import pdb; pdb.set_trace()


    # Create magnitudes, corrected for exposure time
    meastab = make_magnitudes(meastab,iminfo)
        
    # If there's filter information, then get average photometry
    # in each band for the objects
    objtab = average_photometry(objtab,meastab,objindex)
    
        
    return objtab,meastab

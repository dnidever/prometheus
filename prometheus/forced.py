import os
import numpy as np
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

    # Get information about all of the images
    dt = [('dateobs',str,26),('jd',float),('cenra',float),('cendec',float),
          ('nx',int),('ny',int),('vra',float,4),
          ('vdec',float,4),('exptime',float)]
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
    
    # Get the reference epoch, mean epoch
    refepoch = Time(mean(iminfo['jd']),format='jd')

    # Get the reference WCS
    
    # Initialize the array of positions and proper motions
    nobj = len(mastertab)
    outtab = mastertab.copy()
    outtab['cenra'] = 0.0
    outtab['cendec'] = 0.0
    outtab['cenpmra'] = 0.0
    outtab['cenpmdec'] = 0.0    
    fluxes = np.zeros((nobj,nimages),float)
    
    # Iterate until convergence has been reached
    count = 0
    flag = True
    while (flag):

        # Loop over the images:
        for i in range(nimages):
            im = images[i]
            imtime = Time(iminfo['jd'][i],format='jd')

            # Calculate x/y position for each object in this image
            # using the current best overall on-the-sky position
            # and proper motion
            # Need to convert celestial values to x/y position in
            # the image using the WCS.
            coo = SkyCoord(ra=outtab['cenra']*u.deg,dec=outtab['cendec']*u.deg,
                           pm_ra_cosdec=outtab['cenpmra']*u.mas/u.year,
                           pm_dec=outtab['cenpmdec']*u.mas/u.year,
                           obstime=refepoch,frame='icrs')
            # Use apply_space_motion() method to get coordinates for the
            #  epoch of this image
            newcoo = coo.apply_space_motion(imtime)
            
            # Now convert to image X/Y coordinates
            x,y = im.wcs.world_to_pixel(newcoo)
            
            # Fit the fluxes while holding the positions fixed
            dt = [('id',int),('x',float),('y',float),('amp',float),('group_id',int)]
            incat = np.zeros(nobj,dtype=np.dtype(dt))
            incat['id'] = np.arange(nobj)+1
            incat['x'] = x
            incat['y'] = y
            incat['amp'] = fluxes[:,i]
            #incat['group_id'] = ???
            out,model = allfit.fit(im.psf,im,incat,method='qr',fitradius=None,recenter=False,
                                   maxiter=10,minpercdiff=0.5,reskyiter=2,nofreeze=False,
                                   skyfit=True,verbose=False)

            # Calculate dx/dy residuals for each source

        # Calculate new coordinates and proper motions based on the x/y residuals
        

            
            import pdb; pdb.set_trace()

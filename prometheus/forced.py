import os
import numpy as np
from astropy.coordinates import SkyCoord
from . import groupfit,allfit

# ALLFRAME-like forced photometry

# also see multifit.py

def forced(images,objtab,fitpm=False):
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

    Returns
    -------

    Example
    -------

    out = forced()

    """

    # fit proper motions as well
    # don't load all the data at once, only what you need
    #   maybe use memory maps

    nimages = len(images)

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

            # Calculate x/y position for each object in this image
            # using the current best overall on-the-sky position
            # and proper motion
            # Need to convert celestial values to x/y position in
            # the image using the WCS.
            coo = SkyCoord(ra=outtab['cenra'],dec=outtab['cendec'],
                           pm_ra_cosdec=outtab['cenpmra'],pm_dec=outtab['cenpmdec'],
                           unit='degree',frame='icrs')
            # should take proper motion into account for the epoch of the observation
            x,y = im.wcs.world_to_pixel(coo)
            
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

            import pdb; pdb.set_trace()

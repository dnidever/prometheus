#!/usr/bin/env python

"""PROMETHEUS.PY - PSF photometry

"""

__authors__ = 'David Nidever <dnidever@montana.edu?'
__version__ = '20210915'  # yyyymmdd


import os
import sys
import numpy as np
import warnings
from astropy.io import fits
from astropy.table import Table
import logging
import time
from . import detection, aperture, models, getpsf, allfit, utils
from .ccddata import CCDData

# run PSF fitting on an image

def run(image,psfname='gaussian',psffitradius=None,fitradius=None,
        recenter=True,reject=False,verbose=False):
    """
    Run PSF photometry on an image.

    Parameters
    ----------
    image : string or CCDData object
      The input image to fit.  This can be the filename or CCDData object.
    psfname : string, optional
      The name of the PSF type to use.  The options are "gaussian", "moffat",
      "penny" and "gausspow".  Default is "gaussian".
    psffitradius : float, optional
       The fitting readius when constructing the PSF (in pixels).  By default
          the FWHM is used.
    fitradius: float, optional
       The fitting radius when fitting the PSF to the stars in the image (in pixels).
         By default the PSF FWHM is used.
    recenter : boolean, optional
       Allow the centroids to be fit.  Default is True.
    reject : boolean, optional
       When constructin the PSF, reject PSF stars with high RMS values.  Default is False.
    verbose : boolean, optional
      Verbose output to the screen.  Default is False.

    Returns
    -------
    cat : table
       The output table of best-fit PSF values for all of the 
    model : CCDData object
       The best-fitting model for the stars (without sky).
    sky : CCDData object
       The background sky image used for the image.
    psf : PSF object
       The best-fitting PSF model.

    Example
    -------

    cat,model,sky,psf = prometheus.run(image,psfname='gaussian',verbose=True)

    """

    start = time.time()
    
    # Load the file
    if isinstance(image,str):
        filename = image
        if verbose:
            print('Loading image from '+filename)
        image = CCDData.read(filename)
    if isinstance(image,CCDData) is False:
        raise ValueError('Input image must be a filename or CCDData object')
        
    # Processing steps
    #-----------------

    # 1) Detection
    #-------------
    if verbose:
        print('Step 1: Detection')
    objects = detection.detect(image)
    if verbose:
        print(str(len(objects))+' objects detected')
    
    # 2) Aperture photometry
    #-----------------------
    if verbose:
        print('Step 2: Aperture photometry')    
    objects = aperture.aperphot(image,objects)
    nobjects = len(objects)
    # Bright and faint limit, use 5th and 95th percentile
    minmag, maxmag = np.sort(objects['mag_auto'])[[int(np.round(0.05*nobjects)),int(np.round(0.95*nobjects))]]
    if verbose:
        print('Min/Max mag: %5.2f, %5.2f' % (minmag,maxmag))
    
    # 2) Estimate FWHM
    #-----------------
    if verbose:
        print('Step 3: Estimate FWHM')
    fwhm = utils.estimatefwhm(objects,verbose=verbose)
    
    # 3) Pick PSF stars
    #------------------
    if verbose:
        print('Step 3: Pick PSF stars')
    psfobj = utils.pickpsfstars(objects,fwhm,verbose=verbose)
    
    # 4) Construct the PSF iteratively
    #---------------------------------
    if verbose:
        print('Step 4: Construct PSF')
    # Make the initial PSF slightly elliptical so it's easier to fit the orientation
    initpsf = models.psfmodel(psfname,[fwhm/2.35,0.9*fwhm/2.35,0.0])
    psf,psfpars,psfperror,psfcat = getpsf.getpsf(initpsf,image,psfobj,fitradius=psffitradius,
                                                 reject=reject,verbose=(verbose>=2))
    if verbose:
        print('Final PSF: '+str(psf))
        gd, = np.where(psfcat['reject']==0)
        print('Median RMS:  %.4f' % np.median(psfcat['rms'][gd]))

    # 5) Run on all sources
    #----------------------
    if verbose:
        print('Step 5: Get PSF photometry for all objects')
    psfout,model,sky = allfit.fit(psf,image,objects,fitradius=fitradius,recenter=recenter,verbose=(verbose>=2))

    # Combine aperture and PSF columns
    out = objects.copy()
    for n in psfout.columns:
        out[n] = psfout[n]
    
    if verbose:
        print('dt = ',time.time()-start)              
    
    return out,model,sky,psf

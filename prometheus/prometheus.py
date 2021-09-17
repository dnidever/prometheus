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

def run(image,verbose=False):
    """ Run PSF photometry on an image."""

    # Load the file
    if isinstance(image,str):
        if verbose:
            print('Loading image from '+filename)
        filename = image
        image = CCDData.read(filename)

    # Processing steps
    #-----------------

    # 1) Detection
    if verbose:
        print('Step 1: Detection')
    objects = detection.detect(image)
    if verbose:
        print(str(len(objects))+' objects detected')
    
    # 2) Aperture photometry
    if verbose:
        print('Step 2: Aperture photometry')    
    objects = aperture.aperphot(image,objects)
        
    # 2) Estimate FWHM
    if verbose:
        print('Step 3: Estimate FWHM')
    fwhm = utils.estimatefwhm(objects)
    #if verbose:
    #    print('FWHM = %10.3f' % fwhm)
    
    # 3) Pick PSF stars
    if verbose:
        print('Step 3: Pick PSF stars')
    psfobj = utils.pickpsfstars(objects,fwhm)
    #if verbose:
    #    print(str(len(psfobj))+' PSF stars found')
    
    # 4) Construct the PSF iteratively
    if verbose:
        print('Step 4: Construct PSF')
    initpsf = models.PSFGaussian([fwhm,fwhm,0.0])
    psf = getpsf.getpsf(initpsf,image,psfobj)

    import pdb; pdb.set_trace()
    
    # 5) Run on all sources
    if verbose:
        print('Step 5: Get PSF photometry for all objects')
    out = allfit.fit(psf,image,objects)

    return out

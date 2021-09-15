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
from . import detection, aperture, models, getpsf, allfit, psf
from .ccddata import CCDData

# run PSF fitting on an image

def run(image):
    """ Run PSF photometry on an image."""

    # Load the file
    if isinstance(image,str):
        filename = image
        image = CCDData.read(filename)

    # Processing steps
    #-----------------

    # 1) Detection
    objects = detection.detect(image)

    # 2) Photometry
    objects = aperture.aperphot(image,objects)
    
    # 2) Estimate FWHM
    fwhm = psf.estimatefwhm(objects)

    # 3) Pick PSF stars
    psfobj = psf.psfstars(objects,image)

    # 4) Construct the PSF iteratively
    initpsf = models.PSFGaussian([fwhm,fwhm,0.0])
    psf = getpsf.getpsf(initpsf,image,psfobj)
    
    # 5) Run on all sources
    out = allfit.fit(image,psf,objects)

    return out

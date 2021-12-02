#!/usr/bin/env python

"""SYNTH.PY - Make synthetic star catalogs and images

"""

__authors__ = 'David Nidever <dnidever@montana.edu?'
__version__ = '20210821'  # yyyymmdd


import os
import sys
import numpy as np
import warnings
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
from scipy.optimize import curve_fit, least_squares
from scipy.interpolate import interp1d
from dlnpyutils import utils as dln, bindata
import copy
import logging
import time
import matplotlib
from astropy.nddata import CCDData as CCD,StdDevUncertainty
from .ccddata import CCDData
from . import models

# Fit a PSF model to multiple stars in an image

def makecat(nstars=1000,ampr=[100.0,1e5],xr=[50.0,950.0],yr=[50.0,950.0]):
    """
    Make synthetic catalog of stars.
    """
    dt = np.dtype([('id',int),('x',float),('y',float),('amp',float)])
    cat = np.zeros(nstars,dtype=dt)
    for i in range(nstars):
        amp = dln.scale(np.random.rand(1)[0],[0.0,1.0],ampr)
        xcen = dln.scale(np.random.rand(1)[0],[0.0,1.0],xr)
        ycen = dln.scale(np.random.rand(1)[0],[0.0,1.0],yr)
        cat['id'][i] = i+1
        cat['x'][i] = xcen
        cat['y'][i] = ycen
        cat['amp'][i] = amp
    return cat

def makeimage(nstars=1000,nx=1024,ny=1024,psf=None,cat=None,noise=True,backgrnd=1000.0):
    """
    Make synthetic image
    """

    if psf is None:
        npix = 51
        pix = np.arange(npix)
        pars = [1.0,npix//2,npix/2,3.5,3.6,np.deg2rad(30.)] 
        psf = models.PSFGaussian(pars[3:])   
    fwhm = psf.fwhm()
    npsfpix = psf.npix
    if cat is None:
        cat = makecat(nstars=nstars,xr=[npsfpix/2.0+10,nx-npsfpix/2.0-10],
                      yr=[npsfpix/2.0+10,ny-npsfpix/2.0-10])
    else:
        nstars = len(cat)

    # python images are (Y,X)
    im = np.zeros((ny,nx),float)+backgrnd
    for i in range(nstars):
        amp = cat['amp'][i]
        xcen = cat['x'][i]
        ycen = cat['y'][i]
        bbox = psf.starbbox((xcen,ycen),(ny,nx),npsfpix//2)
        im2 = psf(pars=[amp,xcen,ycen],bbox=bbox)
        im[bbox.slices] += im2
    err = np.maximum(np.sqrt(im),1)
    if noise:
        im += err*np.random.randn(*im.shape)

    image = CCDData(im,err,unit='adu')
    
    return image

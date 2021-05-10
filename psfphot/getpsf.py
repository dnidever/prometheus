#!/usr/bin/env python

"""GETPSF.PY - Determine the PSF by fitting to multiple stars in an image

"""

__authors__ = 'David Nidever <dnidever@montana.edu?'
__version__ = '20210430'  # yyyymmdd


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

# Fit a PSF model to multiple stars in an image




def getpsf(psfmodel,image,cat):
    """ PSF model, image object, catalog of sources to fit."""

    pass

def curvefit_psf(func,*args,**kwargs):
    """ Thin wrapper around curve_fit for PSFs."""
    def wrap_psf(xdata,pars,*args,**kwargs):
        return psf(xdata[0],xdata[1],pars,*args,**kwargs)
    return curve_fit(wrap_psf,*args,**kwargs)

def fitstar(im,cat,psf):
    """ Fit a PSF model to a star in an image."""

    # IM should be an image with an uncertainty array as well
    nx,ny = im.data
    
    # Estimate sky
    xc = cat['X'][0]
    yc = cat['Y'][0]
    x0 = np.maximum(0,xc-20)
    x1 = np.minimum(xc-20,nx-1)
    y0 = np.maximum(0,yc-20)
    y1 = np.minium(yc-20,ny-1)
    
    flux = im.data[x0:x1+1,y0:y1+1]
    err = im.uncertainty[x0:x1+1,y0:y1+1]    
    sky = np.median(im.data[x0:x1+1,y0:y1+1])
    height = im.data[int(np.round(xc)),int(np.round(yc))]-sky

    nX = x1-x0+1
    nY = y1-y0+1
    X = np.repeat(np.arange(x0,x1+1),nY).reshape(nX,nY)
    Y = np.repeat(np.arange(y0,y1+1),nX).reshape(nY,nX).T
    xdata = np.vstack((X.ravel(), Y.ravel()))

    initpar = []
    pars,cov = curvefit_psf(psf,xdata,flux.ravel(),sigma=err.ravel(),p0=initpar,bounds=bounds)

    
    import pdb; pdb.set_trace()

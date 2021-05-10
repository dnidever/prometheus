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
    def wrap_psf(xdata,*args2,**kwargs2):
        ## curve_fit separates each parameter while
        ## psf expects on pars array
        pars = args2
        print(pars)
        return func(xdata[0],xdata[1],pars,**kwargs2)
    return curve_fit(wrap_psf,*args,**kwargs)

def curvefit_psfallpars(func,*args,**kwargs):
    """ Thin wrapper around curve_fit for PSFs and fitting ALL parameters."""
    def wrap_psf(xdata,*args2,**kwargs2):
        ## curve_fit separates each parameter while
        ## psf expects on pars array
        allpars = args2
        print(allpars)
        nmpars = len(func.params)
        mpars = allpars[-nmpars:]
        pars = allpars[0:-nmpars]
        return func(xdata[0],xdata[1],pars,mpars=mpars,**kwargs2)
    return curve_fit(wrap_psf,*args,**kwargs)


def fitstar(im,cat,psf):
    """ Fit a PSF model to a star in an image."""

    # IM should be an image with an uncertainty array as well
    nx,ny = im.data.shape


    ### ADD A FITTING RADIUS !!!!
    
    xc = cat['X']
    yc = cat['Y']
    box = 20
    x0 = int(np.maximum(0,np.floor(xc-box)))
    x1 = int(np.minimum(np.ceil(xc+box),nx-1))
    y0 = int(np.maximum(0,np.floor(yc-box)))
    y1 = int(np.minimum(np.ceil(yc+box),ny-1))
    
    flux = im.data[x0:x1+1,y0:y1+1]
    err = im.uncertainty.array[x0:x1+1,y0:y1+1]
    sky = np.median(im.data[x0:x1+1,y0:y1+1])
    height = im.data[int(np.round(xc)),int(np.round(yc))]-sky

    nX = x1-x0+1
    nY = y1-y0+1
    X = np.repeat(np.arange(x0,x1+1),nY).reshape(nX,nY)
    Y = np.repeat(np.arange(y0,y1+1),nX).reshape(nY,nX).T
    xdata = np.vstack((X.ravel(), Y.ravel()))

    #import pdb; pdb.set_trace()

    # Just fit height, xc, yc, sky
    #initpar = [height,xc,yc,sky]
    #bounds = (-np.inf,np.inf)
    #pars,cov = curvefit_psf(psf,xdata,flux.ravel(),sigma=err.ravel(),p0=initpar) #,bounds=bounds)
    #return pars,cov

    # Fit all parameters
    initpar = np.hstack(([height,xc,yc,sky],psf.params.copy()))
    #bounds = (-np.inf,np.inf)
    allpars,cov = curvefit_psfallpars(psf,xdata,flux.ravel(),sigma=err.ravel(),p0=initpar) #,bounds=bounds)

    bpsf = psf.copy()
    bpsf.params = allpars[4:]
    pars = allpars[0:4]
    bmodel = bpsf(X,Y,pars)
    
    import pdb; pdb.set_trace()
    
    return pars,cov,bpsf
    


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
from astropy.nddata import CCDData,StdDevUncertainty
from . import models

# Fit a PSF model to multiple stars in an image

def makecat(nstars=1000,heightr=[100.0,1e5],xr=[50.0,950.0],yr=[50.0,950.0]):
    """
    Make synthetic catalog of stars.
    """
    dt = np.dtype([('id',int),('x',float),('y',float),('height',float)])
    cat = np.zeros(nstars,dtype=dt)
    for i in range(nstars):
        height = dln.scale(np.random.rand(1)[0],[0.0,1.0],heightr)
        xcen = dln.scale(np.random.rand(1)[0],[0.0,1.0],xr)
        ycen = dln.scale(np.random.rand(1)[0],[0.0,1.0],yr)
        cat['id'][i] = i+1
        cat['x'][i] = xcen
        cat['y'][i] = ycen
        cat['height'][i] = height
    return cat

def makeimage(nstars=1000,nx=1024,ny=1024,psf=None,cat=None,backgrnd=1000.0):
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

    im = np.zeros((nx,ny),float)+backgrnd
    im += np.sqrt(backgrnd)*np.random.rand(nx,ny)
    for i in range(nstars):
        height = cat['height'][i]
        xcen = cat['x'][i]
        ycen = cat['y'][i]
        xlo = int(np.round(xcen)-npsfpix/2)
        xhi = int(np.round(xcen)+npsfpix/2)-1
        ylo = int(np.round(ycen)-npsfpix/2)
        yhi = int(np.round(ycen)+npsfpix/2)-1
        im2 = psf(pars=[height,xcen,ycen],xy=[[xlo,xhi],[ylo,yhi]])
        im2noise = im2+np.maximum(np.sqrt(im2),1)*np.random.rand(*im2.shape)
        im[xlo:xhi+1,ylo:yhi+1]+=im2noise    
    err = np.sqrt(im)
    image = CCDData(im,StdDevUncertainty(err),unit='adu')

    return image


def getpsf(psfmodel,image,cat):
    """ PSF model, image object, catalog of sources to fit."""

    pass

def curvefit_psf(func,*args,**kwargs):
    """ Thin wrapper around curve_fit for PSFs."""
    def wrap_psf(xdata,*args2,**kwargs2):
        ## curve_fit separates each parameter while
        ## psf expects a pars array
        pars = args2
        print(pars)
        return func(xdata[0],xdata[1],pars,**kwargs2)
    return curve_fit(wrap_psf,*args,**kwargs)

def curvefit_psfallpars(func,*args,**kwargs):
    """ Thin wrapper around curve_fit for PSFs and fitting ALL parameters."""
    def wrap_psf(xdata,*args2,**kwargs2):
        ## curve_fit separates each parameter while
        ## psf expects a pars array
        allpars = args2
        print(allpars)
        nmpars = len(func.params)
        mpars = allpars[-nmpars:]
        pars = allpars[0:-nmpars]
        return func(xdata[0],xdata[1],pars,mpars=mpars,**kwargs2)
    return curve_fit(wrap_psf,*args,**kwargs)


def fitstar(im,cat,psf,radius=None,allpars=False):
    """ Fit a PSF model to a star in an image."""

    # IM should be an image with an uncertainty array as well
    nx,ny = im.data.shape

    xc = cat['X']
    yc = cat['Y']
    # use FWHM of PSF for the fitting radius
    #box = 20
    if radius is None:
        radius = psf.fwhm()
    x0 = int(np.maximum(0,np.floor(xc-radius)))
    x1 = int(np.minimum(np.ceil(xc+radius),nx-1))
    y0 = int(np.maximum(0,np.floor(yc-radius)))
    y1 = int(np.minimum(np.ceil(yc+radius),ny-1))
    
    flux = im.data[x0:x1+1,y0:y1+1]
    err = im.uncertainty.array[x0:x1+1,y0:y1+1]
    sky = np.median(im.data[x0:x1+1,y0:y1+1])
    height = im.data[int(np.round(xc)),int(np.round(yc))]-sky

    nX = x1-x0+1
    nY = y1-y0+1
    X = np.arange(x0,x1+1).reshape(-1,1)+np.zeros(nY)   # broadcasting is faster
    Y = np.arange(y0,y1+1).reshape(1,-11)+np.zeros(nX).reshape(-1,1)
    #X = np.repeat(np.arange(x0,x1+1),nY).reshape(nX,nY)
    #Y = np.repeat(np.arange(y0,y1+1),nX).reshape(nY,nX).T
    xdata = np.vstack((X.ravel(), Y.ravel()))

    #import pdb; pdb.set_trace()

    # Just fit height, xc, yc, sky
    if allpars==False:
        initpar = [height,xc,yc,sky]
        bounds = (-np.inf,np.inf)
        pars,cov = curve_fit(psf.model,xdata,flux.ravel(),sigma=err.ravel(),p0=initpar,jac=psf.jac)
        perror = np.sqrt(np.diag(cov))
        #pars,cov = curve_fit(psf.model,xdata,flux.ravel(),sigma=err.ravel(),p0=initpar) #,bounds=bounds)    
        #pars,cov = curvefit_psf(psf,xdata,flux.ravel(),sigma=err.ravel(),p0=initpar) #,bounds=bounds)    
        return pars,perror
    
    # Fit all parameters
    else:
        initpar = np.hstack(([height,xc,yc,sky],psf.params.copy()))
        bounds = (np.zeros(len(initpar),float)-np.inf,np.zeros(len(initpar),float)+np.inf)
        allpars,cov = curve_fit(psf.modelall,xdata,flux.ravel(),sigma=err.ravel(),p0=initpar,jac=psf.jacall)
        perror = np.sqrt(np.diag(cov))
        
        #bounds = (-np.inf,np.inf)
        #allpars,cov = curvefit_psfallpars(psf,xdata,flux.ravel(),sigma=err.ravel(),p0=initpar) #,bounds=bounds)

        bpsf = psf.copy()
        bpsf.params = allpars[4:]
        pars = allpars[0:4]
        bmodel = bpsf(X,Y,pars)
    
        return pars,cov,bpsf

    import pdb; pdb.set_trace()


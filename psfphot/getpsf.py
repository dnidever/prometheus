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


class PSFFitter(object):

    def __init__(self,psf,image,cat,fitradius=None,verbose=False):
        self.verbose = verbose
        self.psf = psf
        self.image = image
        self.cat = cat
        self.nstars = len(cat)
        self.niter = 0
        self.npsfpix = psf.npix
        nx,ny = image.data.shape
        self.nx = nx
        self.ny = ny
        if fitradius is None:
            fitradius = psf.fwhm()
        self.fitradius = fitradius
        self.nfitpix = int(np.ceil(fitradius))  # +/- nfitpix
        self.starheight = cat['height'].copy()
        self.starxcen = cat['x'].copy()
        self.starycen = cat['y'].copy()
        
        # Get xdata, ydata, error
        imdata = []
        errdata = []
        xydata = []
        ntotpix = 0
        for i in range(self.nstars):
            xcen = cat['x'][i]
            ycen = cat['y'][i]
            xlo = np.maximum(int(np.round(xcen)-self.nfitpix),0)
            xhi = np.minimum(int(np.round(xcen)+self.nfitpix),nx-1)
            ylo = np.maximum(int(np.round(ycen)-self.nfitpix),0)
            yhi = np.minimum(int(np.round(ycen)+self.nfitpix),ny-1)
            im = image.data[xlo:xhi,ylo:yhi]
            err = image.uncertainty.array[xlo:xhi,ylo:yhi]
            ntotpix += im.size            
            imdata.append(im)
            errdata.append(err)
            xydata.append([[xlo,xhi-1],[ylo,yhi-1]])
 
        self.ntotpix = ntotpix
        self.imdata = imdata
        self.errdata = errdata
        self.xydata = xydata
        # flatten the image and error arrays
        imflatten = np.zeros(ntotpix,float)
        errflatten = np.zeros(ntotpix,float)
        count = 0
        for i in range(self.nstars):
            npix = imdata[i].size
            xcen = cat['x'][i]
            ycen = cat['y'][i]
            im = imdata[i].copy()
            err = errdata[i].copy()
            xy = xydata[i]
            # Zero-out anything beyond the fitting radius
            x = np.arange(xy[0][0],xy[0][1]+1).astype(float)
            y = np.arange(xy[1][0],xy[1][1]+1).astype(float)
            rr = np.sqrt( (x-xcen).reshape(-1,1)**2 + (y-ycen).reshape(1,-1)**2 )
            im[rr>self.fitradius] = 0.0
            err[rr>self.fitradius] = 1e30
            imflatten[count:count+npix] = im.flatten()
            errflatten[count:count+npix] = err.flatten()
            count += npix
        self.imflatten = imflatten
        self.errflatten = errflatten

        
    def model(self,x,*args):
        """ model function."""

        if self.verbose:
            print(self.niter,args)
        
        psf = self.psf.copy()
        psf._params = list(args)
        # Loop over the stars and generate the model image
        out = np.zeros(self.ntotpix,float)
        count = 0
        for i in range(self.nstars):
            height = self.starheight[i]
            xcen = self.starxcen[i]            
            ycen = self.starycen[i]
            xy = self.xydata[i]
            im = psf(pars=[height,xcen,ycen],xy=xy)
            # Zero-out anything beyond the fitting radius
            x = np.arange(xy[0][0],xy[0][1]+1).astype(float)
            y = np.arange(xy[1][0],xy[1][1]+1).astype(float)
            rr = np.sqrt( (x-xcen).reshape(-1,1)**2 + (y-ycen).reshape(1,-1)**2 )
            im[rr>self.fitradius] = 0.0
            npix = im.size
            out[count:count+npix] = im.flatten()
            count += npix

        self.niter += 1
            
        return out

    
def getpsf(psf,image,cat,verbose=False):
    """ PSF model, image object, catalog of sources to fit."""

    # Subtract the background
    backgrnd = np.median(image.data)
    image0 = image.copy()
    image.data -= backgrnd
    
    psffitter = PSFFitter(psf,image,cat,verbose=verbose)
    xdata = np.arange(psffitter.ntotpix)
    #initpar = psf.params.copy()
    initpar = [3.0,4.0,0.1]

    # Perform the fitting
    pars,cov = curve_fit(psffitter.model,xdata,psffitter.imflatten,
                         sigma=psffitter.errflatten,p0=initpar) # jac=psffitter.jac
    perror = np.sqrt(np.diag(cov))

    if verbose:
        print('Best-fitting parameters: ',pars)
        print('Errors: ',perror)
    
    # create the best-fitting PSf
    newpsf = psf.copy()
    newpsf._params = pars

    return newpsf, pars, perror


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


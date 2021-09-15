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
#from astropy.nddata import CCDData,StdDevUncertainty
from dlnpyutils import utils as dln, bindata
import copy
import logging
import time
import matplotlib
import sep

# Fit a PSF model to multiple stars in an image


class PSFFitter(object):

    def __init__(self,psf,image,cat,fitradius=None,verbose=False):
        self.verbose = verbose
        self.psf = psf
        self.image = image
        self.cat = cat
        self.nstars = np.size(cat)
        self.niter = 0
        self.npsfpix = psf.npix
        nx,ny = image.data.shape
        self.nx = nx
        self.ny = ny
        if fitradius is None:
            fitradius = psf.fwhm()
        self.fitradius = fitradius
        self.nfitpix = int(np.ceil(fitradius))  # +/- nfitpix
        self.starheight = np.zeros(self.nstars,float)
        self.starheight[:] = cat['height'].copy()
        self.starxcen = np.zeros(self.nstars,float)
        self.starxcen[:] = cat['x'].copy()
        self.starycen = np.zeros(self.nstars,float)
        self.starycen[:] = cat['y'].copy()
        
        # Get xdata, ydata, error
        imdata = []
        xydata = []
        ntotpix = 0
        for i in range(self.nstars):
            xcen = self.starxcen[i]
            ycen = self.starycen[i]
            xlo = np.maximum(int(np.round(xcen)-self.nfitpix),0)
            xhi = np.minimum(int(np.round(xcen)+self.nfitpix),nx-1)
            ylo = np.maximum(int(np.round(ycen)-self.nfitpix),0)
            yhi = np.minimum(int(np.round(ycen)+self.nfitpix),ny-1)
            im = image[xlo:xhi,ylo:yhi]
            ntotpix += im.size            
            imdata.append(im)
            xydata.append([[xlo,xhi-1],[ylo,yhi-1]])
 
        self.ntotpix = ntotpix
        self.imdata = imdata
        self.xydata = xydata
        # flatten the image and error arrays
        imflatten = np.zeros(ntotpix,float)
        errflatten = np.zeros(ntotpix,float)
        count = 0
        for i in range(self.nstars):
            npix = imdata[i].size
            xcen = self.starxcen[i]
            ycen = self.starycen[i]
            im = imdata[i].data.copy()
            err = imdata[i].error.copy()
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
            print('model: ',self.niter,args)
        
        psf = self.psf.copy()
        psf._params = list(args)
        # Loop over the stars and generate the model image
        allim = np.zeros(self.ntotpix,float)
        pixcnt = 0
        for i in range(self.nstars):
            image = self.imdata[i]
            height = self.starheight[i]
            xcen = self.starxcen[i]            
            ycen = self.starycen[i]
            xy = self.xydata[i]
            x = np.arange(xy[0][0],xy[0][1]+1).astype(float)
            y = np.arange(xy[1][0],xy[1][1]+1).astype(float)
            rr = np.sqrt( (x-xcen).reshape(-1,1)**2 + (y-ycen).reshape(1,-1)**2 )
            mask = rr>self.fitradius

            x0 = xcen-xy[0][0]
            y0 = ycen-xy[1][0]
            
            # Fit height/xcen/ycen if niter=1
            if self.niter<=1:
                pars = psf.fit(image,[height,x0,y0],nosky=True)
                xcen += (pars[1]-x0)
                ycen += (pars[2]-y0)
                height = pars[0]
                self.starheight[i] = height
                self.starxcen[i] = xcen
                self.starycen[i] = ycen                
            # Only fit height if niter>1
            #   do it empirically
            else:
                im1 = psf(pars=[1.0,xcen,ycen],xy=xy)
                wt = 1/image.error**2
                height = np.median(image.data[mask]/im1[mask])
                pars2 = psf.fit(image,[height,x0,y0],nosky=True)
                height = pars2[0]
                
                self.starheight[i] = height
                #self.starxcen[i] = pars2[1]+xy[0][0]
                #self.starycen[i] = pars2[2]+xy[1][0]       
                #print(count,self.starxcen[i],self.starycen[i])
                # updating the X/Y values after the first iteration
                #  causes problems.  bounces around too much
                
            im = psf(pars=[height,xcen,ycen],xy=xy)
            # Zero-out anything beyond the fitting radius
            im[mask] = 0.0
            npix = im.size
            allim[pixcnt:pixcnt+npix] = im.flatten()
            pixcnt += npix
            
        self.niter += 1
            
        return allim

    
    def jac(self,x,*args,retmodel=False):
        """ jacobian."""

        if self.verbose:
            print('jac: ',self.niter,args)
        
        psf = self.psf.copy()
        psf._params = list(args)
        # Loop over the stars and generate the derivatives
        allderiv = np.zeros((self.ntotpix,len(psf.params)),float)
        if retmodel:
            allim = np.zeros(self.ntotpix,float)
        pixcnt = 0
        for i in range(self.nstars):
            height = self.starheight[i]
            xcen = self.starxcen[i]            
            ycen = self.starycen[i]
            xy = self.xydata[i]
            x2,y2 = psf.xylim2xy(xy)
            xdata = np.vstack((x2.ravel(),y2.ravel()))
            allpars = np.concatenate((np.array([height,xcen,ycen]),np.array(args)))
            if retmodel:
                m,deriv = psf.jac(xdata,*allpars,allpars=True,retmodel=True)
            else:
                deriv = psf.jac(xdata,*allpars,allpars=True)                
            deriv = np.delete(deriv,[0,1,2],axis=1)  # remove stellar ht/xc/yc columns
            npix,dum = deriv.shape
            allderiv[pixcnt:pixcnt+npix,:] = deriv
            if retmodel:
                allim[pixcnt:pixcnt+npix] = m
            pixcnt += npix
        if retmodel:
            return allim,allderiv
        else:
            return allderiv
    
    
def getpsf(psf,image,cat,method='qr',maxiter=10,minpercdiff=1.0,verbose=False):
    """
    Fit PSF model to stars in an image.

    Parameters
    ----------
    psf : PSF object
       PSF object with initial parameters to use.
    image : CCDData object
       Image to use to fit PSF model to stars.
    cat : table
       Catalog with initial height/x/y values for the stars to use to fit the PSF.
    method : str, optional
       Method to use for solving the non-linear least squares problem: "qr",
       "svd", and "curve_fit".  Default is "qr".
    maxiter : int, optional
       Maximum number of iterations to allow.  Only for methods "qr" or "svd".
       Default is 10.
    minpercdiff : float, optional
       Minimum percent change in the parameters to allow until the solution is
       considered converged and the iteration loop is stopped.  Only for methods
       "qr" and "svd".  Default is 1.0.
    verbose : boolean, optional
       Verbose output.

    Returns
    -------
    pars : numpy array
       Array of best-fit model parameters
    perror : numpy array
       Uncertainties in "pars".
    newpsf : PSF object
       New PSF object with the best-fit model parameters.

    Example
    -------

    pars,perror,newpsf = getpsf(psf,image,cat)

    """


    # MAKE SURE THE PSF STARS AREN'T TOO CLOSE TO EACH OTHER!!!
    
    nx,ny = image.data.shape
    
    # Get the background using SEP
    bkg = sep.Background(image.data, bw=int(nx/10), bh=int(ny/10), fw=3, fh=3)
    bkg_image = bkg.back()
    
    # Subtract the background
    image0 = image.copy()
    image.data -= bkg_image
    
    psffitter = PSFFitter(psf,image,cat,verbose=verbose)
    xdata = np.arange(psffitter.ntotpix)
    initpar = psf.params.copy()
    
    # Iterate
    count = 0
    percdiff = 1e10
    bestpar = initpar.copy()
    while (count<maxiter and percdiff>minpercdiff):
        m,jac = psffitter.jac(xdata,*bestpar,retmodel=True)
        dy = psffitter.imflatten-m
        # QR decomposition
        if str(method).lower()=='qr':
            q,r = np.linalg.qr(jac)
            rinv = np.linalg.inv(r)
            dbeta = rinv @ (q.T @ dy)
        # SVD:
        elif str(method).lower()=='svd':
            u,s,vt = np.linalg.svd(jac)
            # u: [Npix,Npix]
            # s: [Npars]
            # vt: [Npars,Npars]
            # dy: [Npix]
            sinv = s.copy()*0  # pseudo-inverse
            sinv[s!=0] = 1/s[s!=0]
            npars = len(s)
            dbeta = vt.T @ ((u.T @ dy)[0:npars]*sinv)
        # Curve_fit
        elif str(method).lower()=='curve_fit':
            # Perform the fitting
            bestpar,cov = curve_fit(psffitter.model,xdata,psffitter.imflatten,
                                     sigma=psffitter.errflatten,p0=bestpar,jac=psffitter.jac)
            perror = np.sqrt(np.diag(cov))
            break
        else:
            raise ValueError('Only SVD or QR methods currently supported')
            
        oldpar = bestpar.copy()
        bestpar += dbeta
        diff = np.abs(bestpar-oldpar)
        percdiff = np.max(diff/oldpar*100)
        perror = diff  # rough estimate
        count += 1

        if verbose:
            print(count,bestpar,percdiff)

    pars = bestpar
    if verbose:
        print('Best-fitting parameters: ',pars)
        print('Errors: ',perror)
    
    # create the best-fitting PSF
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

    xc = cat['x']
    yc = cat['y']
    # use FWHM of PSF for the fitting radius
    #box = 20
    if radius is None:
        radius = psf.fwhm()
    x0 = int(np.maximum(0,np.floor(xc-radius)))
    x1 = int(np.minimum(np.ceil(xc+radius),nx-1))
    y0 = int(np.maximum(0,np.floor(yc-radius)))
    y1 = int(np.minimum(np.ceil(yc+radius),ny-1))
    
    flux = im.data[x0:x1+1,y0:y1+1]
    err = im.error[x0:x1+1,y0:y1+1]
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


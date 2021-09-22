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
from . import leastsquares as lsq

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
        if 'height' in cat.colnames:
            self.starheight[:] = cat['height'].copy()
        else:
            # estimate height from flux and fwhm
            # area under 2D Gaussian is 2*pi*A*sigx*sigy
            height = cat['flux']/(2*np.pi*(cat['fwhm']/2.35)**2)
            self.starheight[:] = np.maximum(height,0)   # make sure it's positive
        self.starxcen = np.zeros(self.nstars,float)
        self.starxcen[:] = cat['x'].copy()
        self.starycen = np.zeros(self.nstars,float)
        self.starycen[:] = cat['y'].copy()
        
        # Get xdata, ydata, error
        imdata = []
        bboxdata = []
        npixdata = []
        xlist = []
        ylist = []
        pixstart = []
        imflatten = np.zeros(self.nstars*(2*self.nfitpix+1)**2,float)
        errflatten = np.zeros(self.nstars*(2*self.nfitpix+1)**2,float)
        count = 0
        for i in range(self.nstars):
            xcen = self.starxcen[i]
            ycen = self.starycen[i]
            bbox = psf.starbbox((xcen,ycen),image.shape,radius=self.nfitpix)
            im = image[bbox.slices]
            flux = image.data[bbox.slices]-image.sky[bbox.slices]
            err = image.data[bbox.slices]            
            imdata.append(im)
            bboxdata.append(bbox)
            # Trim to only the pixels that we want to fix
            #flux = im.data.copy()-im.sky.copy()
            #err = im.error.copy()
            # Zero-out anything beyond the fitting radius
            x,y = psf.bbox2xy(bbox)
            rr = np.sqrt( (x-xcen)**2 + (y-ycen)**2 )
            # Use image mask
            #  mask=True for bad values
            if image.mask is not None:           
                gdmask = (rr<=self.fitradius) & (image.mask[y,x]==False)
            else:
                gdmask = rr<=self.fitradius                
            x = x[gdmask]  # raveled
            y = y[gdmask]
            flux = flux[gdmask]
            err = err[gdmask]
            npix = len(flux)
            imflatten[count:count+npix] = flux
            errflatten[count:count+npix] = err
            pixstart.append(count)
            xlist.append(x)
            ylist.append(y)
            npixdata.append(npix)
            count += npix

        #self.imdata = imdata
        self.bboxdata = bboxdata            
        imflatten = imflatten[0:count]
        errflatten = errflatten[0:count]
        self.imflatten = imflatten
        self.errflatten = errflatten
        self.ntotpix = count
        self.xlist = xlist
        self.ylist = ylist
        self.npix = npixdata
        self.pixstart = pixstart
        self.imdata = imdata

        
    def model(self,x,*args,refit=True):
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
            bbox = self.bboxdata[i]
            x = self.xlist[i]
            y = self.ylist[i]
            pixstart = self.pixstart[i]
            npix = self.npix[i]
            flux = self.imflatten[pixstart:pixstart+npix]
            err = self.errflatten[pixstart:pixstart+npix]
            
            #xy = self.xydata[i]
            #x = np.arange(xy[0][0],xy[0][1]+1).astype(float)
            #y = np.arange(xy[1][0],xy[1][1]+1).astype(float)
            #rr = np.sqrt( (x-xcen).reshape(-1,1)**2 + (y-ycen).reshape(1,-1)**2 )
            #mask = rr>self.fitradius

            #x0 = xcen-xy[0][0]
            #y0 = ycen-xy[1][0]
            x0 = xcen - bbox.ixmin
            y0 = ycen - bbox.iymin

            #import pdb; pdb.set_trace()
            
            # Fit height/xcen/ycen if niter=1
            if refit:
                if self.niter<=1:
                    pars,perror,model = psf.fit(image,[height,x0,y0],nosky=True,retpararray=True)
                    xcen += (pars[1]-x0)
                    ycen += (pars[2]-y0)
                    height = pars[0]
                    self.starheight[i] = height
                    self.starxcen[i] = xcen
                    self.starycen[i] = ycen
                    model = psf(x,y,pars=[height,xcen,ycen])                
                # Only fit height if niter>1
                #   do it empirically
                else:
                    #im1 = psf(pars=[1.0,xcen,ycen],bbox=bbox)
                    #wt = 1/image.error**2
                    #height = np.median(image.data[mask]/im1[mask])                
                    model1 = psf(x,y,pars=[1.0,xcen,ycen])
                    wt = 1/err**2
                    height = np.median(flux/model1)
                    #height = np.median(wt*flux/model1)/np.median(wt)
                    #pars2,perror2,model2 = psf.fit(image,[height,x0,y0],nosky=True,retpararray=True)
                    #height = pars2[0]
                
                    self.starheight[i] = height
                    model = model1*height
                    #self.starxcen[i] = pars2[1]+xy[0][0]
                    #self.starycen[i] = pars2[2]+xy[1][0]       
                    #print(count,self.starxcen[i],self.starycen[i])
                    # updating the X/Y values after the first iteration
                    #  causes problems.  bounces around too much
                    
                    if i==1: print(height)
                    #if self.niter==2:
                    #    import pdb; pdb.set_trace()

            # No refit of stellar parameters
            else:
                model = psf(x,y,pars=[height,xcen,ycen])
                
            #model = psf(x,y,pars=[height,xcen,ycen])
            # Zero-out anything beyond the fitting radius
            #im[mask] = 0.0
            #npix = im.size
            npix = len(x)
            allim[pixcnt:pixcnt+npix] = model.flatten()
            pixcnt += npix
            
        self.niter += 1
            
        return allim

    
    def jac(self,x,*args,retmodel=False,refit=True):
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

        # Need to run model() to calculate height/xcen/ycen for first couple iterations
        if self.niter<=1 and refit:
            dum = self.model(x,*args)
            
        for i in range(self.nstars):
            height = self.starheight[i]
            xcen = self.starxcen[i]            
            ycen = self.starycen[i]
            bbox = self.bboxdata[i]
            x = self.xlist[i]
            y = self.ylist[i]
            pixstart = self.pixstart[i]
            npix = self.npix[i]
            flux = self.imflatten[pixstart:pixstart+npix]
            err = self.errflatten[pixstart:pixstart+npix]
            xdata = np.vstack((x,y))
            
            #xy = self.xydata[i]
            #x2,y2 = psf.bbox2xy(bbox)
            #xdata = np.vstack((x2.ravel(),y2.ravel()))

            #x0 = xcen - bbox.ixmin
            #y0 = ycen - bbox.iymin

            #import pdb; pdb.set_trace()
            
            # Get the model and derivative
            allpars = np.concatenate((np.array([height,xcen,ycen]),np.array(args)))
            m,deriv = psf.jac(xdata,*allpars,allpars=True,retmodel=True)            
            #if retmodel:
            #    m,deriv = psf.jac(xdata,*allpars,allpars=True,retmodel=True)
            #else:
            #    deriv = psf.jac(xdata,*allpars,allpars=True)                
            deriv = np.delete(deriv,[0,1,2],axis=1)  # remove stellar ht/xc/yc columns

            # Solve for the best height, and then scale the derivatives (all scale with height)
            if self.niter>1 and refit:
                newheight = height*np.median(flux/m)
                self.starheight[i] = newheight
                m *= newheight
                deriv *= newheight

            #if i==1: print(newheight)
            #import pdb; pdb.set_trace()
            
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
    
    pf = PSFFitter(psf,image,cat,verbose=verbose)
    xdata = np.arange(pf.ntotpix)
    initpar = psf.params.copy()
    method = str(method).lower()

    print('KLUDGE!! FORCING CURVE_FIT FOR NOW')
    method = 'curve_fit'
    #method = 'cholesky'

    # testing the derivative
    #orig = PSFFitter(psf,image,cat,verbose=verbose)
    #m = pf.model(xdata,*psf.params)
    #m2,jac = orig.jac(xdata,*psf.params,retmodel=True)
    #import pdb; pdb.set_trace()
    
    # Curve_fit
    if method=='curve_fit':    
        # Perform the fitting
        bestpar,cov = curve_fit(pf.model,xdata,pf.imflatten,
                                sigma=pf.errflatten,p0=initpar) #,jac=pf.jac)
        perror = np.sqrt(np.diag(cov))
        
    # All other fitting methods
    else:
        # Iterate
        count = 0
        percdiff = 1e10
        bestpar = initpar.copy()

        while (count<maxiter and percdiff>minpercdiff):
            # Get the Jacobian and model
            m,jac = pf.jac(xdata,*bestpar,retmodel=True)
            dy = pf.imflatten-m
            # Weights
            wt = 1/pf.errflatten**2
            # Solve Jacobian
            dbeta = lsq.jac_solve(jac,dy,method=method,weight=wt)
            print('dbeta = ',dbeta)
            
            # Update the parameters
            oldpar = bestpar.copy()
            bestpar += dbeta
            diff = np.abs(bestpar-oldpar)
            denom = np.abs(oldpar.copy())
            denom[denom==0] = 1.0  # deal with zeros
            percdiff = np.max(diff/denom*100)
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

    import pdb; pdb.set_trace()
    
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


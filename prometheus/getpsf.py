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
from . import leastsquares as lsq,models

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
        ny,nx = image.data.shape
        self.nx = nx
        self.ny = ny
        if fitradius is None:
            if type(psf)==models.PSFPenny:
                fitradius = psf.fwhm()*1.5
            else:
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
        # Original X/Y values
        self.starxcenorig = np.zeros(self.nstars,float)
        self.starxcenorig[:] = cat['x'].copy()
        self.starycenorig = np.zeros(self.nstars,float)
        self.starycenorig[:] = cat['y'].copy()
        # current best-fit values
        self.starxcen = np.zeros(self.nstars,float)
        self.starxcen[:] = cat['x'].copy()
        self.starycen = np.zeros(self.nstars,float)
        self.starycen[:] = cat['y'].copy()        
        self.starchisq = np.zeros(self.nstars,float)
        self.starrms = np.zeros(self.nstars,float)        
        self.starnpix = np.zeros(self.nstars,int)
        
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
            err = image.error[bbox.slices]
            imdata.append(im)
            bboxdata.append(bbox)
            # Trim to only the pixels that we want to fit
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
            self.starnpix[i] = npix
            imflatten[count:count+npix] = flux
            errflatten[count:count+npix] = err
            pixstart.append(count)
            xlist.append(x)
            ylist.append(y)
            npixdata.append(npix)
            count += npix

        self.imdata = imdata
        self.bboxdata = bboxdata            
        imflatten = imflatten[0:count]    # remove extra elements
        errflatten = errflatten[0:count]
        self.imflatten = imflatten
        self.errflatten = errflatten
        self.ntotpix = count
        self.xlist = xlist
        self.ylist = ylist
        self.npix = npixdata
        self.pixstart = pixstart

        
    def model(self,x,*args,refit=True,verbose=False):
        """ model function."""
        # input the model parameters
        
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
            xcenorig = self.starxcenorig[i]   
            ycenorig = self.starycenorig[i]
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

            x0orig = xcenorig - bbox.ixmin
            y0orig = ycenorig - bbox.iymin
            x0 = xcen - bbox.ixmin
            y0 = ycen - bbox.iymin            
            
            # Fit height/xcen/ycen if niter=1
            if refit:
                #if (self.niter<=1): # or self.niter%3==0):
                if self.niter>-1:
                    # force the positions to stay within +/-2 pixels of the original values
                    bounds = (np.array([0,np.maximum(x0orig-2,0),np.maximum(y0orig-2,0),-np.inf]),
                              np.array([np.inf,np.minimum(x0orig+2,bbox.shape[1]-1),np.minimum(y0orig+2,bbox.shape[0]-1),np.inf]))
                    # the image still has sky in it, use sky (nosky=False)
                    pars,perror,model = psf.fit(image,[height,x0,y0],nosky=False,retpararray=True,niter=5,bounds=bounds)
                    xcen += (pars[1]-x0)
                    ycen += (pars[2]-y0)
                    height = pars[0]                    
                    self.starheight[i] = height
                    self.starxcen[i] = xcen
                    self.starycen[i] = ycen
                    model = psf(x,y,pars=[height,xcen,ycen])
                    if verbose:
                        print('Star '+str(i)+' Refitting all parameters')
                        print([height,xcen,ycen])

                    #pars2,model2,mpars2 = psf.fit(image,[height,x0,y0],nosky=False,niter=5,allpars=True)
                    #import pdb; pdb.set_trace()
                        
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


                    #count = 0
                    #percdiff = 1e30
                    #while (count<3 and percdiff>0.1):                  
                    #    m,jac = psf.jac(np.vstack((x,y)),*[height,xcen,ycen],retmodel=True)
                    #    jac = np.delete(jac,[1,2],axis=1)
                    #    dy = flux-m
                    #    dbeta = lsq.jac_solve(jac,dy,method='cholesky',weight=wt)
                    #    print(count,height,dbeta)
                    #    height += dbeta
                    #    percdiff = np.abs(dbeta)/np.abs(height)*100
                    #    count += 1
                        
                    #pars2,perror2,model2 = psf.fit(image,[height,x0,y0],nosky=False,retpararray=True,niter=5)
                    #height = pars2[0]
                    #model = psf(x,y,pars=[height,xcen,ycen])
                    
                    self.starheight[i] = height
                    model = model1*height
                    #self.starxcen[i] = pars2[1]+xy[0][0]
                    #self.starycen[i] = pars2[2]+xy[1][0]       
                    #print(count,self.starxcen[i],self.starycen[i])
                    # updating the X/Y values after the first iteration
                    #  causes problems.  bounces around too much

                    if verbose:
                        print('Star '+str(i)+' Refitting height empirically')
                        print(height)
                        
                    #if i==1: print(height)
                    #if self.niter==2:
                    #    import pdb; pdb.set_trace()

            # No refit of stellar parameters
            else:
                model = psf(x,y,pars=[height,xcen,ycen])

            #if self.niter>1:
            #    import pdb; pdb.set_trace()
                
            # Relculate reduced chi squared
            chisq = np.sum((flux-model.ravel())**2/err**2)/npix
            self.starchisq[i] = chisq
            # chi value, RMS of the residuals as a fraction of the height
            rms = np.sqrt(np.mean(((flux-model.ravel())/self.starheight[i])**2))
            self.starrms[i] = rms
            
            #model = psf(x,y,pars=[height,xcen,ycen])
            # Zero-out anything beyond the fitting radius
            #im[mask] = 0.0
            #npix = im.size
            #npix = len(x)
            allim[pixcnt:pixcnt+npix] = model.flatten()
            pixcnt += npix

            #import pdb; pdb.set_trace()
            
        self.niter += 1
            
        return allim

    
    def jac(self,x,*args,retmodel=False,refit=True):
        """ jacobian."""
        # input the model parameters

        if self.verbose:
            print('jac: ',self.niter,args)
        
        psf = self.psf.copy()
        psf._params = list(args)
        # Loop over the stars and generate the derivatives
        #-------------------------------------------------

        # Initalize output arrays
        allderiv = np.zeros((self.ntotpix,len(psf.params)),float)
        if retmodel:
            allim = np.zeros(self.ntotpix,float)
        pixcnt = 0

        # Need to run model() to calculate height/xcen/ycen for first couple iterations
        #if self.niter<=1 and refit:
        #    dum = self.model(x,*args,refit=refit)
        dum = self.model(x,*args,refit=True) #,verbose=True)            
            
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
            #if self.niter>1 and refit:
            #    newheight = height*np.median(flux/m)
            #    self.starheight[i] = newheight
            #    m *= (newheight/height)
            #    deriv *= (newheight/height)

            #if i==1: print(height,newheight)
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

    def starmodel(self,star=None,pars=None):
        """ Generate 2D star model images that can be compared to the original cutouts.
             if star=None, then it will return all of them as a list."""

        psf = self.psf.copy()
        if pars is not None:
            psf._params = pars
        
        model = []
        if star is None:
            star = np.arange(self.nstars)
        else:
            star = [star]

        for i in star:
            image = self.imdata[i]
            height = self.starheight[i]
            xcen = self.starxcen[i]   
            ycen = self.starycen[i]
            bbox = self.bboxdata[i]
            model1 = psf(pars=[height,xcen,ycen],bbox=bbox)
            model.append(model1)
        return model

def fitpsf(psf,image,cat,fitradius=None,method='qr',maxiter=10,minpercdiff=1.0,verbose=False):
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
    fitradius : float, table
       The fitting radius.  If none is input then the initial PSF FWHM will be used.
    method : str, optional
       Method to use for solving the non-linear least squares problem: "qr",
       "svd", "cholesky", and "curve_fit".  Default is "qr".
    maxiter : int, optional
       Maximum number of iterations to allow.  Only for methods "qr", "svd", and "cholesky".
       Default is 10.
    minpercdiff : float, optional
       Minimum percent change in the parameters to allow until the solution is
       considered converged and the iteration loop is stopped.  Only for methods
       "qr" and "svd".  Default is 1.0.
    verbose : boolean, optional
       Verbose output.

    Returns
    -------
    newpsf : PSF object
       New PSF object with the best-fit model parameters.
    pars : numpy array
       Array of best-fit model parameters
    perror : numpy array
       Uncertainties in "pars".
    psfcat : table
       Table of best-fitting height/xcen/ycen values for the PSF stars.

    Example
    -------

    newpsf,pars,perror,psfcat = fitpsf(psf,image,cat)

    """

    t0 = time.time()

    # Fitting the PSF to the stars
    #-----------------------------
    pf = PSFFitter(psf,image,cat,fitradius=fitradius,verbose=False) #verbose)
    xdata = np.arange(pf.ntotpix)
    initpar = psf.params.copy()
    method = str(method).lower()

    # Curve_fit
    if method=='curve_fit':    
        # Perform the fitting
        bestpar,cov = curve_fit(pf.model,xdata,pf.imflatten,
                                sigma=pf.errflatten,p0=initpar,jac=pf.jac)
        perror = np.sqrt(np.diag(cov))
        
    # All other fitting methods
    else:
        # Iterate
        count = 0
        percdiff = 1e10
        bestpar = initpar.copy()

        dchisq = -1
        oldchisq = 1e30
        bounds = psf.bounds
        maxsteps = psf._steps
        while (count<maxiter and percdiff>minpercdiff and dchisq<0):
            # Get the Jacobian and model
            m,jac = pf.jac(xdata,*bestpar,retmodel=True)
            chisq = np.sum((pf.imflatten-m)**2/pf.errflatten**2)
            dy = pf.imflatten-m
            # Weights
            wt = 1/pf.errflatten**2
            # Solve Jacobian
            dbeta = lsq.jac_solve(jac,dy,method=method,weight=wt)
            if verbose:
                print('  pars = ',bestpar)
                print('  dbeta = ',dbeta)
            
            # Update the parameters
            oldpar = bestpar.copy()
            bestpar = psf.newpars(bestpar,dbeta,bounds,maxsteps)
            diff = np.abs(bestpar-oldpar)
            denom = np.abs(oldpar.copy())
            denom[denom==0] = 1.0  # deal with zeros
            percdiff = np.max(diff/denom*100)
            dchisq = chisq-oldchisq
            percdiffchisq = dchisq/oldchisq*100
            oldchisq = chisq
            count += 1
            
            if verbose:
                print('  ',count+1,bestpar,percdiff,chisq)
                
    # Make the best model
    bestmodel = pf.model(xdata,*bestpar)
    
    # Estimate uncertainties
    if method != 'curve_fit':
        # Calculate covariance matrix
        cov = lsq.jac_covariance(jac,dy,wt=wt)
        perror = np.sqrt(np.diag(cov))
                
    pars = bestpar
    if verbose:
        print('Best-fitting parameters: ',pars)
        print('Errors: ',perror)
        print('Median RMS: ',np.median(pf.starrms))

    # create the best-fitting PSF
    newpsf = psf.copy()
    newpsf._params = pars                

    # Output best-fitting values for the PSF stars as well
    dt = np.dtype([('id',int),('height',float),('x',float),('y',float),('npix',int),('rms',float),
                   ('chisq',float),('ixmin',int),('ixmax',int),('iymin',int),('iymax',int)])
    psfcat = np.zeros(len(cat),dtype=dt)
    if 'id' in cat.colnames:
        psfcat['id'] = cat['id']
    else:
        psfcat['id'] = np.arange(len(cat))+1
    psfcat['height'] = pf.starheight
    psfcat['x'] = pf.starxcen
    psfcat['y'] = pf.starycen
    psfcat['chisq'] = pf.starchisq
    psfcat['rms'] = pf.starrms
    psfcat['npix'] = pf.starnpix    
    for i in range(len(cat)):
        bbox = pf.bboxdata[i]
        psfcat['ixmin'][i] = bbox.ixmin
        psfcat['ixmax'][i] = bbox.ixmax
        psfcat['iymin'][i] = bbox.iymin
        psfcat['iymax'][i] = bbox.iymax        
    psfcat = Table(psfcat)
        
    if verbose:
        print('dt = %.2f sec' % (time.time()-t0))
        
    # Make the star models
    #starmodels = pf.starmodel(pars=pars)
    
    return newpsf, pars, perror, psfcat

    
def getpsf(psf,image,cat,fitradius=None,method='qr',subnei=False,allcat=None,
           maxiter=10,minpercdiff=1.0,reject=False,maxrejiter=3,verbose=False):
    """
    Fit PSF model to stars in an image with outlier rejection of badly-fit stars.

    Parameters
    ----------
    psf : PSF object
       PSF object with initial parameters to use.
    image : CCDData object
       Image to use to fit PSF model to stars.
    cat : table
       Catalog with initial height/x/y values for the stars to use to fit the PSF.
    fitradius : float, table
       The fitting radius.  If none is input then the initial PSF FWHM will be used.
    method : str, optional
       Method to use for solving the non-linear least squares problem: "qr",
       "svd", "cholesky", and "curve_fit".  Default is "qr".
    subnei : boolean, optional
       Subtract stars neighboring the PSF stars.  Default is False.
    allcat : table, optional
       Catalog of all objects in the image.  This is needed for bad PSF star
       rejection.
    maxiter : int, optional
       Maximum number of iterations to allow.  Only for methods "qr", "svd", and "cholesky".
       Default is 10.
    minpercdiff : float, optional
       Minimum percent change in the parameters to allow until the solution is
       considered converged and the iteration loop is stopped.  Only for methods
       "qr" and "svd".  Default is 1.0.
    reject : boolean, optional
       Reject PSF stars with high RMS values.  Default is False.
    maxrejiter : int, boolean
       Maximum number of PSF star rejection iterations.  Default is 3.
    verbose : boolean, optional
       Verbose output.

    Returns
    -------
    newpsf : PSF object
       New PSF object with the best-fit model parameters.
    pars : numpy array
       Array of best-fit model parameters
    perror : numpy array
       Uncertainties in "pars".
    psfcat : table
       Table of best-fitting height/xcen/ycen values for the PSF stars.

    Example
    -------

    newpsf,pars,perror,psfcat = getpsf(psf,image,cat)

    """

    t0 = time.time()

    # Fitting radius
    if fitradius is None:
        if type(psf)==models.PSFPenny:
            fitradius = psf.fwhm()*1.5
        else:
            fitradius = psf.fwhm()
        
    # subnei but no allcat input
    if subnei and allcat is None:
        raise ValueError('allcat is needed for PSF neighbor star subtraction')
        
    if 'id' not in cat.colnames:
        cat['id'] = np.arange(len(cat))+1
    psfcat = cat.copy()

    # Initializing output PSF star catalog
    dt = np.dtype([('id',int),('height',float),('x',float),('y',float),('npix',int),('rms',float),
                   ('chisq',float),('ixmin',int),('ixmax',int),('iymin',int),('iymax',int),('reject',int)])
    outcat = np.zeros(len(cat),dtype=dt)
    outcat = Table(outcat)
    for n in ['id','x','y']:
        outcat[n] = cat[n]
    
    # Remove stars that are too close to the edge
    ny,nx = image.shape
    bd = (psfcat['x']<fitradius) | (psfcat['x']>(nx-1-fitradius)) | \
         (psfcat['y']<fitradius) | (psfcat['y']>(ny-1-fitradius))
    nbd = np.sum(bd)
    if nbd > 0:
        if verbose:
            print('Removing '+str(nbd)+' stars near the edge')
        psfcat = psfcat[~bd]

    # Outlier rejection iterations
    curpsf = psf.copy()
    nrejiter = 0
    flag = 0
    nrejstar = 100
    fitrad = fitradius
    useimage = image.copy()
    while (flag==0):
        if verbose:
            print('--- Iteration '+str(nrejiter+1)+' ---')                

        # Update the fitting radius
        if nrejiter>0:
            fitrad = curpsf.fwhm()
        if verbose:
            print('  Fitting radius = %5.3f' % (fitrad))
        
            
        # Reject outliers
        if reject and nrejiter>0:
            medrms = np.median(pcat['rms'])
            sigrms = dln.mad(pcat['rms'].data)
            gd, = np.where(pcat['rms'] < medrms+3*sigrms)
            nrejstar = len(psfcat)-len(gd)
            if verbose:
                print('  RMS = %6.4f +/- %6.4f' % (medrms,sigrms))
                print('  Threshold RMS = '+str(medrms+3*sigrms))
                print('  Rejecting '+str(nrejstar)+' stars')
            if nrejstar>0:
                psfcat = psfcat[gd]

        # Subtract neighbors
        if nrejiter>0 and subnei:
            if verbose:
                print('Subtracting neighbors')
                # Find the neighbors in allcat
                # Fit the neighbors and PSF stars
                # Subtract neighbors from the image
                import pdb; pdb.set_trace()
                
        # Fitting the PSF to the stars
        #-----------------------------
        newpsf,pars,perror,pcat = fitpsf(curpsf,useimage,psfcat,fitradius=fitrad,method=method,
                                         maxiter=maxiter,minpercdiff=minpercdiff,verbose=verbose)

        # Add information into the output catalog
        ind1,ind2 = dln.match(outcat['id'],pcat['id'])
        outcat['reject'] = 1
        for n in pcat.columns:
            outcat[n][ind1] = pcat[n][ind2]
        outcat['reject'][ind1] = 0

        # Compare PSF parameters
        pardiff = newpsf.params-curpsf.params
        sumpardiff = np.sum(np.abs(pardiff))
        curpsf = newpsf.copy()
        
        # Stopping criteria
        if reject is False or sumpardiff<0.05 or nrejiter>=maxrejiter or nrejstar==0: flag=1
        if subnei is True and nrejiter==0: flag=0   # iterate at least once with neighbor subtraction
        
        nrejiter += 1
        
    if verbose:
        print('dt = %.2f sec' % (time.time()-t0))
    
    return newpsf, pars, perror, outcat

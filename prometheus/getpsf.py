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
from scipy.interpolate import interp1d,interp2d
from scipy.interpolate import RectBivariateSpline
#from astropy.nddata import CCDData,StdDevUncertainty
from dlnpyutils import utils as dln, bindata, ladfit, coords
from scipy.spatial import cKDTree
import copy
import logging
import time
import matplotlib
import sep
from . import leastsquares as lsq,models,utils
from .ccddata import CCDData

# Fit a PSF model to multiple stars in an image

def starcube(cat,image,npix=51,fillvalue=np.nan):
    """ Produce a cube of cutouts of stars."""

    # Get the residuals data
    nstars = len(cat)
    nhpix = npix//2
    cube = np.zeros((npix,npix,nstars),float)
    xx,yy = np.meshgrid(np.arange(npix)-nhpix,np.arange(npix)-nhpix)
    rr = np.sqrt(xx**2+yy**2)        
    x = xx[0,:]
    y = yy[:,0]
    for i in range(nstars):
        xcen = cat['x'][i]            
        ycen = cat['y'][i]
        bbox = starbbox((xcen,ycen),image.shape,nhpix)
        im = image[bbox.slices]
        flux = image.data[bbox.slices]-image.sky[bbox.slices]
        err = image.error[bbox.slices]
        if 'height' in cat.columns:
            height = cat['height'][i]
        elif 'peak' in cat.columns:
            height = cat['peak'][i]
        else:
            height = flux[int(np.round(ycen)),int(np.round(xcen))]
        xim,yim = np.meshgrid(im.x,im.y)
        xim = xim.astype(float)-xcen
        yim = yim.astype(float)-ycen
        # We need to interpolate this onto the grid
        f = RectBivariateSpline(yim[:,0],xim[0,:],flux/height)
        im2 = np.zeros((npix,npix),float)+np.nan
        xcover = (x>=bbox.ixmin-xcen) & (x<=bbox.ixmax-1-xcen)
        xmin,xmax = dln.minmax(np.where(xcover)[0])
        ycover = (y>=bbox.iymin-ycen) & (y<=bbox.iymax-1-ycen)
        ymin,ymax = dln.minmax(np.where(ycover)[0])            
        im2[ymin:ymax+1,xmin:xmax+1] = f(y[ycover],x[xcover],grid=True)
        # Stuff it into 3D array
        cube[:,:,i] = im2
    return cube

def mkempirical(cube,order=0,coords=None,shape=None,rect=False):
    """ Take a star cube and collapse it to make an empirical PSF using median
        and outlier rejection."""

    ny,nx,nstar = cube.shape
    npix = ny
    nhpix = ny//2
    
    # Do outlier rejection in each pixel
    med = np.nanmedian(cube,axis=2)
    bad = ~np.isfinite(med)
    if np.sum(bad)>0:
        med[bad] = np.nanmedian(med)
    sig = dln.mad(cube,axis=2)
    bad = ~np.isfinite(sig)
    if np.sum(bad)>0:
        sig[bad] = np.nanmedian(sig)        
    # Mask outlier points
    outliers = ((np.abs(cube-med.reshape((med.shape)+(-1,)))>3*sig.reshape((med.shape)+(-1,)))
                & np.isfinite(cube))
    nbadstar = np.sum(outliers,axis=(0,1))
    goodmask = ((np.abs(cube-med.reshape((med.shape)+(-1,)))<3*sig.reshape((med.shape)+(-1,)))
                & np.isfinite(cube))    
    # Now take the mean of the unmasked pixels
    macube = np.ma.array(cube,mask=~goodmask)
    medim = macube.mean(axis=2)
    medim = medim.data
    
    # Check how well each star fits the median
    goodpix = macube.count(axis=(0,1))
    rms = np.sqrt(np.nansum((cube-medim.reshape((medim.shape)+(-1,)))**2,axis=(0,1))/goodpix)

    xx,yy = np.meshgrid(np.arange(npix)-nhpix,np.arange(npix)-nhpix)
    rr = np.sqrt(xx**2+yy**2)        
    x = xx[0,:]
    y = yy[:,0]
    
    # Constant
    if order==0:
        # Make sure it goes to zero at large radius
        outer = np.median(medim[rr>nhpix*0.8])
        medim -= outer
        if rect:
            fpars = [RectBivariateSpline(y,x,medim)]
        else:
            fpars = medim
            
    # Linear
    elif order==1:
        if coords is None or shape is None:
            raise ValueError('Need coords and shape with order=1')
        pars = np.zeros((ny,nx,4),float)
        # scale coordinates to -1 to +1
        xcen,ycen = coords
        relx,rely = relcoord(xcen,ycen,shape)
        # Loop over pixels and fit line to x/y
        for i in range(ny):
            for j in range(nx):
                data1 = cube[i,j,:]
                # maybe use a small maxiter
                pars1,perror1 = utils.poly2dfit(relx,rely,data1)
                pars[i,j,:] = pars1
        # Make sure it goes to zero at large radius
        if rect:
            fpars = []
            for i in range(4):
                #outer = np.median(pars[rr>nhpix*0.8,i])
                #pars[:,:,i] -= outer
                # Set up the spline function that we can use to do
                # the interpolation
                fpars.append(RectBivariateSpline(y,x,pars[:,:,i]))
        else:
            fpars = pars
                
    return fpars,nbadstar,rms

def findpsfnei(allcat,psfcat,npix):
    """ Find stars near PSF stars."""
    # Returns distance and index of closest neighbor
    
    nallcat = len(allcat)
    npsfcat = len(psfcat)
    
    # Use KD-tree
    X1 = np.vstack((allcat['x'].data,allcat['y'].data)).T
    X2 = np.vstack((psfcat['x'].data,psfcat['y'].data)).T
    kdt = cKDTree(X1)
    # Get distance for 2 closest neighbors
    dist, ind = kdt.query(X2, k=50, distance_upper_bound=np.sqrt(2)*npix//2)
    # closest neighbor is always itself, remove it 
    dist = dist[:,1:]
    ind = ind[:,1:]
    # Add up all the stars
    gdall, = np.where(np.isfinite(dist.ravel()))
    indall = ind.ravel()[gdall]
    # Get unique ones
    indall = np.unique(indall)

    # Make sure none of them are our psf stars
    ind1,ind2,dist = coords.xmatch(allcat['x'][indall],allcat['y'][indall],psfcat['x'],psfcat['y'],5)
    # Remove any matches
    if len(ind1)>0:
        indall = np.delete(indall,ind1)
    
    return indall

def subtractnei(image,allcat,psfcat,psf):
    """ Subtract neighboring stars to PSF stars from the image."""
    indnei = findpsfnei(allcat,psfcat,psf.npix)
    nnei = len(indnei)

    flux = image.data-image.sky
    resid = image.copy()
    fitradius = psf.fwhm()*0.5
    
    # Loop over neighboring stars and fit just the core
    for i in range(nnei):
        x1 = allcat['x'][indnei[i]]
        xp1 = int(np.minimum(np.maximum(np.round(x1),0),image.shape[1]-1))
        y1 = allcat['y'][indnei[i]]
        yp1 = int(np.minimum(np.maximum(np.round(y1),0),image.shape[0]-1))
        if 'height' in allcat.columns:
            h1 = allcat['height'][indnei[i]]
        elif 'peak' in allcat.columns:
            h1 = allcat['peak'][indnei[i]]
        else:
            h1 = flux[yp1,xp1]
        initpars = [h1,x1,y1] #image.sky[yp1,xp1]]
        bbox = psf.starbbox((initpars[1],initpars[2]),image.shape,psf.radius)
        # Fit height empirically with central pixels
        flux1 = flux[bbox.slices]
        err1 = image[bbox.slices].error
        model1 = psf(pars=initpars,bbox=bbox)
        good = ((flux1/err1>2) & (flux1>0) & (model1/np.max(model1)>0.25))
        height = np.median(flux1[good]/model1[good]) * initpars[0]
        pars = [height, x1, y1]
        #starcat,perror = psf.fit(flux,pars=initpars,radius=fitradius,recenter=False,niter=2)
        #pars = [starcat['height'][0],starcat['x'][0],starcat['y'][0]]
        im1 = psf(pars=pars,bbox=bbox)
        resid[bbox.slices].data -= im1
    return resid

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
            print('model: '+str(self.niter)+' '+str(args))
        
        psf = self.psf.copy()
        if type(psf)!=models.PSFEmpirical:
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
                        print(str([height,xcen,ycen]))

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
                        print(str(height))
                        
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
            print('jac: '+str(self.niter)+' '+str(args))
        
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

    def mklookup(self,order=0):
        """ Make an empirical look-up table for the residuals."""

        # Get the residuals data
        npix = self.psf.npix
        nhpix = npix//2
        resid = np.zeros((npix,npix,self.nstars),float)
        xx,yy = np.meshgrid(np.arange(npix)-nhpix,np.arange(npix)-nhpix)
        rr = np.sqrt(xx**2+yy**2)        
        x = xx[0,:]
        y = yy[:,0]
        for i in range(self.nstars):
            height = self.starheight[i]
            xcen = self.starxcen[i]            
            ycen = self.starycen[i]
            bbox = self.psf.starbbox((xcen,ycen),self.image.shape,radius=nhpix)
            im = self.image[bbox.slices]
            flux = self.image.data[bbox.slices]-self.image.sky[bbox.slices]
            err = self.image.error[bbox.slices]
            xim,yim = np.meshgrid(im.x,im.y)
            xim = xim.astype(float)-xcen
            yim = yim.astype(float)-ycen
            # We need to interpolate this onto the grid
            f = RectBivariateSpline(yim[:,0],xim[0,:],flux/height)
            im2 = np.zeros((npix,npix),float)+np.nan
            xcover = (x>=bbox.ixmin-xcen) & (x<=bbox.ixmax-1-xcen)
            xmin,xmax = dln.minmax(np.where(xcover)[0])
            ycover = (y>=bbox.iymin-ycen) & (y<=bbox.iymax-1-ycen)
            ymin,ymax = dln.minmax(np.where(ycover)[0])            
            im2[ymin:ymax+1,xmin:xmax+1] = f(y[ycover],x[xcover],grid=True)
            # Get model
            model = self.psf(pars=[1.0,0.0,0.0],bbox=[[-nhpix,nhpix+1],[-nhpix,nhpix+1]])
            # Stuff it into 3D array
            resid[:,:,i] = im2-model
            
        # Constant
        if order==0:
            # Do outlier rejection in each pixel
            med = np.nanmedian(resid,axis=2)
            bad = ~np.isfinite(med)
            if np.sum(bad)>0:
                med[bad] = np.nanmedian(med)
            sig = dln.mad(resid,axis=2)
            bad = ~np.isfinite(sig)
            if np.sum(bad)>0:
                sig[bad] = np.nanmedian(sig)        
            # Mask outlier points
            mask = ( (np.abs(resid-med.reshape((med.shape)+(-1,)))<3*sig.reshape((med.shape)+(-1,))) &
                     np.isfinite(resid) )
            # Now take the mean of the unmasked pixels
            resid[~mask] = np.nan
            pars = np.nanmean(resid,axis=2)
            # Make sure it goes to zero at large radius
            outer = np.median(pars[rr>nhpix*0.8])
            pars -= outer
            # Set up the spline function that we can use to do
            # the interpolation
            fpars = [RectBivariateSpline(y,x,pars)]
            
        # Linear
        elif order==1:
            pars = np.zeros((npix,npix,4),float)
            # scale coordinates to -1 to +1
            relx = (self.starxcen-self.image.shape[1]//2)/self.image.shape[1]*2
            rely = (self.starycen-self.image.shape[0]//2)/self.image.shape[0]*2
            #return resid,relx,rely
            # Loop over pixels and fit line to x/y
            for i in range(npix):
                for j in range(npix):
                    data1 = resid[i,j,:]
                    # not sure this is any faster than curve_fit
                    # maybe use a small maxiter
                    pars1,perror1 = utils.poly2dfit(relx,rely,data1)
                    
                    #gd, = np.where(np.isfinite(data1))
                    #xdata = [relx[gd],rely[gd]]
                    #initpars = np.zeros(4,float)
                    #med = np.median(data1[gd])
                    #xcoef,xadev = ladfit.ladfit(relx[gd],data1[gd])
                    #ycoef,yadev = ladfit.ladfit(rely[gd],data1[gd])
                    #initpars = np.array([xcoef[0],xcoef[1],ycoef[1],0.0])
                    #diff = data1-poly2d([relx,rely],*initpars)
                    #meddiff = np.nanmedian(diff)
                    #sigdiff = dln.mad(diff)
                    #gd, = np.where( (np.abs(diff-meddiff)<3*sigdiff) & np.isfinite(diff))
                    #xdata = [relx[gd],rely[gd]]
                    #pars1,cov1 = curve_fit(poly2d,xdata,data1[gd],initpars,sigma=np.zeros(len(gd),float)+1)
                    # REPLACE CURVE_FIT WITH SOMETHING FASTER!!
                    pars[i,j,:] = pars1
            # Make sure it goes to zero at large radius
            fpars = []
            for i in range(4):
                outer = np.median(pars[rr>nhpix*0.8,i])
                pars[:,:,i] -= outer
                # Set up the spline function that we can use to do
                # the interpolation
                fpars.append(RectBivariateSpline(y,x,pars[:,:,i]))
                    
        else:
            raise ValueError('Only lookup order=0 or 1 allowed')

        # DAOPHOT does some extra analysis to make sure the flux
        # in the residual component is okay

        # -make sure
        #  -to take the total flux into account (not varying across image)
        #  -make sure the height=1 at center
        #  -make sure all PSF values are >=0
                             
        # Add the lookup table to the PSF model
        self.psf.lookup = True
        self.psf._lookup_order = order
        self.psf._lookup_data = pars
        self.psf._lookup_interp = fpars  # spline functions                             
        self.psf._lookup_midpt = [self.image.shape[0]//2,self.image.shape[1]//2]
        self.psf._lookup_shape = self.image.shape      

        #import pdb; pdb.set_trace()
        
        
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

    
def fitpsf(psf,image,cat,fitradius=None,method='qr',maxiter=10,minpercdiff=1.0,
           verbose=False):
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
    print = utils.getprintfunc() # Get print function to be used locally, allows for easy logging   

    # Initialize the output catalog best-fitting values for the PSF stars
    dt = np.dtype([('id',int),('height',float),('x',float),('y',float),('npix',int),('rms',float),
                   ('chisq',float),('ixmin',int),('ixmax',int),('iymin',int),('iymax',int)])
    psfcat = np.zeros(len(cat),dtype=dt)
    if 'id' in cat.colnames:
        psfcat['id'] = cat['id']
    else:
        psfcat['id'] = np.arange(len(cat))+1
    

    # Fitting the PSF to the stars
    #-----------------------------

    # Empirical PSF - done differently
    if type(psf)==models.PSFEmpirical:
        cube1 = starcube(cat,image,npix=psf.npix,fillvalue=np.nan)
        epsf1,nbadstar1,rms1 = mkempirical(cube1,order=psf.order)
        initpsf = models.PSFEmpirical(epsf1,imshape=image.shape,order=psf.order)
        pf = PSFFitter(initpsf,image,cat,fitradius=fitradius,verbose=False)
        # Fit the height, xcen, ycen properly
        xdata = np.arange(pf.ntotpix)
        out = pf.model(xdata,[])
        # Put information into the psfcat table
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
        # Remake the empirical EPSF    
        cube = starcube(psfcat,image,npix=psf.npix,fillvalue=np.nan)
        epsf,nbadstar,rms = mkempirical(cube,order=psf.order)
        newpsf = models.PSFEmpirical(epsf,imshape=image.shape,order=psf.order)
        if verbose:
            print('Median RMS: '+str(np.median(pf.starrms)))        
        return newpsf, None, None, psfcat, pf
        
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
                print('  pars = '+str(bestpar))
                print('  dbeta = '+str(dbeta))
            
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
                print('  '+str(count+1)+' '+str(bestpar)+' '+str(percdiff)+' '+str(chisq))
                
    # Make the best model
    bestmodel = pf.model(xdata,*bestpar)
    
    # Estimate uncertainties
    if method != 'curve_fit':
        # Calculate covariance matrix
        cov = lsq.jac_covariance(jac,dy,wt=wt)
        perror = np.sqrt(np.diag(cov))
                
    pars = bestpar
    if verbose:
        print('Best-fitting parameters: '+str(pars))
        print('Errors: '+str(perror))
        print('Median RMS: '+str(np.median(pf.starrms)))

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
    
    return newpsf, pars, perror, psfcat, pf

    
def getpsf(psf,image,cat,fitradius=None,lookup=False,lorder=0,method='qr',subnei=False,
           allcat=None,maxiter=10,minpercdiff=1.0,reject=False,maxrejiter=3,verbose=False):
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
    lookup : boolean, optional
       Use an empirical lookup table.  Default is False.
    lorder : int, optional
       The order of the spatial variations (0=constant, 1=linear).  Default is 0.
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
    print = utils.getprintfunc() # Get print function to be used locally, allows for easy logging   

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

    # Generate an empirical image of the stars
    # and fit a model to it to get initial estimates
    if type(psf)!=models.PSFEmpirical:
        cube = starcube(psfcat,image,npix=psf.npix,fillvalue=np.nan)
        epsf,nbadstar,rms = mkempirical(cube,order=0)
        epsfim = CCDData(epsf,error=epsf.copy()*0+1,mask=~np.isfinite(epsf))
        pars,perror,mparams = psf.fit(epsfim,pars=[1.0,psf.npix/2,psf.npix//2],allpars=True)
        initpar = mparams.copy()
        curpsf = psf.copy()
        curpsf.params = initpar
        if verbose:
            print('Initial estimate from empirical PSF fit = '+str(mparams))
    else:
        curpsf = psf.copy()
        initpar = psf.params.copy()
        
    # Outlier rejection iterations
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
                useimage = image.copy()  # start with original image
                useimage = subtractnei(useimage,allcat,cat,curpsf)
                
        # Fitting the PSF to the stars
        #-----------------------------
        newpsf,pars,perror,pcat,pf = fitpsf(curpsf,useimage,psfcat,fitradius=fitrad,method=method,
                                            maxiter=maxiter,minpercdiff=minpercdiff,verbose=verbose)

        # Add information into the output catalog
        ind1,ind2 = dln.match(outcat['id'],pcat['id'])
        outcat['reject'] = 1
        for n in pcat.columns:
            outcat[n][ind1] = pcat[n][ind2]
        outcat['reject'][ind1] = 0

        # Compare PSF parameters
        if type(newpsf)!=models.PSFEmpirical:
            pardiff = newpsf.params-curpsf.params
        else:
            pardiff = newpsf._data-curpsf._data
        sumpardiff = np.sum(np.abs(pardiff))
        curpsf = newpsf.copy()
        
        # Stopping criteria
        if reject is False or sumpardiff<0.05 or nrejiter>=maxrejiter or nrejstar==0: flag=1
        if subnei is True and nrejiter==0: flag=0   # iterate at least once with neighbor subtraction
        
        nrejiter += 1
        
    # Generate an empirical look-up table of corrections
    if lookup:
        if verbose:
            print('Making empirical lookup table with order='+str(lorder))
            
        #return pf.mklookup(lorder)

        pf.mklookup(lorder)
        newpsf = pf.psf.copy()
        
    if verbose:
        print('dt = %.2f sec' % (time.time()-t0))
    
    return newpsf, pars, perror, outcat

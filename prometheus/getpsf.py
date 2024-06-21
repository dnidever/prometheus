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
from scipy.optimize import curve_fit, least_squares, line_search
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
    """
    Produce a cube of cutouts of stars.

    Parameters
    ----------
    cat : table
       The catalog of stars to use.  This should have "x" and "y" columns and
         preferably also "amp".
    image : CCDData object
       The image to use to generate the stellar images.
    fillvalue : float, optional
       The fill value to use for pixels that are bad are off the image.
            Default is np.nan.

    Returns
    -------
    cube : numpy array
       Two-dimensional cube (Npix,Npix,Nstars) of the star images.

    Example
    -------

    cube = starcube(cat,image)

    """

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
        bbox = models.starbbox((xcen,ycen),image.shape,nhpix)
        im = image[bbox.slices]
        flux = image.data[bbox.slices]-image.sky[bbox.slices]
        err = image.error[bbox.slices]
        if 'amp' in cat.columns:
            amp = cat['amp'][i]
        elif 'peak' in cat.columns:
            amp = cat['peak'][i]
        else:
            amp = flux[int(np.round(ycen)),int(np.round(xcen))]
        xim,yim = np.meshgrid(im.x,im.y)
        xim = xim.astype(float)-xcen
        yim = yim.astype(float)-ycen
        # We need to interpolate this onto the grid
        f = RectBivariateSpline(yim[:,0],xim[0,:],flux/amp)
        im2 = np.zeros((npix,npix),float)+np.nan
        xcover = (x>=bbox.ixmin-xcen) & (x<=bbox.ixmax-1-xcen)
        xmin,xmax = dln.minmax(np.where(xcover)[0])
        ycover = (y>=bbox.iymin-ycen) & (y<=bbox.iymax-1-ycen)
        ymin,ymax = dln.minmax(np.where(ycover)[0])            
        im2[ymin:ymax+1,xmin:xmax+1] = f(y[ycover],x[xcover],grid=True)
        # Stuff it into 3D array
        cube[:,:,i] = im2
    return cube

def mkempirical(cube,order=0,coords=None,shape=None,rect=False,lookup=False):
    """
    Take a star cube and collapse it to make an empirical PSF using median
    and outlier rejection.

    Parameters
    ----------
    cube : numpy array
      Three-dimensional cube of star images (or residual images) of shape
        (Npix,Npix,Nstars).
    order : int, optional
      The order of the variations. 0-constant, 1-linear terms.  If order=1,
        Then coords and shape must be input.
    coords : tuple, optional
      Two-element tuple of the X/Y coordinates of the stars.  This is needed
        to generate the linear empirical model (order=1).
    shape : tuple, optional
      Two-element tuple giving the shape (Ny,Nx) of the image.  This is
        needed to generate the linear empirical model (order=1).
    rect : boolean, optional
      Return a list of RectBivariateSpline functions rather than a numpy array.
    lookup : boolean, optional
      Parameter to indicate if this is a lookup table.  If lookup=False, then
      the constant term is constrained to be non-negative. Default is False.

    Returns
    -------
    epsf : numpy array
      The empirical PSF model  If order=0, then this is just a 2D image.  If
        order=1, then it will be a 3D cube (Npix,Npix,4) where the four terms
        are [constant, X-term, Y-term, X*Y-term].  If rect=True, then a list
        of RectBivariateSpline functions are returned.

    Example
    -------

    epsf = mkempirical(cube,order=0)

    or

    epsf = mkempirical(cube,order=1,coords=coords,shape=im.shape)

    """

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
    mask = (rr<=nhpix)
    
    # Constant
    if order==0:
        # Make sure it goes to zero at large radius
        medim *= mask  # mask corners
        # Make sure values are positive
        if lookup==False:
            medim = np.maximum(medim,0.0)
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
        relx,rely = models.relcoord(xcen,ycen,shape)
        # Loop over pixels and fit line to x/y
        for i in range(ny):
            for j in range(nx):
                data1 = cube[i,j,:]
                if np.sum(np.abs(data1)) != 0:
                    # maybe use a small maxiter
                    pars1,perror1 = utils.poly2dfit(relx,rely,data1)
                    pars[i,j,:] = pars1
        # Make sure it goes to zero at large radius
        if rect:
            fpars = []
            for i in range(4):
                # Make sure edges are zero on average for higher-order terms
                outer = np.median(pars[rr>nhpix*0.8,i])
                pars[:,:,i] -= outer
                # Mask corners
                pars[:,:,i] *= mask
                # Each higher-order term must have ZERO total volume
                # Set up the spline function that we can use to do
                # the interpolation
                fpars.append(RectBivariateSpline(y,x,pars[:,:,i]))
        else:
            fpars = pars
                
    return fpars,nbadstar,rms

def findpsfnei(allcat,psfcat,npix):
    """
    Find stars near PSF stars.

    Parameters
    ----------
    allcat : table
      Catalog of all sources in the image.
    psfcat : table
      Catalog of PSF stars.
    npix : int
      Search radius in pixels.

    Returns
    -------
    indall : numpy array
       List of indices into allcat that are neighbors to
        the PSF stars (within radius of npix), and are not
        PSF stars themselves.

    Example
    -------

    indall = findpsfnei(allcat,psfcat,npix)

    """
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
    """
    Subtract neighboring stars to PSF stars from the image.

    Parameters
    ----------
    image : CCDDdata object
       The input image from which to subtract PSF neighbor stars.
    allcat : table
      Catalog of all sources in the image.
    psfcat : table
      Catalog of PSF stars.
    psf : PSF object
      The PSF model.

    Returns
    -------
    resid : CCDData object
      The input images with the PSF neighbor stars subtracted.

    Example
    -------

    subim = subtractnei(image,allcat,psfcat,psf)

    """

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
        if 'amp' in allcat.columns:
            h1 = allcat['amp'][indnei[i]]
        elif 'peak' in allcat.columns:
            h1 = allcat['peak'][indnei[i]]
        else:
            h1 = flux[yp1,xp1]
        initpars = [h1,x1,y1] #image.sky[yp1,xp1]]
        bbox = psf.starbbox((initpars[1],initpars[2]),image.shape,psf.radius)
        # Fit amp empirically with central pixels
        flux1 = flux[bbox.slices]
        err1 = image[bbox.slices].error
        model1 = psf(pars=initpars,bbox=bbox)
        good = ((flux1/err1>2) & (flux1>0) & (model1/np.max(model1)>0.25))
        amp = np.median(flux1[good]/model1[good]) * initpars[0]
        pars = [amp, x1, y1]
        #starcat,perror = psf.fit(flux,pars=initpars,radius=fitradius,recenter=False,niter=2)
        #pars = [starcat['amp'][0],starcat['x'][0],starcat['y'][0]]
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
        self.staramp = np.zeros(self.nstars,float)
        if 'amp' in cat.colnames:
            self.staramp[:] = cat['amp'].copy()
        else:
            # estimate amp from flux and fwhm
            # area under 2D Gaussian is 2*pi*A*sigx*sigy
            amp = cat['flux']/(2*np.pi*(cat['fwhm']/2.35)**2)
            self.staramp[:] = np.maximum(amp,0)   # make sure it's positive
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

        # Limit the parameters to the boundaries
        if type(psf)!=models.PSFEmpirical:
            lbnds,ubnds = psf.bounds
            for i in range(len(psf.params)):
                psf._params[i] = np.minimum(np.maximum(args[i],lbnds[i]),ubnds[i])
                
        # Loop over the stars and generate the model image
        allim = np.zeros(self.ntotpix,float)
        pixcnt = 0
        for i in range(self.nstars):
            image = self.imdata[i]
            amp = self.staramp[i]
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
            
            # Fit amp/xcen/ycen if niter=1
            if refit:
                #if (self.niter<=1): # or self.niter%3==0):
                if self.niter>-1:
                    # force the positions to stay within +/-2 pixels of the original values
                    bounds = (np.array([0,np.maximum(x0orig-2,0),np.maximum(y0orig-2,0),-np.inf]),
                              np.array([np.inf,np.minimum(x0orig+2,bbox.shape[1]-1),np.minimum(y0orig+2,bbox.shape[0]-1),np.inf]))
                    # the image still has sky in it, use sky (nosky=False)
                    if np.isfinite(psf.fwhm())==False:
                        print('nan fwhm')
                        import pdb; pdb.set_trace()
                    pars,perror,model = psf.fit(image,[amp,x0,y0],nosky=False,retpararray=True,niter=5,bounds=bounds)
                    xcen += (pars[1]-x0)
                    ycen += (pars[2]-y0)
                    amp = pars[0]                    
                    self.staramp[i] = amp
                    self.starxcen[i] = xcen
                    self.starycen[i] = ycen
                    model = psf(x,y,pars=[amp,xcen,ycen])
                    if verbose:
                        print('Star '+str(i)+' Refitting all parameters')
                        print(str([amp,xcen,ycen]))

                    #pars2,model2,mpars2 = psf.fit(image,[amp,x0,y0],nosky=False,niter=5,allpars=True)
                    #import pdb; pdb.set_trace()
                        
                # Only fit amp if niter>1
                #   do it empirically
                else:
                    #im1 = psf(pars=[1.0,xcen,ycen],bbox=bbox)
                    #wt = 1/image.error**2
                    #amp = np.median(image.data[mask]/im1[mask])                
                    model1 = psf(x,y,pars=[1.0,xcen,ycen])
                    wt = 1/err**2
                    amp = np.median(flux/model1)
                    #amp = np.median(wt*flux/model1)/np.median(wt)


                    #count = 0
                    #percdiff = 1e30
                    #while (count<3 and percdiff>0.1):                  
                    #    m,jac = psf.jac(np.vstack((x,y)),*[amp,xcen,ycen],retmodel=True)
                    #    jac = np.delete(jac,[1,2],axis=1)
                    #    dy = flux-m
                    #    dbeta = lsq.jac_solve(jac,dy,method='cholesky',weight=wt)
                    #    print(count,amp,dbeta)
                    #    amp += dbeta
                    #    percdiff = np.abs(dbeta)/np.abs(amp)*100
                    #    count += 1
                        
                    #pars2,perror2,model2 = psf.fit(image,[amp,x0,y0],nosky=False,retpararray=True,niter=5)
                    #amp = pars2[0]
                    #model = psf(x,y,pars=[amp,xcen,ycen])
                    
                    self.staramp[i] = amp
                    model = model1*amp
                    #self.starxcen[i] = pars2[1]+xy[0][0]
                    #self.starycen[i] = pars2[2]+xy[1][0]       
                    #print(count,self.starxcen[i],self.starycen[i])
                    # updating the X/Y values after the first iteration
                    #  causes problems.  bounces around too much

                    if verbose:
                        print('Star '+str(i)+' Refitting amp empirically')
                        print(str(amp))
                        
                    #if i==1: print(amp)
                    #if self.niter==2:
                    #    import pdb; pdb.set_trace()

            # No refit of stellar parameters
            else:
                model = psf(x,y,pars=[amp,xcen,ycen])

            #if self.niter>1:
            #    import pdb; pdb.set_trace()
                
            # Relculate reduced chi squared
            chisq = np.sum((flux-model.ravel())**2/err**2)/npix
            self.starchisq[i] = chisq
            # chi value, RMS of the residuals as a fraction of the amp
            rms = np.sqrt(np.mean(((flux-model.ravel())/self.staramp[i])**2))
            self.starrms[i] = rms
            
            #model = psf(x,y,pars=[amp,xcen,ycen])
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

        # Need to run model() to calculate amp/xcen/ycen for first couple iterations
        #if self.niter<=1 and refit:
        #    dum = self.model(x,*args,refit=refit)
        dum = self.model(x,*args,refit=True) #,verbose=True)            
            
        for i in range(self.nstars):
            amp = self.staramp[i]
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
            allpars = np.concatenate((np.array([amp,xcen,ycen]),np.array(args)))
            m,deriv = psf.jac(xdata,*allpars,allpars=True,retmodel=True)
            #if retmodel:
            #    m,deriv = psf.jac(xdata,*allpars,allpars=True,retmodel=True)
            #else:
            #    deriv = psf.jac(xdata,*allpars,allpars=True)                
            deriv = np.delete(deriv,[0,1,2],axis=1)  # remove stellar ht/xc/yc columns

            # Solve for the best amp, and then scale the derivatives (all scale with amp)
            #if self.niter>1 and refit:
            #    newamp = amp*np.median(flux/m)
            #    self.staramp[i] = newamp
            #    m *= (newamp/amp)
            #    deriv *= (newamp/amp)

            #if i==1: print(amp,newamp)
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

    def linesearch(self,xdata,bestpar,dbeta,m,jac):
        # Perform line search along search gradient
        flux = self.imflatten
        # Weights
        wt = 1/self.errflatten**2
        
        start_point = bestpar
        search_gradient = dbeta
        def obj_func(pp,m=None):
            """ chisq given the parameters."""
            if m is None:
                m = self.model(xdata,*pp)                        
            chisq = np.sum((flux.ravel()-m.ravel())**2 * wt.ravel())
            #print('obj_func: pp=',pp)
            #print('obj_func: chisq=',chisq)
            return chisq
        def obj_grad(pp,m=None,jac=None):
            """ Gradient of chisq wrt the parameters."""
            if m is None or jac is None:
                m,jac = self.jac(xdata,*pp,retmodel=True)
            # d chisq / d parj = np.sum( 2*jac_ij*(m_i-d_i))/sig_i**2)
            dchisq = np.sum( 2*jac * (m.ravel()-flux.ravel()).reshape(-1,1)
                             * wt.ravel().reshape(-1,1),axis=0)
            #print('obj_grad: pp=',pp)
            #print('obj_grad: dchisq=',dchisq)            
            return dchisq

        # Inside model() the parameters are limited to the PSF bounds()
        f0 = obj_func(start_point,m=m)
        # Do our own line search with three points and a quadratic fit.
        f1 = obj_func(start_point+0.5*search_gradient)
        f2 = obj_func(start_point+search_gradient)
        alpha = dln.quadratic_bisector(np.array([0.0,0.5,1.0]),np.array([f0,f1,f2]))
        alpha = np.minimum(np.maximum(alpha,0.0),1.0)  # 0<alpha<1
        if ~np.isfinite(alpha):
            alpha = 1.0
        # Use scipy.optimize.line_search()
        #grad0 = obj_grad(start_point,m=m,jac=jac)        
        #alpha,fc,gc,new_fval,old_fval,new_slope = line_search(obj_func, obj_grad, start_point, search_gradient, grad0,f0,maxiter=3)
        #if alpha is None:  # did not converge
        #    alpha = 1.0
        pars_new = start_point + alpha * search_gradient
        new_dbeta = alpha * search_gradient
        return alpha,new_dbeta

    def mklookup(self,order=0):
        """ Make an empirical look-up table for the residuals."""

        # Make the empirical EPSF
        cube = self.psf.resid(self.cat,self.image,fillvalue=np.nan)
        coords = (self.cat['x'].data,self.cat['y'].data)
        epsf,nbadstar,rms = mkempirical(cube,order=order,coords=coords,shape=self.image.shape,lookup=True)
        lookup = models.PSFEmpirical(epsf,imshape=self.image.shape,order=order,lookup=True)

        # DAOPHOT does some extra analysis to make sure the flux
        # in the residual component is okay

        # -make sure
        #  -to take the total flux into account (not varying across image)
        #  -make sure the amp=1 at center
        #  -make sure all PSF values are >=0
                             
        # Add the lookup table to the PSF model
        self.psf.lookup = lookup

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
            amp = self.staramp[i]
            xcen = self.starxcen[i]   
            ycen = self.starycen[i]
            bbox = self.bboxdata[i]
            model1 = psf(pars=[amp,xcen,ycen],bbox=bbox)
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
       Catalog with initial amp/x/y values for the stars to use to fit the PSF.
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
       Table of best-fitting amp/xcen/ycen values for the PSF stars.

    Example
    -------

    newpsf,pars,perror,psfcat = fitpsf(psf,image,cat)

    """

    t0 = time.time()
    print = utils.getprintfunc() # Get print function to be used locally, allows for easy logging   

    # Initialize the output catalog best-fitting values for the PSF stars
    dt = np.dtype([('id',int),('amp',float),('x',float),('y',float),('npix',int),('rms',float),
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
        coords = (cat['x'].data,cat['y'].data)
        epsf1,nbadstar1,rms1 = mkempirical(cube1,order=psf.order,coords=coords,shape=psf._shape)
        initpsf = models.PSFEmpirical(epsf1,imshape=image.shape,order=psf.order)
        pf = PSFFitter(initpsf,image,cat,fitradius=fitradius,verbose=False)
        # Fit the amp, xcen, ycen properly
        xdata = np.arange(pf.ntotpix)
        out = pf.model(xdata,[])
        # Put information into the psfcat table
        psfcat['amp'] = pf.staramp
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
        epsf,nbadstar,rms = mkempirical(cube,order=psf.order,coords=coords,shape=psf._shape)
        newpsf = models.PSFEmpirical(epsf,imshape=image.shape,order=psf.order)
        if verbose:
            print('Median RMS: '+str(np.median(pf.starrms)))
            print('dt = %.2f sec' % (time.time()-t0))
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

            # Perform line search
            alpha,new_dbeta = pf.linesearch(xdata,bestpar,dbeta,m,jac)
            
            if verbose:
                print('  pars = '+str(bestpar))
                print('  dbeta = '+str(dbeta))

            # Update the parameters
            oldpar = bestpar.copy()
            #bestpar = psf.newpars(bestpar,dbeta,bounds,maxsteps)
            bestpar = psf.newpars(bestpar,new_dbeta,bounds,maxsteps)  
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
    dt = np.dtype([('id',int),('amp',float),('x',float),('y',float),('npix',int),('rms',float),
                   ('chisq',float),('ixmin',int),('ixmax',int),('iymin',int),('iymax',int)])
    psfcat = np.zeros(len(cat),dtype=dt)
    if 'id' in cat.colnames:
        psfcat['id'] = cat['id']
    else:
        psfcat['id'] = np.arange(len(cat))+1
    psfcat['amp'] = pf.staramp
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
       Catalog with initial amp/x/y values for the stars to use to fit the PSF.
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
       Table of best-fitting amp/xcen/ycen values for the PSF stars.

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
    dt = np.dtype([('id',int),('amp',float),('x',float),('y',float),('npix',int),('rms',float),
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

        pf.mklookup(lorder)
        # Fit the stars again and get new RMS values
        xdata = np.arange(pf.ntotpix)
        out = pf.model(xdata,*pf.psf.params)
        newpsf = pf.psf.copy()
        # Update information in the output catalog
        ind1,ind2 = dln.match(outcat['id'],pcat['id'])
        outcat['reject'] = 1
        outcat['reject'][ind1] = 0
        outcat['amp'][ind1] = pf.staramp[ind2]
        outcat['x'][ind1] = pf.starxcen[ind2]
        outcat['y'][ind1] = pf.starycen[ind2]
        outcat['rms'][ind1] = pf.starrms[ind2]
        outcat['chisq'][ind1] = pf.starchisq[ind2]                
        if verbose:
            print('Median RMS: '+str(np.median(pf.starrms)))            
            
    if verbose:
        print('dt = %.2f sec' % (time.time()-t0))
    
    return newpsf, pars, perror, outcat

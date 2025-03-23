#!/usr/bin/env python

"""GROUPFIT.PY - Fit groups of stars in an image

"""

__authors__ = 'David Nidever <dnidever@montana.edu?'
__version__ = '20210826'  # yyyymmdd


import numpy as np
from numba import njit,types,from_dtype
from numba.experimental import jitclass
from numba_kdtree import KDTree
from . import models_numba as mnb, utils_numba as utils, getpsf_numba as gnb
from .clock_numba import clock

# Fit a PSF model to multiple stars in an image


@njit(cache=True)
#@njit
def getstarinfo(imshape,mask,xcen,ycen,hpsfnpix,fitradius,skyradius):
    """ Return a star's full footprint, fitted pixels, and sky pixels data."""
    # always return the same size
    # a distance of 6.2 pixels spans 6 full pixels but you could have
    # 0.1 left on one side and 0.1 left on the other side
    # that's why we have to add 2 pixels
    maxpix = int(3.14*skyradius**2)
    nskypix = int(2*skyradius+1)
    skybbox = utils.starbbox((xcen,ycen),imshape,int(skyradius))
    nsx = skybbox[1]-skybbox[0]
    nsy = skybbox[3]-skybbox[2]
    skyravelindex = np.zeros(maxpix,np.int64)-1
    nfpix = hpsfnpix*2+1
    #fbbox = utils.starbbox((xcen,ycen),imshape,hpsfnpix)
    # extra buffer is ALWAYS at the end of each dimension    
    fravelindex = np.zeros(nfpix*nfpix,np.int64)-1
    # Fitting pixels
    npix = int(np.floor(2*fitradius))+2
    bbox = utils.starbbox((xcen,ycen),imshape,fitradius)
    nx = bbox[1]-bbox[0]
    ny = bbox[3]-bbox[2]
    ravelindex = np.zeros(npix*npix,np.int64)-1
    fcount = 0
    count = 0
    skycount = 0
    for j in range(nskypix):
        y = j + skybbox[2]
        for i in range(nskypix):
            x = i + skybbox[0]
            if mask[y,x]:   # exclude bad pixels
                continue
            r = np.sqrt((x-xcen)**2 + (y-ycen)**2)
            if x>=skybbox[0] and x<=skybbox[1]-1 and y>=skybbox[2] and y<=skybbox[3]-1:
                #if x>=fbbox[0] and x<=fbbox[1]-1 and y>=fbbox[2] and y<=fbbox[3]-1:
                if r <= 1.0*skyradius and r >= 0.7*skyradius:
                    skymulti_index = (np.array([y]),np.array([x]))
                    skyravelindex[skycount] = utils.ravel_multi_index(skymulti_index,imshape)[0]
                    skycount += 1
                if r <= 1.0*hpsfnpix:
                    fmulti_index = (np.array([y]),np.array([x]))
                    fravelindex[fcount] = utils.ravel_multi_index(fmulti_index,imshape)[0]
                    fcount += 1
                if r <= fitradius:
                    multi_index = (np.array([y]),np.array([x]))
                    ravelindex[count] = utils.ravel_multi_index(multi_index,imshape)[0]
                    count += 1
    return (fravelindex,fcount,ravelindex,count,skyravelindex,skycount)

@njit(cache=True)
#@njit
def collatestarsinfo(imshape,mask,starx,stary,hpsfnpix,fitradius,skyradius):
    """ Get full footprint, fitted pixels, and sky pixels data for all stars."""
    nstars = len(starx)
    nfpix = 2*hpsfnpix+1
    npix = int(np.floor(2*fitradius))+2
    # Full footprint arrays
    fravelindex = np.zeros((nstars,nfpix*nfpix),np.int64)
    fndata = np.zeros(nstars,np.int32)
    # Fitting pixel arrays
    ravelindex = np.zeros((nstars,npix*npix),np.int64)
    ndata = np.zeros(nstars,np.int32)
    skyravelindex = np.zeros((nstars,int(3.14*skyradius**2)),np.int64)
    skyndata = np.zeros(nstars,np.int32)
    for i in range(nstars):
        out = getstarinfo(imshape,mask,starx[i],stary[i],hpsfnpix,fitradius,skyradius)
        # full footprint information
        fravelindex1,fn1,ravelindex1,n1,skyravelindex1,skyn1 = out
        fravelindex[i,:] = fravelindex1
        fndata[i] = fn1
        # fitting pixel information
        ravelindex[i,:] = ravelindex1
        ndata[i] = n1
        # sky pixel information
        skyravelindex[i,:] = skyravelindex1
        skyndata[i] = skyn1
    # Trim arrays
    maxfn = np.max(fndata)
    fravelindex = fravelindex[:,:maxfn]
    maxn = np.max(ndata)
    ravelindex = ravelindex[:,:maxn]
    maxskyn = np.max(skyndata)
    skyravelindex = skyravelindex[:,:maxskyn]
    return (fravelindex,fndata,ravelindex,ndata,skyravelindex,skyndata)

@njit(cache=True)
#@njit
def initstararrays(image,error,mask,tab,psfnpix,fitradius,skyradius,skyfit):
    """ Initialize all of the star arrays."""

    # Star arrays
    #------------
    #  -full footprint: pixels of a star within the psf radius and not masked
    #  -fitting pixels: pixels of a star within its fitting radius and not masked
    #  -sky pixels: pixels in an annulus around a star and not masked
    #  -flat pixels: all fitting pixels of all stars combined (unique),
    #                  these are the pixels that we are actually fitting
    # Full footprint information
    #   starravelindex -
    #   starndata - number of pixels for star each star
    # Fitting pixel information
    #   starfitravelindex - fitting pixel index into the raveled 1D full image/resid array
    #   starfitndata - number of fitting pixels for each star
    # Sky pixel information
    #   skyravelindex - sky pixel index into the raveled 1D full image/resid array
    #   skyndata - number of star pixels for each star
    # Flat information
    #   xflat/yflat - x/y values of the flat pixels
    #   indflat - index of the flat pixels into the full 1D raveled arrays
    #   starflat_index - index of all flat pixels that are within a star's full footprint
    #                         index into the flat 1D array
    #   starflat_ndata - number of flat pixels for each star
    
    nstars = len(tab)
    ny,nx = image.shape                             # save image dimensions, python images are (Y,X)
    imshape = np.array([ny,nx])
    im = image.copy().astype(np.float64)
    err = error.copy().astype(np.float64)
    msk = mask.copy().astype(np.bool_)
    xx,yy = utils.meshgrid(np.arange(nx),np.arange(ny))
    xx = xx.flatten()
    yy = yy.flatten()
    # Order stars by flux, brightest first
    si = np.argsort(tab[:,1])[::-1]       # largest amp first
    startab = tab[si]                     # ID, amp, xcen, ycen
    initpars = startab[:,1:4]             # amp, xcen, ycen
    # Initialize the parameter array
    initpars = startab[:,1:4]
    npars = nstars*3
    if skyfit:
        npars += 1
    pars = np.zeros(npars,float) # amp, xcen, ycen
    pars[0:3*nstars:3] = initpars[:,0]
    pars[1:3*nstars:3] = initpars[:,1]
    pars[2:3*nstars:3] = initpars[:,2]

    # Get information for all the stars
    xcen = pars[1:3*nstars:3]
    ycen = pars[2:3*nstars:3]
    hpsfnpix = psfnpix//2
    out = collatestarsinfo(imshape,msk,xcen,ycen,hpsfnpix,fitradius,skyradius)
    starravelindex,starndata,starfitravelindex,starfitndata,skyravelindex,skyndata = out

    # Put indices of all the unique fitted pixels into one array
    ntotpix = np.sum(starfitndata)
    allravelindex = np.zeros(ntotpix,np.int64)
    count = 0
    for i in range(nstars):
        n1 = starfitndata[i]
        ravelind1 = starfitravelindex[i,:n1]
        allravelindex[count:count+n1] = ravelind1
        count += n1
    allravelindex = np.unique(allravelindex)
    ravelindex = allravelindex
    ntotpix = len(allravelindex)

    # Combine all of the X and Y values (of the pixels we are fitting) into one array
    ntotpix = np.sum(starfitndata)
    xall = np.zeros(ntotpix,np.int32)
    yall = np.zeros(ntotpix,np.int32)
    count = 0
    for i in range(nstars):
        n1 = starfitndata[i]
        ravelindex1 = starfitravelindex[i,:n1]
        xdata1 = xx[ravelindex1]
        ydata1 = yy[ravelindex1]
        xall[count:count+n1] = xdata1
        yall[count:count+n1] = ydata1
        count += n1
    
    # Create 1D unraveled indices, python images are (Y,X)
    ind1 = utils.ravel_multi_index((yall,xall),imshape)
    # Get unique indices and inverse indices
    #   the inverse index list takes you from the full/duplicated pixels
    #   to the unique ones
    uind1,uindex1,invindex = utils.unique_index(ind1)
    ntotpix = len(uind1)
    ucoords = utils.unravel_index(uind1,image.shape)
    yflat = ucoords[:,0]
    xflat = ucoords[:,1]
    # x/y coordinates of the unique fitted pixels
    indflat = uind1
    
    # Save information on the "flattened" and unique fitted pixel arrays
    imflat = np.zeros(len(uind1),np.float64)
    imflat[:] = image.ravel()[uind1]
    errflat = np.zeros(len(uind1),np.float64)
    errflat[:] = error.ravel()[uind1]
    resflat = imflat.copy()
    
    # Add inverse index for the fitted pixels
    #  to be used with imflat/resflat/errflat/xflat/yflat/indflat
    maxfitpix = np.max(starfitndata)
    starfitinvindex = np.zeros((nstars,maxfitpix),np.int64)-1
    for i in range(nstars):
        n1 = starfitndata[i]
        if i==0:
            invlo = 0
        else:
            invlo = np.sum(starfitndata[:i])
        invindex1 = invindex[invlo:invlo+n1]
        starfitinvindex[i,:n1] = invindex1
    
    # For all of the fitting pixels, find the ones that
    # a given star contributes to (basically within its PSF radius)
    starflat_index = np.zeros((nstars,ntotpix),np.int64)-1
    starflat_ndata = np.zeros(nstars,np.int64)
    for i in range(ntotpix):
        x1 = xflat[i]
        y1 = yflat[i]
        for j in range(nstars):
            pars1 = pars[3*i:3*i+3]
            r = np.sqrt((xflat[i]-pars[1])**2 + (yflat[i]-pars[2])**2)
            if r <= psfnpix:
                starflat_index[j,starflat_ndata[j]] = i
                starflat_ndata[j] += 1
    maxndata = np.max(starflat_ndata)
    starflat_index = starflat_index[:,:maxndata]

    return (im,err,msk,xx,yy,pars,npars,
            starravelindex,starndata,starfitravelindex,starfitndata,skyravelindex,skyndata,
            xflat,yflat,indflat,imflat,errflat,resflat,ntotpix,
            starfitinvindex,starflat_index,starflat_ndata)

@njit(cache=True)
#@njit
def sky(image,modelim,method='sep',rin=None,rout=None):
    """ (Re)calculate the sky."""
    # Remove the current best-fit model
    resid = image-modelim    # remove model
    skyim = utils.sky(resid)
    # Calculate sky value for each star
    #  use center position
    for i in range(nstars):
        starsky[i] = skyim[int(np.round(starycen[i])),
                           int(np.round(starxcen[i]))]
    return starsky
        

    #     # Annulus aperture
    #     elif method=='annulus':
    #         if rin is None:
    #             rin = self.psf.fwhm()*1.5
    #         if rout is None:
    #             rout = self.psf.fwhm()*2.5
    #         positions = list(zip(self.starxcen,self.starycen))
    #         annulus = CircularAnnulus(positions,r_in=rin,r_out=rout)
    #         for i in range(self.nstars):
    #             annulus_mask = annulus[i].to_mask(method='center')
    #             annulus_data = annulus_mask.multiply(resid,fill_value=np.nan)
    #             data = annulus_data[(annulus_mask.data>0) & np.isfinite(annulus_data)]
    #             mean_sigclip, median_sigclip, _ = sigma_clipped_stats(data,stdfunc=dln.mad)
    #             self.starsky[i] = mean_sigclip
    #         if hasattr(self,'skyim') is False:
    #             self.skyim = np.zeros(self.image.shape,float)
    #         if self.skyim is None:
    #             self.skyim = np.zeros(self.image.shape,float)
    #         self.skyim += np.median(self.starsky)
    #     else:
    #         raise ValueError("Sky method "+method+" not supported")

@njit(cache=True)
#@njit
def psf(xdata,pars,psfdata):
    """ Thin wrapper for getting a PSF model for a single star."""
    xind,yind = xdata
    psftype,psfparams,psflookup,psforder,imshape = psfdata
    im1,_ = mnb.psf(xind,yind,pars,psftype,psfparams,psflookup,
                    imshape,deriv=False,verbose=False)
    return im1
        
@njit(cache=True)
#@njit
def psfjac(xdata,pars,psfdata):
    """ Thin wrapper for getting the PSF model and Jacobian for a single star."""
    xind,yind = xdata
    psftype,psfparams,psflookup,psforder,imshape = psfdata
    im1,jac1 = mnb.psf(xind,yind,pars,psftype,psfparams,psflookup,
                       imshape,deriv=True,verbose=False)
    return im1,jac1
    
@njit(cache=True)
#@njit
def model(psfdata,freezedata,flatdata,pars,trim=False,allparams=False,verbose=False):
    """ Calculate the model for the stars and pixels we are fitting."""

    if verbose==True:
        print('model: ',pars)

    # Unpack the PSF data
    psftype,psfparams,psflookup,psforder,imshape = psfdata
    # Unpack freeze information
    freezepars,freezestars = freezedata
    # Unpack flat information
    starflat_ndata,starflat_index,xflat,yflat,indflat,ntotpix = flatdata

    # Args are [amp,xcen,ycen] for all Nstars + sky offset
    # so 3*Nstars+1 parameters
    
    nstars = len(freezestars)
    npars = len(pars)
    nfreezestars = np.sum(freezestars)
    nfreezepars = np.sum(freezepars)
    nfreepars = npars-nfreezepars
    skyfit = (npars % 3 != 0)
    
    # Figure out the parameters of ALL the stars
    #  some stars and parameters are FROZEN
    # if nfreezepars>0 and allparams==False:
    #     allpars = np.zeros(len(freezepars),np.float64)
    #     allpars[np.where(freezepars==False)] = pars
    # else:
    #     allpars = pars
    allpars = pars

    allim = np.zeros(ntotpix,np.float64)
    
    # Loop over the stars and generate the model image        
    # ONLY LOOP OVER UNFROZEN STARS
    dostars = np.arange(nstars)[np.where(freezestars==False)]
    for i in dostars:
        pars1 = allpars[i*3:(i+1)*3]
        n1 = starflat_ndata[i]
        invind1 = starflat_index[i,:n1]
        xind1 = xflat[invind1]
        yind1 = yflat[invind1]
        xdata1 = (xind1,yind1)
        # we need the inverse index to the unique fitted pixels
        im1 = psf(xdata1,pars1,psfdata)
        allim[invind1] += im1

    if skyfit:
        allim += allpars[-1]  # add sky offset
        
    # if trim and nusepix<self.ntotpix:            
    #     unused = np.arange(self.ntotpix)[~usepix]
    #     allim = np.delete(allim,unused)
        
    return allim

@njit(cache=True)
#@njit
def fullmodel(psfdata,stardata,pars):
    """ Calculate the model for all the stars and the full footprint."""
    
    # Unpack the PSF data
    psftype,psfparams,psflookup,psforder,imshape = psfdata
    # Unpack flat information
    starravelindex,starndata,xx,yy = stardata

    nstars = len(starndata)
    npars = len(pars)
    im = np.zeros(imshape[0]*imshape[1],np.float64)
    
    # Loop over all stars and generate the model image        
    for i in range(nstars):
        pars1 = pars[i*3:(i+1)*3]
        n1 = starndata[i]
        ravelindex1 = starravelindex[i,:n1]
        xind1 = xx[ravelindex1]
        yind1 = yy[ravelindex1]
        xdata1 = (xind1,yind1)
        # we need the inverse index to the unique fitted pixels
        im1 = psf(xdata1,pars1,psfdata)
        im[ravelindex1] += im1
        
    return im

@njit(cache=True)
#@njit  
def jac(psfdata,freezedata,flatdata,pars,trim=False,allparams=False):
    """ Calculate the jacobian for the pixels and parameters we are fitting"""

    # Unpack the PSF data
    psftype,psfparams,psflookup,psforder,imshape = psfdata
    # Unpack freeze information
    freezepars,freezestars = freezedata
    # Unpack flat information
    starflat_ndata,starflat_index,xflat,yflat,indflat,ntotpix = flatdata
    
    # Args are [amp,xcen,ycen] for all Nstars + sky offset
    # so 3*Nstars+1 parameters

    nstars = len(freezestars)
    npars = len(pars)
    nfreezestars = np.sum(freezestars)
    nfreezepars = np.sum(freezepars)
    nfreepars = npars-nfreezepars
    skyfit = (npars % 3 != 0)

    # Figure out the parameters of ALL the stars
    #  some stars and parameters are FROZEN
    # if nfreezepars>0 and allparams==False:
    #     allpars = np.zeros(len(freezepars),np.float64)
    #     if len(pars) != (len(freezepars)-nfreezepars):
    #         print('problem')
    #         print('len(pars):',len(pars))
    #         print('len(freezepars):',len(freezepars))
    #         print('nfreezepars:',nfreezepars)
    #         return np.zeros(1,np.float64)+np.nan,np.zeros((1,1),np.float64)+np.nan
    #     allpars[np.where(freezepars==False)] = pars
    # else:
    #     allpars = pars
    allpars = pars
        
    im = np.zeros(ntotpix,np.float64)
    jac = np.zeros((ntotpix,len(pars)),np.float64)    # image covered by star
    usepix = np.zeros(ntotpix,np.bool_)
    
    # Loop over the stars and generate the model image        
    # ONLY LOOP OVER UNFROZEN STARS
    dostars = np.arange(nstars)[np.where(freezestars==False)]
    for i in dostars:
        pars1 = allpars[i*3:(i+1)*3]
        n1 = starflat_ndata[i]
        invind1 = starflat_index[i,:n1]
        xind1 = xflat[invind1]
        yind1 = yflat[invind1]
        xdata1 = (xind1,yind1)
        # we need the inverse index to the unique fitted pixels
        im1,jac1 = psfjac(xdata1,pars1,psfdata)
        jac[invind1,i*3] = jac1[:,0]
        jac[invind1,i*3+1] = jac1[:,1]
        jac[invind1,i*3+2] = jac1[:,2]
        im[invind1] += im1
        usepix[invind1] = True
        
    # Sky gradient
    if skyfit:
        jac[:,-1] = 1
            
    # Remove frozen columns
    #if nfreezepars>0 and allparams==False:
    if nfreezepars>0:
        origjac = jac.copy()
        jac = np.zeros((ntotpix,nfreepars),np.float64)
        freeparind, = np.where(freezepars==False)
        for count,i in enumerate(freeparind):
            jac[:,count] = origjac[:,i]

    # # Trim out unused pixels
    # if trim and nusepix<ntotpix:
    #     unused = np.arange(ntotpix)[~usepix]
    #     origjac = jac
    #     jac = np.zeros((nusepix,nfreepars),np.float64)
    #     usepixind, = np.where(usepix==True)
    #     im = im[usepixind]
    #     for count,i in enumerate(usepixind):
    #         for j in range(nfreepars):
    #             jac[count,j] = origjac[i,j]
        
    return im,jac

@njit(cache=True)
#@njit
def chisqflat(freezedata,flatdata,psfdata,resflat,errflat,pars):
    """ Return chi-squared of the flat data"""
    # Note this ignores any frozen stars
    bestmodel = model(psfdata,freezedata,flatdata,pars,False,True)   # trim, allparams
    chisq = np.sum(resflat**2/errflat**2)
    return chisq

@njit(cache=True)
#@njit
def cov(psfdata,freezedata,covflatdata,pars):
    """ Determine the covariance matrix."""
    
    # Unpack the PSF data
    psftype,psfparams,psflookup,psforder,imshape = psfdata
    # Unpack freeze information
    freezepars,freezestars = freezedata
    # Unpack flat information
    starflat_ndata,starflat_index,xflat,yflat,indflat,ntotpix,imflat,errflat,skyflat = covflatdata
    
    # https://stats.stackexchange.com/questions/93316/
    #             parameter-uncertainty-after-non-linear-least-squares-estimation
    # more background here, too: http://ceres-solver.org/nnls_covariance.html        
    # Hessian = J.T * T, Hessian Matrix
    #  higher order terms are assumed to be small
    # https://www8.cs.umu.se/kurser/5DA001/HT07/lectures/lsq-handouts.pdf
    flatdata2 = (starflat_ndata,starflat_index,xflat,yflat,indflat,ntotpix)
    bestmodel,mjac = jac(psfdata,freezedata,flatdata2,pars,False,True)   # trim, allparams
    # Weights
    #   If weighted least-squares then
    #   J.T * W * J
    #   where W = I/sig_i**2
    wt = np.diag(1/errflat**2)
    hess = mjac.T @ (wt @ mjac)
    # cov = H-1, covariance matrix is inverse of Hessian matrix
    cov_orig = utils.inverse(hess)
    # Rescale to get an unbiased estimate
    # cov_scaled = cov * (RSS/(m-n)), where m=number of measurements, n=number of parameters
    # RSS = residual sum of squares
    #  using rss gives values consistent with what curve_fit returns
    resid = imflat-bestmodel-skyflat
    # Use chi-squared, since we are doing the weighted least-squares and weighted Hessian
    chisq = np.sum(resid**2/errflat**2)        
    cov = cov_orig * (chisq/(ntotpix-len(pars)))  # what MPFITFUN suggests, but very small
        
    return cov

@njit(cache=True)
#@njit
def dofreeze(frzpars,pars,freezedata,flatdata,psfdata,resid,resflat):
    """ Freeze par/stars."""

    freezepars,freezestars = freezedata
    starflat_ndata,starflat_index,xflat,yflat,indflat,ntotpix = flatdata
    nstars = len(freezestars)
    
    # Freeze parameters/stars that converged
    #  also subtract models of fixed stars
    #  also return new free parameters
    #frzpars = percdiff<=minpercdiff
    freeparsind, = np.where(freezepars==False)
    # Update freeze values for "free" parameters
    tempfreezepars = freezepars.copy()
    tempfreezepars[np.where(freezepars==False)] = frzpars   # stick in updated values for "free" parameters
    # Only freeze stars, not individual parameters
    oldfreezestars = freezestars.copy()
    freezestars = np.sum(tempfreezepars[0:3*nstars].copy().reshape(nstars,3),axis=1)==3
    # Subtract model for newly frozen stars
    newfreezestars, = np.where((oldfreezestars==False) & (freezestars==True))
    
    # Freezing more stars
    if len(newfreezestars)>0:
        for i in range(len(newfreezestars)):
            istar = newfreezestars[i]
            freezestars[istar] = True
            freezepars[3*istar:3*istar+3] = True
            # Subtract model of frozen star from residuals
            pars1 = pars[3*istar:3*istar+3]
            n1 = starflat_ndata[istar]
            invind1 = starflat_index[istar,:n1]
            xind1 = xflat[invind1]
            yind1 = yflat[invind1]
            xdata1 = (xind1,yind1)
            im1 = psf(xdata1,pars1,psfdata)
            resflat[invind1] -= im1
            resid[indflat[invind1]] -= im1
                    
        # Get the new array of free parameters
        freeparsind, = np.where(freezepars==False)
        bestpar = pars[freeparsind]
        nfreezepars = np.sum(freezepars)
        nfreezestars = np.sum(freezestars)

    return freezepars,freezestars,resid,resflat

    
@njit(cache=True)
#@njit
def groupfit(psftype,psfparams,psfnpix,psflookup,psfflux,
             image,error,mask,tab,fitradius,maxiter=10,
             minpercdiff=0.5,reskyiter=2,nofreeze=False,
             skyfit=False,verbose=False):
    """
    Fit PSF to a group of stars simultaneously.

    Parameters
    ----------
    psftype : int
       PSF type.
    psfparams: numpy array
       PSF analytical parameters.
    psfnpix : int
       Number of pixels in the PSF footprint.
    psflookup : numpy array
       PSF lookup table.  Must be 3D.
    psfflux : float
       PSF flux.
    image : numpy array
       The image to fit.
    error : numpy array
       Uncertainty for image.
    mask : numpy array
       Boolean mask for image.  True - bad, False - good.
    tab : numpy array
       Table of initial guesses for each star.
       id, amp, xcen, ycen
    maxiter : int, optional
       Maximum number of iterations to allow.  Only for methods "cholesky", "qr" or "svd".
       Default is 10.
    minpercdiff : float, optional
       Minimum percent change in the parameters to allow until the solution is
       considered converged and the iteration loop is stopped.  Only for methods
       "cholesky", "qr" and "svd".  Default is 0.5.
    reskyiter : int, optional
       After how many iterations to re-calculate the sky background. Default is 2.
    nofreeze : boolean, optional
       Do not freeze any parameters even if they have converged.  Default is False.
    skyfit : boolean, optional
       Fit a constant sky offset with the stellar parameters.  Default is False.
    verbose : boolean, optional
       Verbose output.

    Returns
    -------
    out : table
       Table of best-fitting parameters for each star.
       id, amp, amp_error, x, x_err, y, y_err, sky
    model : numpy array
       Best-fitting model of the stars and sky background.
    sky : numpy array
       Best-fitting sky image.

    Examples
    --------

    out,model,sky = groupfit(psf,image,error,mask,tab)
    
    """
    start = clock()
    
    nstars = len(tab)
    skyradius = psfnpix//2 + 10
    ny,nx = image.shape                             # save image dimensions, python images are (Y,X)
    imshape = np.array([ny,nx])
    
    # PSF information
    #----------------
    _,_,npsforder = psflookup.shape
    psforder = 0
    if npsforder==4:
        psforder = 1
    if psflookup.ndim != 3:
        raise Exception('psflookup must have 3 dimensions')
    psforder = psflookup.shape[2]
    if psforder>1:
        psforder = 1
    # Package up the PSF information into a tuple to pass to the functions
    psfdata = (psftype,psfparams,psflookup,psforder,imshape)

    # Star arrays
    #------------
    #  -full footprint: pixels of a star within the psf radius and not masked
    #  -fitting pixels: pixels of a star within its fitting radius and not masked
    #  -sky pixels: pixels in an annulus around a star and not masked
    #  -flat pixels: all fitting pixels of all stars combined (unique),
    #                  these are the pixels that we are actually fitting
    # Full footprint information
    #   starravelindex -
    #   starndata - number of pixels for star each star
    # Fitting pixel information
    #   starfitravelindex - fitting pixel index into the raveled 1D full image/resid array
    #   starfitndata - number of fitting pixels for each star
    # Sky pixel information
    #   skyravelindex - sky pixel index into the raveled 1D full image/resid array
    #   skyndata - number of star pixels for each star
    # Flat information
    #   xflat/yflat - x/y values of the flat pixels
    #   indflat - index of the flat pixels into the full 1D raveled arrays
    #   starflat_index - index of all flat pixels that are within a star's full footprint
    #                         index into the flat 1D array
    #   starflat_ndata - number of flat pixels for each star
    
    # Initialize the star arrays
    initdata = initstararrays(image,error,mask,tab,psfnpix,fitradius,skyradius,skyfit)
    im,err,msk,xx,yy,pars,npars = initdata[:7]
    starravelindex,starndata,starfitravelindex,starfitndata,skyravelindex,skyndata = initdata[7:13]
    xflat,yflat,indflat,imflat,errflat,resflat,ntotpix = initdata[13:20]
    starfitinvindex,starflat_index,starflat_ndata = initdata[20:]
    
    # Package up the star information into tuples for easy transport
    stardata = (starravelindex,starndata,xx,yy)
    flatdata = (starflat_ndata,starflat_index,xflat,yflat,indflat,ntotpix)

    # Image arrays
    #-------------
    # Full arrays:
    #   im/err/mask - full 1D raveled image, error and mask arrays
    #   resid - full 1D raveled residual image with the smooth sky subtracted
    #             AND any frozen stars subtracted
    # Flat arrays:
    #   indflat - index of the flat pixels into the full 1D raveled arrays
    #   imflat/errflat - 1D raveled image and error arrays for the flat/fitting pixels
    #   skyflat - smooth sky values for the flat pixels
    #   resflat - flat pixel values of "resid"
    
    # Create initial sky image
    #  improve initial sky estimate by removing initial models
    tresid = np.zeros(imshape[0]*imshape[1],np.float64)
    tresid[:] = im.copy().astype(np.float64).flatten()
    for i in range(nstars):
        pars1 = pars[3*i:3*i+3]
        n1 = starndata[i]
        ravelind1 = starravelindex[i]
        xind1 = xx[ravelind1]
        yind1 = yy[ravelind1]
        xdata1 = (xind1,yind1)
        m = psf(xdata1,pars1,psfdata)
        tresid[ravelind1] -= m
    skyim = utils.sky(tresid.copy().reshape(imshape[0],imshape[1])).flatten()
    skyflat = skyim[indflat]

    # Initialize RESID and subtract initial smooth sky
    resid = np.zeros(imshape[0]*imshape[1],np.float64)
    resid[:] = im.copy().astype(np.float64).flatten()   # flatten makes it easier to modify
    resid[:] -= skyim          # subtract smooth sky
    resflat = resid[indflat]   # initialize resflat

    
    # Perform the fitting
    #--------------------

    # Initialize arrays for the stars and freezing parameters
    starsky = np.zeros(nstars,np.float64)
    starniter = np.zeros(nstars,np.int64)
    starchisq = np.zeros(nstars,np.float64)
    starrms = np.zeros(nstars,np.float64)
    freezestars = np.zeros(nstars,np.bool_)
    freezepars = np.zeros(len(pars),np.bool_)
    
    # Initial estimates
    initpar = pars.copy()
    # Make bounds
    #  this requires all 3*Nstars parameters to be input
    bounds = utils.mkbounds(initpar,imshape,2)    
    
    # Iterate
    bestpar_all = initpar.copy()
    allpars = initpar.copy()
    bestpar = initpar.copy()
    # centroids fixed, only keep amps
    #if recenter==False:
    #    initpar = initpar[0::3]  # amps and sky offset
    # sky offset fixed
    #if skyfit==False:
    #    initpar = initpar[0:-1]  # no sky offset
    niter = 1
    maxpercdiff = 1e10
    maxsteps = utils.steps(initpar)  # maximum steps, requires all 3*Nstars parameters
    nfreepars = len(pars)
    nfreestars = nstars
    # While loop
    #   keep looping until: (a) reached max iterations, (b) changes are tiny
    #                      or (c) there are no free stars left
    while (niter<maxiter and maxpercdiff>minpercdiff and nfreestars>0):
        start0 = clock()

        # bestpar: current best values for the free parameters
        # pars: current best values for ALL parameters
        
        # --- Jacobian solvers ---
        # Get the Jacobian and model
        #  only for pixels that are affected by the "free" parameters
        freezedata = (freezepars,freezestars)
        model0,j = jac(psfdata,freezedata,flatdata,pars,False,True)  # trim, allparams
        # Residuals
        skyflat = skyim[indflat]
        dy = resflat-skyflat-model0
        # Weights
        wt = 1/errflat**2
        # Solve linear least squares with the Jacobian
        dbeta_free = utils.qr_jac_solve(j,dy,weight=wt)
        dbeta_free[~np.isfinite(dbeta_free)] = 0.0  # deal with NaNs, shouldn't happen
        dbeta = np.zeros(len(pars),np.float64)
        dbeta[np.where(freezepars==False)] = dbeta_free
        
        # --- Perform line search ---
        #  move the solution along the dbeta vector to find the lowest chisq
        #  get models: pars + 0*dbeta, pars + 0.5*dbeta, pars + 1.0*dbeta
        freezedata = (freezepars,freezestars)
        model1 = model(psfdata,freezedata,flatdata,pars+0.5*dbeta,False,True)   # trim, allparams
        model2 = model(psfdata,freezedata,flatdata,pars+dbeta,False,True)       # trim, allparams
        chisq0 = np.sum((resflat-model0)**2/errflat**2)
        chisq1 = np.sum((resflat-model1)**2/errflat**2)
        chisq2 = np.sum((resflat-model2)**2/errflat**2)
        if verbose:
            print('linesearch:',chisq0,chisq1,chisq2)
        # Use quadratic bisector on the three points to find the lowest chisq 
        alpha = utils.quadratic_bisector(np.array([0.0,0.5,1.0]),
                                         np.array([chisq0,chisq1,chisq2]))
        # The bisector can be negative (outside of the acceptable range) if the chisq shape is concave
        if alpha <= 0 and np.min(np.array([chisq0,chisq1,chisq2]))==chisq2:
            alpha = 1.0
        #alpha = np.minimum(np.maximum(alpha,0.0),1.0)  # 0<alpha<1
        alpha = utils.clip(alpha,0.0,1.0)  # 0<alpha<1
        if np.isfinite(alpha)==False:
            alpha = 1.0
        pars_new = pars + alpha * dbeta
        new_dbeta_free = alpha * dbeta_free
        new_dbeta = np.zeros(len(pars),float)
        new_dbeta[np.where(freezepars==False)] = new_dbeta_free
        
        # Update the free parameters and impose step limits and bounds
        maxsteps = utils.steps(pars)  # maximum steps, requires all 3*Nstars parameters
        bounds = utils.mkbounds(pars,imshape,2)   
        oldpar = bestpar.copy()
        oldpar_all = pars.copy()
        bestpar_all = utils.newpars(pars,new_dbeta,bounds,maxsteps)            
        bestpar = bestpar_all[np.where(freezepars==False)]
        pars[np.where(freezepars==False)] = bestpar
        
        # Check differences and percent changes
        diff_all = np.abs(bestpar_all-oldpar_all)
        percdiff_all = diff_all.copy()*0
        percdiff_all[0:3*nstars:3] = diff_all[0:3*nstars:3]/np.maximum(oldpar_all[0:3*nstars:3],0.0001)*100  # amp
        percdiff_all[1:3*nstars:3] = diff_all[1:3*nstars:3]*100               # x
        percdiff_all[2:3*nstars:3] = diff_all[2:3*nstars:3]*100               # y
        if skyfit:
            percdiff_all[-1] = diff_all[-1]/np.maximum(np.abs(oldpar_all[-1]),1)*100
        diff = diff_all[np.where(freezepars==False)]
        percdiff = percdiff_all[np.where(freezepars==False)]
        
        # Freeze parameters/stars that converged
        #  this will subtract the model of converged stars from RESID and RESFLAT
        if nofreeze==False:
           frzpars = percdiff<=minpercdiff
           freezedata = (freezepars,freezestars)
           out = dofreeze(frzpars,pars,freezedata,flatdata,psfdata,resid,resflat)
           freezepars,freezestars,resid,resflat = out
        nfreestars = np.sum(freezestars==False)
        
        # Get model and chisq
        freezedata = (freezepars,freezestars)
        flatdata = (starflat_ndata,starflat_index,xflat,yflat,indflat,ntotpix)
        bestmodel = model(psfdata,freezedata,flatdata,pars,False,True)   # trim, allparams
        chisq = np.sum(resflat**2/errflat**2)
        
        if verbose:
            print('Iter = ',niter)
            print('Pars = ',pars)
            print('chisq = ',chisq)
            
        # Re-estimate the sky
        if niter % reskyiter == 0:
            if verbose:
                print('Re-estimating the sky')
            prevsky = skyim.copy()            
            # Get model of full footprints
            bestfullmodel = fullmodel(psfdata,stardata,pars)
            # Remove the current best-fit model and re-estimate the sky
            tresid = im-bestfullmodel.copy().reshape(imshape[0],imshape[1])
            skyim = utils.sky(tresid).flatten()
            # Update resid
            #  add previous sky back in
            # Remake the resid array using the new smooth sky
            resid[:] += prevsky
            resid[:] -= skyim   # subtract smooth sky
            resflat = resid[indflat]
            skyflat = skyim[indflat]
            
        if verbose:
            print('iter dt =',(clock()-start0)/1e9,'sec.')
                
        niter += 1     # increment counter
        
    # Check that all starniter are set properly
    #  if we stopped "prematurely" then not all stars were frozen
    #  and didn't have starniter set
    starniter[np.where(starniter==0)] = niter
    
    # Unfreeze everything
    freezepars[:] = False
    freezestars[:] = False
    # Make final model
    finalmodel = fullmodel(psfdata,stardata,pars).copy().reshape(imshape[0],imshape[0])
    
    # Estimate uncertainties
    #   calculate covariance matrix
    freezedata = (freezepars,freezestars)
    covflatdata = (starflat_ndata,starflat_index,xflat,yflat,indflat,ntotpix,imflat,errflat,skyflat)
    cov1 = cov(psfdata,freezedata,covflatdata,pars)
    perror = np.sqrt(np.diag(cov1))
    
    if verbose:
        print('Best-fitting parameters: ',pars)
        print('Errors: ',perror)

    # # Calculate smooth sky value for each star
    # #  use center position
    # for i in range(nstars):
    #     x1 = int(np.round(starxcen[i]))
    #     y1 = int(np.round(starycen[i]))
    #     sind1 = utils.ravel_multi_index((y1,x1),imshape)
    #     starsky[i] = skyim[sind1]
    
    # Put in catalog
    outtab = np.zeros((nstars,15),np.float64)
    outtab[:,0] = np.arange(nstars)+1                # id
    outtab[:,1] = pars[0:3*nstars:3]                 # amp
    outtab[:,2] = perror[0:3*nstars:3]               # amp_error
    outtab[:,3] = pars[1:3*nstars:3]                 # x
    outtab[:,4] = perror[1:3*nstars:3]               # x_error
    outtab[:,5] = pars[2:3*nstars:3]                 # y
    outtab[:,6] = perror[2:3*nstars:3]               # y_error
    if skyfit:
        outtab[:,7] = starsky + pars[-1]             # sky
    else:
        outtab[:,7] = starsky                        # sky
    outtab[:,8] = outtab[:,1]*psfflux                # flux
    outtab[:,9] = outtab[:,2]*psfflux                # flux_error
    outtab[:,10] = -2.5*np.log10(np.maximum(outtab[:,8],1e-10))+25.0   # mag
    outtab[:,11] = (2.5/np.log(10))*outtab[:,9]/outtab[:,8]            # mag_error
    # rms
    # chisq
    outtab[:,14] = starniter                      # niter, what iteration it converged on
    
    # Recalculate chi-squared and RMS of fit for each star
    for i in range(nstars):
        n1 = starfitndata[i]
        ravelind1 = starfitravelindex[i,:n1]
        xind1 = xx[ravelind1]
        yind1 = yy[ravelind1]
        flux1 = image.ravel()[ravelind1].copy() #[yind1,xind1].copy()
        err1 = error.ravel()[ravelind1].copy()  #[yind1,xind1]
        pars1 = np.zeros(3,np.float64)
        pars1[0] = outtab[i,1]  # amp
        pars1[1] = outtab[i,3]  # x
        pars1[2] = outtab[i,5]  # y
        xdata1 = (xind1,yind1)
        model1 = psf(xdata1,pars1,psfdata)
        sky1 = outtab[i,7]
        chisq1 = np.sum((flux1-sky1-model1)**2/err1**2)/n1
        outtab[i,13] = chisq1
        # chi value, RMS of the residuals as a fraction of the amp
        rms1 = np.sqrt(np.mean(((flux1-sky1-model1)/pars1[0])**2))
        outtab[i,12] = rms1

    if verbose:
        print('dt =',(clock()-start)/1e9,'sec.')
 
    return outtab,finalmodel,skyim

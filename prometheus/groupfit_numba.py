#!/usr/bin/env python

"""GROUPFIT.PY - Fit groups of stars in an image

"""

__authors__ = 'David Nidever <dnidever@montana.edu?'
__version__ = '20210826'  # yyyymmdd


import os
import sys
import numpy as np
import time
import scipy
import warnings
#from astropy.io import fits
#from astropy.table import Table
#import astropy.units as u
#from scipy.optimize import curve_fit, least_squares, line_search
#from scipy.interpolate import interp1d
#from scipy import sparse
##from astropy.nddata import CCDData,StdDevUncertainty
#from dlnpyutils import utils as dln, bindata
import copy
#import logging
#import time
#import matplotlib
#import sep
#from photutils.aperture import CircularAnnulus
#from astropy.stats import sigma_clipped_stats
#from . import leastsquares as lsq,utils
#from .ccddata import CCDData,BoundingBox
from numba import njit,types,from_dtype
from numba.experimental import jitclass
from numba_kdtree import KDTree
from . import models_numba as mnb, utils_numba as utils, getpsf_numba as gnb
from .clock_numba import clock

# Fit a PSF model to multiple stars in an image


# For each star's footprint, save the indices into the whole image


#@njit(cache=True)
@njit
def getstar(imshape,mask,xcen,ycen,hpsfnpix,fitradius,skyradius):
    """ Return a star's full footprint and fitted pixels data."""
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

#@njit(cache=True)
@njit
def collatestars(imshape,mask,starx,stary,hpsfnpix,fitradius,skyradius):
    """ Get full footprint and fitted pixels data for all stars."""
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
        out = getstar(imshape,mask,starx[i],stary[i],hpsfnpix,fitradius,skyradius)
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

@njit
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
        
    #     # SEP smoothly varying background
    #     if method=='sep':
    #         bw = np.maximum(int(self.nx/10),64)
    #         bh = np.maximum(int(self.ny/10),64)
    #         bkg = sep.Background(resid, mask=None, bw=bw, bh=bh, fw=3, fh=3)
    #         self.skyim = bkg.back()
    #         # Calculate sky value for each star
    #         #  use center position
    #         self.starsky[:] = self.skyim[np.round(self.starycen).astype(int),np.round(self.starxcen).astype(int)]
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


# @njit
# def starfitchisq(i):
#     """ Return chisq of current best-fit for one star."""
#     pars1,xind1,yind1,ravelind1,invind1 = getstarfit(i)
#     n1 = starfitnpix(i)
#     flux1 = image.ravel()[ravelind1].copy()
#     err1 = error.ravel()[ravelind1].copy()
#     model1 = psf(xind1,yind1,pars1)
#     sky1 = pars[-1]
#     chisq1 = np.sum((flux1-sky1-model1)**2/err1**2)/n1
#     return chisq1
    
# def starfitrms(i):
#     """ Return rms of current best-fit for one star."""
#     pars1,xind1,yind1,ravelind1,invind1 = getstarfit(i)
#     n1 = starfitnpix(i)
#     flux1 = image.ravel()[ravelind1].copy()
#     err1 = error.ravel()[ravelind1].copy()
#     model1 = psf(xind1,yind1,pars1)
#     sky1 = pars[-1]
#     # chi value, RMS of the residuals as a fraction of the amp
#     rms1 = np.sqrt(np.mean(((flux1-sky1-model1)/pars1[0])**2))
#     return rms1

# @njit    
# def freeze(allpars,freezepars,freezestars,pars,frzpars):
#     """ Freeze stars and parameters"""
#     # PARS: best-fit values of free parameters
#     # FRZPARS: boolean array of which "free" parameters
#     #            should now be frozen

#     nstars = (len(allstars)-1)//3
    
#     # Update all the free parameters
#     allpars[np.where(freezepars==False)] = pars

#     # Update freeze values for "free" parameters
#     freezepars[np.where(freezepars==False)] = frzpars   # stick in the new values for the "free" parameters
        
#     # Check if we need to freeze any new parameters
#     nfrz = np.sum(frzpars)
#     if nfrz==0:
#         return pars
        
#     # Freeze new stars
#     oldfreezestars = freezestars.copy()
#     freezestars = np.sum(freezepars[0:3*nstars].copy().reshape(nstars,3),axis=1)==3
#     # Subtract model for newly frozen stars
#     newfreezestars, = np.where((oldfreezestars==False) & (freezestars==True))
#     if len(newfreezestars)>0:
#         # add models to a full image
#         # WHY FULL IMAGE??
#         newmodel = np.zeros(imshape[0]*imshape[1],np.float64)
#         for i in newfreezestars:
#             # Save on what iteration this star was frozen
#             starniter[i] = niter+1
#             #print('freeze: subtracting model for star ',i)
#             pars1,xind1,yind1,ravelind1,invind1 = getstarfit(i)
#             n1 = len(xind1)
#             im1 = psf(xind1,yind1,pars1)
#             newmodel[ravelind1] += im1
#         # Only keep the pixels being fit
#         #  and subtract from the residuals
#         newmodel1 = newmodel[indflat]
#         resflat -= newmodel1
  
#     # Return the new array of free parameters
#     frzind = np.arange(len(frzpars))[np.where(frzpars==True)]
#     pars = np.delete(pars,frzind)
#     return pars

    
# def modelstar(pars,xind,yind,ravelind,full=False):
#     """ Return model of one star (full footprint) with the current best values."""
#     m = psf(xind,yind,pars)        
#     if full==True:
#         modelim = np.zeros((imshape[0]*imshape[1]),np.float64)
#         modelim[ravelind] = m
#         modelim = modelim.reshape((imshape[0],imshape[1]))
#     else:
#         bbox = self.star_bbox[i,:]
#         nx = bbox[1]-bbox[0]+1
#         ny = bbox[3]-bbox[2]+1
#         xind1 = xind-bbox[0]
#         yind1 = yind-bbox[2]
#         ind1 = utils.ravel_multi_index((xind1,yind1),(ny,nx))
#         modelim = np.zeros(nx*ny,np.float64)
#         modelim[ind1] = m
#         modelim = modelim.reshape((ny,nx))
#     return modelim

# def modelstarfit(self,i,full=False):
#     """ Return model of one star (only fitted pixels) with the current best values."""
#     pars,xind,yind,ravelind,invind = self.getstarfit(i)
#     m = self.psf(xind,yind,pars)        
#     if full==True:
#         modelim = np.zeros((self.imshape[0]*self.imshape[1]),np.float64)
#         modelim[ravelind] = m
#         modelim = modelim.reshape((self.imshape[0],self.imshape[1]))
#     else:
#         bbox = self.starfit_bbox[i,:]
#         nx = bbox[1]-bbox[0]+1
#         ny = bbox[3]-bbox[2]+1
#         xind1 = xind-bbox[0]
#         yind1 = yind-bbox[2]
#         ind1 = utils.ravel_multi_index((xind1,yind1),(ny,nx))
#         modelim = np.zeros(nx*ny,np.float64)
#         modelim[ind1] = m
#         modelim = modelim.reshape((ny,nx))
#     return modelim
    

# def modelim(imshape):
#     """ This returns the full image of the current best model (no sky)
#             using the PARS values."""
#     modelim = np.zeros((imshape[0],imshape[1]),np.float64).ravel()
#     for i in range(nstars):
#         pars1,xind1,yind1,ravelindex1 = self.getstar(i)
#         modelim1 = psf(xind1,yind1,pars1)
#         modelim[ravelindex1] += modelim1
#     modelim = modelim.reshape((imshape[0],imshape[1]))
#     return modelim

@njit
def psf(xdata,pars,psfdata):
    """ Thin wrapper for getting a single PSF model."""
    xind,yind = xdata
    psftype,psfparams,psflookup,psforder,imshape = psfdata
    im1,_ = mnb.psf(xind,yind,pars,psftype,psfparams,psflookup,
                    imshape,deriv=False,verbose=False)
    return im1
        
        
@njit
def psfjac(xdata,pars,psfdata):
    """ Thin wrapper for getting a single PSF model and Jacobian."""
    xind,yind = xdata
    psftype,psfparams,psflookup,psforder,imshape = psfdata
    im1,jac1 = mnb.psf(xind,yind,pars,psftype,psfparams,psflookup,
                       imshape,deriv=True,verbose=False)
    return im1,jac1
    

@njit
def model(psfdata,freezedata,flatdata,pars,trim=False,allparams=False,verbose=False):
    """ Calculate the model for the stars and pixels we are fitting."""

    if verbose==True:
        print('model: ',pars)

    # Unpack the PSF data
    psftype,psfparams,psflookup,psforder,imshape = psfdata
    # Unpack freeze information
    freezepars,freezestars = freezedata
    # Unpack flat information
    starflat_ndata,starflat_index,xflat,yflat,ntotpix = flatdata

    # Args are [amp,xcen,ycen] for all Nstars + sky offset
    # so 3*Nstars+1 parameters
    
    nstars = len(freezestars)
    npars = len(pars)
    nfreezestars = np.sum(freezestars)
    nfreezepars = np.sum(freezepars)
    nfreepars = npars-nfreezepars
        
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
        
    allim += allpars[-1]  # add sky offset
        
    # if trim and nusepix<self.ntotpix:            
    #     unused = np.arange(self.ntotpix)[~usepix]
    #     allim = np.delete(allim,unused)
        
    return allim

@njit
def fullmodel(psfdata,stardata,pars):
    """ Calculate the model for the stars and pixels we are fitting."""
    
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

@njit  
def jac(psfdata,freezedata,flatdata,pars,trim=False,allparams=False):
      #  freezepars,freezestars,
      #  starflat_ndata,starflat_index,xflat,yflat,ntotpix,
      #  pars,trim=False,allparams=False):
    #,retmodel=False,trim=False,allparams=False,verbose=None):
    """ Calculate the jacobian for the pixels and parameters we are fitting"""
    
    # if verbose is None and verbose:
    #     print('jac: ',njaciter,args)

    # Unpack the PSF data
    psftype,psfparams,psflookup,psforder,imshape = psfdata
    # Unpack freeze information
    freezepars,freezestars = freezedata
    # Unpack flat information
    starflat_ndata,starflat_index,xflat,yflat,ntotpix = flatdata
    
    # Args are [amp,xcen,ycen] for all Nstars + sky offset
    # so 3*Nstars+1 parameters

    nstars = len(freezestars)
    npars = len(pars)
    nfreezestars = np.sum(freezestars)
    nfreezepars = np.sum(freezepars)
    nfreepars = npars-nfreezepars
    
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
    
    print('jac len(allpars):',len(allpars))
    print('jac len(pars):',len(pars))
        
    im = np.zeros(ntotpix,np.float64)
    jac = np.zeros((ntotpix,len(pars)),np.float64)    # image covered by star
    usepix = np.zeros(ntotpix,np.bool_)
    
    # Loop over the stars and generate the model image        
    # ONLY LOOP OVER UNFROZEN STARS
    dostars = np.arange(nstars)[np.where(freezestars==False)]
    print('jac dostars:',dostars)
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

        if np.sum(~np.isfinite(im1))>0:
            print('model has some nans')
            print('pars:',pars1)
            print(im1)
        if np.sum(~np.isfinite(jac1))>0:
            print('jac has some nans')
            print('pars:',pars1)
            print(jac1)
        
    # Sky gradient
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

    if np.sum(~np.isfinite(im))>0:
        print('model has some nans')
        print('pars:',pars)
        print(im)
    if np.sum(~np.isfinite(jac))>0:
        print('jac has some nans')
        print('pars:',pars)
        print(jac)

    print('jac() im:',im)
        
    return im,jac

# @njit
# def chisq(pars):
#     """ Return chi-squared """
#     flux = resflat[usepix]-skyflat[usepix]
#     wt = 1/errflat[usepix]**2
#     bestmodel = model(pars,False,True)   # allparams,Trim
#     resid = flux-bestmodel[usepix]
#     chisq1 = np.sum(resid**2/errflat[usepix]**2)
#     return chisq1

# @njit
# def score(x,y,err,im,pars):
#     """
#     The score is the partial derivative of the ln likelihood
#     """

#     m,j = self.psfjac(x,y,pars)

#     # ln likelihood is
#     # ln likelihood = -0.5 * Sum( (y_i - m_i)**2/err_i**2 + ln(2*pi*err_i**2))
#     #                 -0.5 * Sum( (1/err_i**2) * (y_i**2 - 2*y_i*m_i + m_i**2) + ln(2*pi*err_i**2))
#     # only one pixel
#     # d/dtheta []  = -0.5 * ( (1/err_i**2) * ( 0 - 2*y_i*dm_i/dtheta + 2*m_i*dm_i/dtheta) + 0)
#     #              = -0.5 * (  (1/err_i**2) * (-2*y_i*dm_i/dtheta + 2*m_i*dm_i/dtheta) )
#     scr = np.zeros(j.shape,float)
#     scr[:,0] = - (1/err**2) * (-im + m)*j[:,0]
#     scr[:,1] = - (1/err**2) * (-im + m)*j[:,1]
#     scr[:,2] = - (1/err**2) * (-im + m)*j[:,2]
#     return scr

# @njit
# def information(x,y,err,pars):
#     """
#     This calculates the "information" in pixels for a given star.
#     x/y/err for a set of pixels
#     pars are [amp,xc,yc] for a star
#     """

#     m,j = self.psfjac(x,y,pars)

#     # |dm/dtheta| * (S/N)
#     # where dm/dtheta is given by the Jacobian
#     # and we use the model for the "signal"

#     # since there are 3 parameters, we are going to add up
#     # partial derivatives for all three
#     info = (np.abs(j[:,0])+np.abs(j[:,1])+np.abs(j[:,2])) * (m/err)**2

#     return info
        
# @njit
# def ampfit(trim=True):
#     """ Fit the amps only for the stars."""

#     # linear least squares problem
#     # Ax = b
#     # A is the set of models pixel values for amp, [Npix, Nstar]
#     # x is amps [Nstar] we are solving for
#     # b is pixel values, or residflat values
        
#     # All parameters
#     allpars = self.pars
        
#     A = np.zeros((self.ntotpix,self.nstars),float)
#     usepix = np.zeros(self.ntotpix,np.int8)

#     # Loop over the stars and generate the model image        
#     # ONLY LOOP OVER UNFROZEN STARS
#     dostars = np.arange(self.nstars)[self.freestars]
#     guess = np.zeros(self.nfreestars,float)
#     for count,i in enumerate(dostars):
#         pars = allpars[i*3:(i+1)*3].copy()
#         guess[count] = pars[0]
#         pars[0] = 1.0  # unit amp
#         n1 = self.starfitnpix(i)
#         xind1 = self.starfitx(i)
#         yind1 = self.starfity(i)
#         invind1 = self.starfitinvindex(i)
#         # we need the inverse index to the unique fitted pixels
#         im1 = self.psf(xind1,yind1,pars)
#         A[invind1,i] = im1
#         usepix[invind1] = 1

#     nusepix = np.sum(usepix)

#     # Residual data
#     dy = self.resflat-self.skyflat
        
#     # if trim and nusepix<self.ntotpix:
#     #     unused = np.arange(self.ntotpix)[~usepix]
#     #     A = np.delete(A,unused,axis=0)
#     #     dy = dy[usepix]

#     # from scipy import sparse
#     # A = sparse.csc_matrix(A)   # make sparse

#     # # Use guess to get close to the solution
#     dy2 = dy - A.dot(guess)
#     #par = sparse.linalg.lsqr(A,dy2,atol=1e-4,btol=1e-4)
#     #dbeta = utils.qr_jac_solve(A,dy2,weight=wt1d)
#     par = np.linalg.lstsq(A,dy2)
        
#     damp = par[0]
#     amp = guess+damp
        
#     # preconditioning!
        
#     return amp

# @njit
# def centroid():
#     """ Centroid all of the stars."""

#     # Start with the residual image and all stars subtracted.
#     # Then add in the model of one star at a time and centroid it with the best-fitting model.

#     # All parameters
#     allpars = self.pars

#     #resid = self.image.copy()-self.skyim
#     resid1d = self.image.copy().ravel()-self.skyim.copy().ravel()

#     # Generate full models 
#     # Loop over the stars and generate the model image        
#     # ONLY LOOP OVER UNFROZEN STARS
#     dostars = np.arange(self.nstars)[self.freestars]
#     maxndata = np.max(self.star_ndata)
#     fmodels = np.zeros((len(dostars),maxndata),np.float64)    # full footprint
#     maxfitndata = np.max(self.starfit_ndata)
#     models = np.zeros((len(dostars),maxfitndata),np.float64)  # fit pixels
#     jac = np.zeros((len(dostars),maxfitndata,3),np.float64)  # fit pixels
#     usepix = np.zeros(self.ntotpix,np.int8)        
#     for count,i in enumerate(dostars):
#         # Full star footprint model
#         pars1,fxind1,fyind1,fravelind1 = self.getstar(i)
#         fim1 = self.psf(fxind1,fyind1,pars1)
#         fmodels[i,:len(fim1)] = fim1
#         resid1d[fravelind1] -= fim1        # subtract model
#         # Model for fitting pixels
#         pars1,xind1,yind1,ravelind1,invind1 = self.getstarfit(i)
#         im1,jac1 = self.psfjac(xind1,yind1,pars1)
#         models[i,:len(im1)] = im1
#         jac[i,:len(im1),0] = jac1[:,0]
#         jac[i,:len(im1),1] = jac1[:,1]
#         jac[i,:len(im1),2] = jac1[:,2]

#     # Loop over all free stars and fit centroid
#     xnew = np.zeros(self.nfreestars,np.float64)
#     ynew = np.zeros(self.nfreestars,np.float64)
#     for count,i in enumerate(dostars):
#         pars = self.starpars[i]
#         freezepars = self.freezepars[i*3:(i+1)*3]
#         xnew[count] = pars[1]
#         ynew[count] = pars[2]

#         # Both x/y frozen
#         if freezepars[1]==True and freezepars[2]==True:
#             continue
            
#         # Add the model for this star back in
#         fxind = self.starx(i)
#         fyind = self.stary(i)
#         fn1 = self.star_ndata[i]
#         fmodel1 = fmodels[i,:fn1]
            
#         # crowdsource, does a sum
#         #  y = 2 / integral(P*P*W) * integral(x*(I-P)*W)
#         #  where x = x/y coordinate, I = isolated stamp, P = PSF model, W = weight

#         # Use the derivatives instead
#         pars1,xind1,yind1,ravelind1,invind1 = self.getstarfit(i)
#         n1 = self.starfit_ndata[i]
#         jac1 = jac[i,:,:]
#         jac1 = jac1[:n1,:]   # trim extra pixels
#         jac1 = jac1[:,1:]     # trim amp column
#         resid1 = resid1d[ravelind1]

#         # CHOLESKY_JAC_SOLVE NEEDS THE ACTUAL RESIDUALS!!!!
#         #  with the star removed!!
            
#         # Use cholesky to solve
#         # If only one position frozen, solve for the free one
#         # X frozen, solve for Y
#         if freezepars[1]==True:
#             oldjac1 = jac1
#             jac1 = np.zeros((n1,1),np.float64)
#             jac1[:,0] = oldjac1[:,1]
#             dbeta = utils.qr_jac_solve(jac1,resid1)
#             ynew[count] += dbeta[0]
#         # Y frozen, solve for X
#         elif freezepars[2]==True:
#             oldjac1 = jac1
#             jac1 = np.zeros((n1,1),np.float64)
#             jac1[:,0] = oldjac1[:,0]
#             dbeta = utils.qr_jac_solve(jac1,resid1)
#             xnew[count] += dbeta[0]
#         # Solve for both X and Y
#         else:
#             dbeta = utils.qr_jac_solve(jac1,resid1)
#             xnew[count] += dbeta[0]
#             ynew[count] += dbeta[1]            

#         # Remove the star again
#         #resid[fxind,fyind] -= fmodel1
                
#     return xnew,ynew
        
@njit
def cov(psfdata,freezedata,flatdata,pars):
#        freezepars,freezestars,
#        starflat_ndata,starflat_index,xflat,yflat,ntotpix,
#        imflat,errflat,skyflat,pars):
    """ Determine the covariance matrix."""
    
    # Unpack the PSF data
    psftype,psfparams,psflookup,psforder,imshape = psfdata
    # Unpack freeze information
    freezepars,freezestars = freezedata
    # Unpack flat information
    starflat_ndata,starflat_index,xflat,yflat,ntotpix,imflat,errflat,skyflat = flatdata
    
    # https://stats.stackexchange.com/questions/93316/parameter-uncertainty-after-non-linear-least-squares-estimation
    # more background here, too: http://ceres-solver.org/nnls_covariance.html        
    #xdata = np.arange(ntotpix)
    # Hessian = J.T * T, Hessian Matrix
    #  higher order terms are assumed to be small
    # https://www8.cs.umu.se/kurser/5DA001/HT07/lectures/lsq-handouts.pdf
    flatdata2 = (starflat_ndata,starflat_index,xflat,yflat,ntotpix)
    bestmodel,mjac = jac(psfdata,freezedata,flatdata2,pars,False,True)   # trim, allparams
    # Weights
    #   If weighted least-squares then
    #   J.T * W * J
    #   where W = I/sig_i**2
    wt = np.diag(1/errflat**2)
    hess = mjac.T @ (wt @ mjac)
    #hess = mjac.T @ mjac  # not weighted
    # cov = H-1, covariance matrix is inverse of Hessian matrix
    cov_orig = utils.inverse(hess)
    # Rescale to get an unbiased estimate
    # cov_scaled = cov * (RSS/(m-n)), where m=number of measurements, n=number of parameters
    # RSS = residual sum of squares
    #  using rss gives values consistent with what curve_fit returns
    resid = imflat-bestmodel-skyflat
    #cov = cov_orig * (np.sum(resid**2)/(self.ntotpix-len(self.pars)))
    # Use chi-squared, since we are doing the weighted least-squares and weighted Hessian
    chisq = np.sum(resid**2/errflat**2)        
    cov = cov_orig * (chisq/(ntotpix-len(pars)))  # what MPFITFUN suggests, but very small

    # cov = lqr.jac_covariange(mjac,resid,wt)
        
    return cov

@njit
def dofreeze(freezepars,freezestars,oldpars,newpars,nofreeze=False,minpercdiff=0.5):
    """ Check if we need to freeze pars or stars."""
    
    # Check differences and changes
    diff_all = np.abs(newpars_all-oldpars_all)
    percdiff_all = diff_all.copy()*0
    percdiff_all[0:-1:3] = diff_all[0:-1:3]/np.maximum(oldpar_all[0:-1:3],0.0001)*100  # amp
    percdiff_all[1::3] = diff_all[1::3]*100               # x
    percdiff_all[2::3] = diff_all[2::3]*100               # y
    percdiff_all[-1] = diff_all[-1]/np.maximum(np.abs(oldpar_all[-1]),1)*100
    diff = diff_all[np.where(freezepars==False)]
    percdiff = percdiff_all[np.where(freezepars==False)]
    print('percdiff:',percdiff)
    newfreezestars = np.zeros(1,np.int64)

    return
    
    # # Freeze parameters/stars that converged
    # #  also subtract models of fixed stars
    # #  also return new free parameters
    # if nofreeze==False:
    #     frzpars = percdiff<=minpercdiff
    #     freeparsind, = np.where(freezepars==False)
    #     # Only freeze stars, not individual parameters
    #     oldfreezestars = freezestars.copy()
    #     freezestars = np.sum(freezepars[0:3*nstars].copy().reshape(nstars,3),axis=1)==3
    #     # Subtract model for newly frozen stars
    #     newfreezestars, = np.where((oldfreezestars==False) & (freezestars==True))

    #     print(newfreezestars)
    #     return
        
    #     #bestpar = freeze(bestpar,frzpars)
    #     # If the only free parameter is the sky offset,
    #     #  then freeze it as well
    #     if nfreepars==1 and freepars[-1]==True:
    #         freezepars[-1] = True
    #         bestpar = np.zeros((1),np.float64)
    #     npar = len(bestpar)
    #     if verbose:
    #         print('Nfrozen pars = '+str(nfreezepars))
    #         print('Nfrozen stars = '+str(nfreezestars))
    #         print('Nfree pars = '+str(npar))
    # else:
    #     pars[:len(bestpar)] = bestpar    # also works if skyfit=False
    # maxpercdiff = np.max(percdiff)

    # return freezepars,freezestars,newfreezestars

    
#@njit(cache=True)
@njit
def groupfit(psftype,psfparams,psfnpix,psflookup,psfflux,
             image,error,mask,tab,fitradius,maxiter=10,
             minpercdiff=0.5,reskyiter=2,verbose=False,
             nofreeze=False):
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

    out,model,sky = allfit(psf,image,error,mask,tab)
    
    """
    start = clock()
    
    nstars = len(tab)
    _,_,npsforder = psflookup.shape
    psforder = 0
    if npsforder==4:
        psforder = 1
    if psflookup.ndim != 3:
        raise Exception('psflookup must have 3 dimensions')
    psforder = psflookup.shape[2]
    if psforder>1:
        psforder = 1
    ny,nx = image.shape                             # save image dimensions, python images are (Y,X)
    nx = nx
    ny = ny
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
    nfitpix = int(np.ceil(fitradius))     # +/- nfitpix
    skyradius = psfnpix//2 + 10
    # Initialize the parameter array
    initpars = startab[:,1:4] 
    pars = np.zeros(nstars*3+1,float) # amp, xcen, ycen
    pars[0:-1:3] = initpars[:,0]
    pars[1:-1:3] = initpars[:,1]
    pars[2:-1:3] = initpars[:,2]
    pars[-1] = 0.0  # sky offset
    npars = len(pars)
    
    # Package up the PSF information into a tuple to pass to the functions
    psfdata = (psftype,psfparams,psflookup,psforder,imshape)
    
    # Get information for all the stars
    xcen = pars[1::3]
    ycen = pars[2::3]
    hpsfnpix = psfnpix//2
    out = collatestars(imshape,msk,xcen,ycen,hpsfnpix,fitradius,skyradius)
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
    usepix = np.ones(ntotpix,np.bool_)
    
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

    # Package up the star information
    stardata = (starravelindex,starndata,xx,yy)
    flatdata = (starflat_ndata,starflat_index,xflat,yflat,ntotpix)
    
    # Create initial sky image
    skyim = utils.sky(im).flatten()
    skyflat = skyim[indflat]
    
    # Subtract the initial models from the residual array
    resid = np.zeros(imshape[0]*imshape[1],np.float64)
    resid[:] = im.copy().astype(np.float64).flatten()   # flatten makes it easier to modify
    resid[:] -= skyim   # subtract smooth sky
    modelim = np.zeros(imshape[0]*imshape[1],np.float64)
    for i in range(nstars):
        pars1 = pars[3*i:3*i+3]
        n1 = starndata[i]
        ravelind1 = starravelindex[i]
        xind1 = xx[ravelind1]
        yind1 = yy[ravelind1]
        xdata1 = (xind1,yind1)
        m = psf(xdata1,pars1,psfdata)
        resid[ravelind1] -= m
        modelim[ravelind1] += m

    # Sky and Niter arrays for the stars
    starsky = np.zeros(nstars,np.float64)
    starniter = np.zeros(nstars,np.int64)
    starchisq = np.zeros(nstars,np.float64)
    starrms = np.zeros(nstars,np.float64)
    freezestars = np.zeros(nstars,np.bool_)
    freezepars = np.zeros(len(pars),np.bool_)

    # Perform the fitting
    #--------------------
    
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
    while (niter<maxiter and maxpercdiff>minpercdiff and nfreestars>0):
        start0 = clock()
        print(' ')
        print('--- START OF NEW ITERATION',niter,'---')
        print(' ')
        
        # bestpar: current best values for the free parameters
        # pars: current best values for ALL parameters
        
        # --- Jacobian solvers ---
        # Get the Jacobian and model
        #  only for pixels that are affected by the "free" parameters
        freezedata = (freezepars,freezestars)

        print('bestpar:',bestpar)
        print('len(bestpar):',len(bestpar))
        print('nfreezepars:',np.sum(freezepars==True))
        print('nfreezestars:',np.sum(freezestars==True))
        model0,j = jac(psfdata,freezedata,flatdata,pars,False,True)  # trim, allparams
        print('I got to here!')
        # Residuals
        print('len(skyim):',len(skyim))
        print('len(indflat):',len(indflat))
        skyflat = skyim[indflat]
        print('len(skyflat):',len(skyflat))
        dy = resflat-skyflat-model0
        print('len(dy):',len(dy))
        # Weights
        wt = 1/errflat**2
        # Solve Jacobian
        print('bestpar:',bestpar)
        print(np.sum(~np.isfinite(model0)))
        print(np.sum(~np.isfinite(j)))
        print(np.sum(~np.isfinite(dy)))
        print(model0)
        print('j.shape:',j.shape)
        print('dy.shape:',dy.shape)
        print('wt.shape:',wt.shape)
        print('nfreezepars:',np.sum(freezepars==True))
        dbeta_free = utils.qr_jac_solve(j,dy,weight=wt)
        dbeta_free[~np.isfinite(dbeta_free)] = 0.0  # deal with NaNs, shouldn't happen
        print('dbeta_free:',dbeta_free)
        dbeta = np.zeros(len(pars),np.float64)
        dbeta[np.where(freezepars==False)] = dbeta_free
        print('dbeta:',dbeta)
        
        # --- Perform line search ---
        #  get models
        freezedata = (freezepars,freezestars)
        model1 = model(psfdata,freezedata,flatdata,pars+0.5*dbeta,False,True)   # trim, allparams
        print('linesearch model1')
        model2 = model(psfdata,freezedata,flatdata,pars+dbeta,False,True)       # trim, allparams
        print('linesearch model2')
        print('len(resflat):',len(resflat))
        print('len(model0):',len(model0))
        chisq0 = np.sum((resflat-model0)**2/errflat**2)
        chisq1 = np.sum((resflat-model1)**2/errflat**2)
        chisq2 = np.sum((resflat-model2)**2/errflat**2)
        #if verbose:
        print('linesearch:',chisq0,chisq1,chisq2)
        alpha = utils.quadratic_bisector(np.array([0.0,0.5,1.0]),
                                         np.array([chisq0,chisq1,chisq2]))
        alpha = np.minimum(np.maximum(alpha,0.0),1.0)  # 0<alpha<1
        if np.isfinite(alpha)==False:
            alpha = 1.0
        pars_new = pars + alpha * dbeta
        new_dbeta_free = alpha * dbeta_free
        new_dbeta = np.zeros(len(pars),float)
        print('nfreepars:',np.sum(freezepars==False))
        print('len(new_dbeta_free):',len(new_dbeta_free))
        new_dbeta[np.where(freezepars==False)] = new_dbeta_free
        print('new_dbeta:',new_dbeta)
        
        # Update parameters
        oldpar = bestpar.copy()
        oldpar_all = pars.copy()  #bestpar_all.copy()
        bestpar_all = utils.newpars(pars,new_dbeta,bounds,maxsteps)            
        bestpar = bestpar_all[np.where(freezepars==False)]
        pars[np.where(freezepars==False)] = bestpar
        
        # Check differences and changes
        diff_all = np.abs(bestpar_all-oldpar_all)
        percdiff_all = diff_all.copy()*0
        percdiff_all[0:-1:3] = diff_all[0:-1:3]/np.maximum(oldpar_all[0:-1:3],0.0001)*100  # amp
        percdiff_all[1::3] = diff_all[1::3]*100               # x
        percdiff_all[2::3] = diff_all[2::3]*100               # y
        percdiff_all[-1] = diff_all[-1]/np.maximum(np.abs(oldpar_all[-1]),1)*100
        diff = diff_all[np.where(freezepars==False)]
        percdiff = percdiff_all[np.where(freezepars==False)]

        print('percdiff:',percdiff)
        
        # Freeze parameters/stars that converged
        #  also subtract models of fixed stars
        #  also return new free parameters
        if nofreeze==False:
            frzpars = percdiff<=minpercdiff
            freeparsind, = np.where(freezepars==False)
            # Update freeze values for "free" parameters
            tempfreezepars = freezepars.copy()
            tempfreezepars[np.where(freezepars==False)] = frzpars   # stick in updated values for "free" parameters
            # Only freeze stars, not individual parameters
            oldfreezestars = freezestars.copy()
            freezestars = np.sum(tempfreezepars[0:3*nstars].copy().reshape(nstars,3),axis=1)==3
            # Subtract model for newly frozen stars
            newfreezestars, = np.where((oldfreezestars==False) & (freezestars==True))
            
            print('frzpars:',frzpars)
            print('freeparsind:',freeparsind)
            print('newfreezestars:',newfreezestars)

            # Freezing more stars
            if len(newfreezestars)>0:
                print('Freezing ',len(newfreezestars),' stars')
                for i in range(len(newfreezestars)):
                    istar = newfreezestars[i]
                    print(istar)
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
                    resflat[ravelindex1] -= im1
                    
                # Get the new array of free parameters
                freeparsind, = np.where(freezepars==False)
                print('len(freezepars):',len(freezepars))
                print('len(bestpar):',len(bestpar))
                print('len(freeparsind):',len(freeparsind))
                print('freeparsind:',freeparsind)
                bestpar = pars[freeparsind]
                print('new bestpar:',bestpar)
                nfreezepars = np.sum(freezepars)
                nfreezestars = np.sum(freezestars)

            if verbose:
                print('Nfrozen pars = '+str(nfreezepars))
                print('Nfrozen stars = '+str(nfreezestars))
                print('Nfree pars = '+str(npars))

        nfreestars = np.sum(freezestars==False)
        
        # Get model and chisq
        freezedata = (freezepars,freezestars)
        flatdata = (starflat_ndata,starflat_index,xflat,yflat,ntotpix)
        bestmodel = model(psfdata,freezedata,flatdata,pars,False,True)   # trim, allparams
        resid = imflat-bestmodel-skyflat
        chisq = np.sum(resid**2/errflat**2)

        if verbose:
            print('Iter = ',niter)
            print('Pars = ',pars)
            print('chisq = ',chisq)
            
        # Re-estimate the sky
        if niter % reskyiter == 0:
            print('Re-estimating the sky')
            prevsky = skyim.copy()            
            # Get model of full footprints
            #bestfullmodel = fullmodel(psfdata,stardata,pars)
            #print('re-estimate sky len(bestfullmodel):',len(bestfullmodel))
            # Remove the current best-fit model and re-estimate the sky
            #tresid = im-bestfullmodel.copy().reshape(imshape[0],imshape[1])
            # skyim = utils.sky(tresid).flatten()
            # # Update resid
            # resid[:] += prevsky[indflat]
            # resid[:] -= skyim[indflat]
            # skyflat = skyim[indflat]
            
        if verbose:
            print('iter dt =',(clock()-start0)/1e9,'sec.')
                
        niter += 1     # increment counter

        print('niter=',niter)
        
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
    covdata = (starflat_ndata,starflat_index,xflat,yflat,ntotpix,imflat,errflat,skyflat)
    cov1 = cov(psfdata,freezedata,covdata,pars)
    perror = np.sqrt(np.diag(cov1))
        
    # if verbose:
    print('Best-fitting parameters: ',pars)
    print('Errors: ',perror)
    
    # Put in catalog
    outtab = np.zeros((nstars,15),np.float64)
    outtab[:,0] = np.arange(nstars)+1                # id
    outtab[:,1] = pars[0:-1:3]                       # amp
    outtab[:,2] = perror[0:-1:3]                     # amp_error
    outtab[:,3] = pars[1::3]                         # x
    outtab[:,4] = perror[1::3]                       # x_error
    outtab[:,5] = pars[2::3]                         # y
    outtab[:,6] = perror[2::3]                       # y_error
    outtab[:,7] = starsky + pars[-1]                 # sky
    outtab[:,8] = outtab[:,1]*psfflux                # flux
    outtab[:,9] = outtab[:,2]*psfflux                # flux_error
    outtab[:,10] = -2.5*np.log10(np.maximum(outtab[:,8],1e-10))+25.0   # mag
    outtab[:,11] = (2.5/np.log(10))*outtab[:,9]/outtab[:,8]            # mag_error
    # rms
    # chisq
    outtab[:,14] = starniter                      # niter, what iteration it converged on
    
    # Recalculate chi-squared and RMS of fit
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



        
#@njit(cache=True)
def fit(psf,image,error,tab,fitradius=0.0,recenter=True,maxiter=10,minpercdiff=0.5,
        reskyiter=2,nofreeze=False,skyfit=True,absolute=False,verbose=False):
    """
    Fit PSF to group of stars in an image.

    Parameters
    ----------
    psf : PSF object
       PSF object.
    image : CCDData object
       Image to use to fit PSF model to stars.
    tab : table
       Catalog with initial amp/x/y values for the stars to use to fit the PSF.
    fitradius: float, optional
       The fitting radius in pixels.  By default the PSF FWHM is used.
    recenter : boolean, optional
       Allow the centroids to be fit.  Default is True.
    maxiter : int, optional
       Maximum number of iterations to allow.  Only for methods "cholesky", "qr" or "svd".
       Default is 10.
    minpercdiff : float, optional
       Minimum percent change in the parameters to allow until the solution is
       considered converged and the iteration loop is stopped.  Only for methods
       "cholesky", "qr" and "svd".  Default is 0.5.
    reskyiter : int, optional
       After how many iterations to re-calculate the sky background. Default is 2.
    absolute : boolean, optional
       Input and output coordinates are in "absolute" values using the image bounding box.
         Default is False, everything is relative.
    nofreeze : boolean, optional
       Do not freeze any parameters even if they have converged.  Default is False.
    skyfit : boolean, optional
       Fit a constant sky offset with the stellar parameters.  Default is True.
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

    Example
    -------

    outtab,model,sky = fit(psf,image,tab)

    """

    start = clock()
    nstars = len(tab)

    if fitradius <= 0:
        fitradius = psf.fwhm()*0.5
 
    # Start the Group Fitter
    gf = GroupFitter(psf.psftype,psf.params,image,error,tab,fitradius,
                     psf.lookup,psf.npix,verbose)    
    #gf = GroupFitter(psf,image,tab,fitradius=fitradius,verbose=(verbose>=2))
    xdata = np.arange(gf.ntotpix)
    if skyfit==False:
        gf.freezepars[-1] = True  # freeze sky value
        
    # Centroids fixed
    if recenter==False:
        gf.freezepars[1::3] = True  # freeze X values
        gf.freezepars[2::3] = True  # freeze Y values
        
    # Perform the fitting
    #--------------------
    
    # Initial estimates
    initpar = np.zeros(nstars*3+1,float)
    initpar[0:-1:3] = gf.staramp
    initpar[1::3] = gf.starxcen
    initpar[2::3] = gf.starycen
    initpar[-1] = 0.0
    
    # Make bounds
    #  this requires all 3*Nstars parameters to be input
    bounds = gf.mkbounds(initpar,image.shape,2)    

    # Iterate
    bestpar_all = initpar.copy()        
    # centroids fixed, only keep amps
    if recenter==False:
        initpar = initpar[0::3]  # amps and sky offset
    # sky offset fixed
    if skyfit==False:
        initpar = initpar[0:-1]  # no sky offset
    gf.niter = 0
    maxpercdiff = 1e10
    bestpar = initpar.copy()
    npars = len(bestpar)
    maxsteps = gf.steps(gf.pars,bounds)  # maximum steps, requires all 3*Nstars parameters
    while (gf.niter<maxiter and maxpercdiff>minpercdiff and gf.nfreepars>0):
        start0 = clock()
        
        # Jacobian solvers
        # Get the Jacobian and model
        #  only for pixels that are affected by the "free" parameters
        m,jac = gf.jac(bestpar,True,False)   # trim,allparams
        # Residuals
        dy = gf.resflat[gf.usepix]-gf.skyflat[gf.usepix]-m
        # Weights
        wt = 1/gf.errflat[gf.usepix]**2
        # Solve Jacobian
        dbeta_free = utils.qr_jac_solve(jac,dy,weight=wt)
        dbeta_free[~np.isfinite(dbeta_free)] = 0.0  # deal with NaNs, shouldn't happen
        dbeta = np.zeros(len(gf.pars),np.float64)
        dbeta[gf.freepars] = dbeta_free

        # Perform line search
        alpha,new_dbeta_free = gf.linesearch(bestpar,dbeta_free,m,jac)
        new_dbeta = np.zeros(len(gf.pars),float)
        new_dbeta[gf.freepars] = new_dbeta_free

        # Update parameters
        oldpar = bestpar.copy()
        oldpar_all = bestpar_all.copy()
        bestpar_all = gf.newpars(gf.pars,new_dbeta,bounds,maxsteps)            
        bestpar = bestpar_all[gf.freepars]
        # Check differences and changes
        diff_all = np.abs(bestpar_all-oldpar_all)
        percdiff_all = diff_all.copy()*0
        percdiff_all[0:-1:3] = diff_all[0:-1:3]/np.maximum(oldpar_all[0:-1:3],0.0001)*100  # amp
        percdiff_all[1::3] = diff_all[1::3]*100               # x
        percdiff_all[2::3] = diff_all[2::3]*100               # y
        percdiff_all[-1] = diff_all[-1]/np.maximum(np.abs(oldpar_all[-1]),1)*100
        diff = diff_all[gf.freepars]
        percdiff = percdiff_all[gf.freepars]
        
        # Freeze parameters/stars that converged
        #  also subtract models of fixed stars
        #  also return new free parameters
        if nofreeze==False:
            frzpars = percdiff<=minpercdiff
            freeparsind, = np.where(gf.freezepars==False)
            bestpar = gf.freeze(bestpar,frzpars)
            # If the only free parameter is the sky offset,
            #  then freeze it as well
            if gf.nfreepars==1 and gf.freepars[-1]==True:
                gf.freezepars[-1] = True
                bestpar = np.zeros((1),np.float64)
            npar = len(bestpar)
            if verbose:
                print('Nfrozen pars = '+str(gf.nfreezepars))
                print('Nfrozen stars = '+str(gf.nfreezestars))
                print('Nfree pars = '+str(npar))
        else:
            gf.pars[:len(bestpar)] = bestpar    # also works if skyfit=False
        maxpercdiff = np.max(percdiff)

        # Get model and chisq
        bestmodel = gf.model(gf.pars,False,True)  # trim,allparams
        resid = gf.imflat-bestmodel-gf.skyflat
        chisq = np.sum(resid**2/gf.errflat**2)

        # from gf.chisq() method
        # flux = self.resflat[self.usepix]-self.skyflat[self.usepix]
        # wt = 1/self.errflat[self.usepix]**2
        # bestmodel = self.model(pars) #,allparams=True,trim=False,verbose=False)
        # resid = flux-bestmodel[self.usepix]
        # chisq1 = np.sum(resid**2/self.errflat[self.usepix]**2)

        
        if verbose:
            print('Iter = ',gf.niter)
            print('Pars = ',gf.pars)
            print('Percent diff = ',percdiff)
            print('Diff = ',diff)
            print('chisq = ',chisq)
                
        # Re-estimate the sky
        if gf.niter % reskyiter == 0:
            print('Re-estimating the sky')
            #gf.sky()

        if verbose:
            print('iter dt =',(clock()-start0)/1e9,'sec.')
                
        gf.niter += 1     # increment counter

        print('niter=',gf.niter)

        import pdb; pdb.set_trace()
        
    # Check that all starniter are set properly
    #  if we stopped "prematurely" then not all stars were frozen
    #  and didn't have starniter set
    gf.starniter[np.where(gf.starniter==0)] = gf.niter
    
    # Make final model
    gf.unfreeze()
    model = gf.modelim
        
    # estimate uncertainties
    # Calculate covariance matrix
    cov = gf.cov()
    perror = np.sqrt(np.diag(cov))
        
    pars = gf.pars
    if verbose:
        print('Best-fitting parameters: ',pars)
        print('Errors: ',perror)

    # Put in catalog
    outtab = np.zeros((nstars,15),np.float64)
    outtab[:,0] = np.arange(nstars)+1                # id
    outtab[:,1] = pars[0:-1:3]                       # amp
    outtab[:,2] = perror[0:-1:3]                     # amp_error
    outtab[:,3] = pars[1::3]                         # x
    outtab[:,4] = perror[1::3]                       # x_error
    outtab[:,5] = pars[2::3]                         # y
    outtab[:,6] = perror[2::3]                       # y_error
    outtab[:,7] = gf.starsky + pars[-1]              # sky
    outtab[:,8] = outtab[:,1]*psf.flux()             # flux
    outtab[:,9] = outtab[:,2]*psf.flux()             # flux_error
    outtab[:,10] = -2.5*np.log10(np.maximum(outtab[:,8],1e-10))+25.0   # mag
    outtab[:,11] = (2.5/np.log(10))*outtab[:,9]/outtab[:,8]            # mag_error
    # rms
    # chisq
    outtab[:,14] = gf.starniter                      # niter, what iteration it converged on
    
    # Recalculate chi-squared and RMS of fit
    for i in range(nstars):
        _,xind1,yind1,ravelind1,invind1 = gf.getstarfit(i)
        n1 = gf.starfitnpix(i)
        flux1 = image.ravel()[ravelind1].copy() #[yind1,xind1].copy()
        err1 = error.ravel()[ravelind1].copy()  #[yind1,xind1]
        pars1 = np.zeros(3,np.float64)
        pars1[0] = outtab[i,1]  # amp
        pars1[1] = outtab[i,3]  # x
        pars1[2] = outtab[i,5]  # y
        model1 = gf.psf(xind1,yind1,pars1)
        sky1 = outtab[i,7]
        chisq1 = np.sum((flux1-sky1-model1)**2/err1**2)/n1
        outtab[i,13] = chisq1
        # chi value, RMS of the residuals as a fraction of the amp
        rms1 = np.sqrt(np.mean(((flux1-sky1-model1)/pars1[0])**2))
        outtab[i,12] = rms1

    if verbose:
        print('dt =',(clock()-start)/1e9,'sec.')
 
    return outtab,model,gf.skyim.copy()

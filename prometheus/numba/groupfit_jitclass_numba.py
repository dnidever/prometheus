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

@njit(cache=True)
def getstar(imshape,xcen,ycen,hpsfnpix,fitradius):
    """ Return a star's full footprint and fitted pixels data."""
    # always return the same size
    # a distance of 6.2 pixels spans 6 full pixels but you could have
    # 0.1 left on one side and 0.1 left on the other side
    # that's why we have to add 2 pixels
    nfpix = hpsfnpix*2+1
    fbbox = gnb.starbbox((xcen,ycen),imshape,hpsfnpix)
    nfx = fbbox[1]-fbbox[0]
    nfy = fbbox[3]-fbbox[2]
    # extra buffer is ALWAYS at the end of each dimension    
    fxdata = np.zeros(nfpix*nfpix,np.int64)-1
    fydata = np.zeros(nfpix*nfpix,np.int64)-1
    fravelindex = np.zeros(nfpix*nfpix,np.int64)-1
    fcount = 0
    # Fitting pixels
    npix = int(np.floor(2*fitradius))+2
    bbox = gnb.starbbox((xcen,ycen),imshape,fitradius)
    nx = bbox[1]-bbox[0]
    ny = bbox[3]-bbox[2]
    xdata = np.zeros(npix*npix,np.int64)-1
    ydata = np.zeros(npix*npix,np.int64)-1
    ravelindex = np.zeros(npix*npix,np.int64)-1
    mask = np.zeros(npix*npix,np.int8)
    fcount = 0
    count = 0
    for j in range(nfpix):
        y = j + fbbox[2]
        for i in range(nfpix):
            x = i + fbbox[0]
            r = np.sqrt((x-xcen)**2 + (y-ycen)**2)
            if x>=fbbox[0] and x<=fbbox[1]-1 and y>=fbbox[2] and y<=fbbox[3]-1:
                if r <= 1.0*hpsfnpix:
                    fxdata[fcount] = x
                    fydata[fcount] = y
                    fmulti_index = (np.array([y]),np.array([x]))
                    fravelindex[fcount] = utils.ravel_multi_index(fmulti_index,imshape)[0]
                    fcount += 1
                if r <= fitradius:
                    xdata[count] = x
                    ydata[count] = y
                    multi_index = (np.array([y]),np.array([x]))
                    ravelindex[count] = utils.ravel_multi_index(multi_index,imshape)[0]
                    mask[count] = 1
                    count += 1
    return (fxdata,fydata,fravelindex,fbbox,nfx,nfy,fcount,
            xdata,ydata,ravelindex,bbox,count,mask)

@njit(cache=True)
def collatestars(imshape,starx,stary,hpsfnpix,fitradius):
    """ Get full footprint and fitted pixels data for all stars."""
    nstars = len(starx)
    nfpix = 2*hpsfnpix+1
    npix = int(np.floor(2*fitradius))+2
    # Full footprint arrays
    fxdata = np.zeros((nstars,nfpix*nfpix),np.int64)
    fydata = np.zeros((nstars,nfpix*nfpix),np.int64)
    fravelindex = np.zeros((nstars,nfpix*nfpix),np.int64)
    fbbox = np.zeros((nstars,4),np.int32)
    fshape = np.zeros((nstars,2),np.int32)
    fndata = np.zeros(nstars,np.int32)
    # Fitting pixel arrays
    xdata = np.zeros((nstars,npix*npix),np.int64)
    ydata = np.zeros((nstars,npix*npix),np.int64)
    ravelindex = np.zeros((nstars,npix*npix),np.int64)
    bbox = np.zeros((nstars,4),np.int32)
    shape = np.zeros((nstars,2),np.int32)
    ndata = np.zeros(nstars,np.int32)
    mask = np.zeros((nstars,npix*npix),np.int8)
    for i in range(nstars):
        out = getstar(imshape,starx[i],stary[i],hpsfnpix,fitradius)
        # full footprint information
        fxdata1,fydata1,fravelindex1,fbbox1,fnx1,fny1,fn1 = out[:7]
        fxdata[i,:] = fxdata1
        fydata[i,:] = fydata1
        fravelindex[i,:] = fravelindex1
        fbbox[i,:] = fbbox1
        fshape[i,0] = fny1
        fshape[i,1] = fnx1
        fndata[i] = fn1
        # fitting pixel information
        xdata1,ydata1,ravelindex1,bbox1,n1,mask1 = out[7:]
        xdata[i,:] = xdata1
        ydata[i,:] = ydata1
        ravelindex[i,:] = ravelindex1
        bbox[i,:] = bbox1
        ndata[i] = n1
        mask[i,:] = mask1
    # Trim arrays
    maxfn = np.max(fndata)
    fxdata = fxdata[:,:maxfn]
    fydata = fydata[:,:maxfn]
    fravelindex = fravelindex[:,:maxfn]
    maxn = np.max(ndata)
    xdata = xdata[:,:maxn]
    ydata = ydata[:,:maxn]
    ravelindex = ravelindex[:,:maxn]
    mask = mask[:,:maxn]
    
    return (fxdata,fydata,fravelindex,fbbox,fshape,fndata,
            xdata,ydata,ravelindex,bbox,ndata,mask)


@njit(cache=True)
def getfullstar(imshape,xcen,ycen,hpsfnpix):
    """ Return the entire footprint image/error/x/y arrays for one star."""
    # always return the same size
    # a distance of 6.2 pixels spans 6 full pixels but you could have
    # 0.1 left on one side and 0.1 left on the other side
    # that's why we have to add 2 pixels
    npix = hpsfnpix*2+1
    bbox = gnb.starbbox((xcen,ycen),imshape,hpsfnpix)
    nx = bbox[1]-bbox[0]
    ny = bbox[3]-bbox[2]
    # extra buffer is ALWAYS at the end of each dimension    
    xdata = np.zeros(npix*npix,np.int32)-1
    ydata = np.zeros(npix*npix,np.int32)-1
    ravelindex = np.zeros(npix*npix,np.int64)-1
    count = 0
    for j in range(npix):
        y = j + bbox[2]
        for i in range(npix):
            x = i + bbox[0]
            if x>=bbox[0] and x<=bbox[1]-1 and y>=bbox[2] and y<=bbox[3]-1:
                xdata[count] = x
                ydata[count] = y
                multi_index = (np.array([y]),np.array([x]))
                ravelindex[count] = utils.ravel_multi_index(multi_index,imshape)[0]
                count += 1
    return xdata,ydata,ravelindex,bbox,nx,ny,count

@njit(cache=True)
def collatefullstars(imshape,starx,stary,hpsfnpix):
    """ Get the entire footprint image/error/x/y for all of the stars."""
    nstars = len(starx)
    npix = 2*hpsfnpix+1
    # Get xdata, ydata, error
    xdata = np.zeros((nstars,npix*npix),np.int32)
    ydata = np.zeros((nstars,npix*npix),np.int32)
    ravelindex = np.zeros((nstars,npix*npix),np.int64)
    bbox = np.zeros((nstars,4),np.int32)
    shape = np.zeros((nstars,2),np.int32)
    ndata = np.zeros(nstars,np.int32)
    for i in range(nstars):
        xdata1,ydata1,ravelindex1,bbox1,nx1,ny1,n1 = getfullstar(imshape,starx[i],stary[i],hpsfnpix)
        xdata[i,:] = xdata1
        ydata[i,:] = ydata1
        ravelindex[i,:] = ravelindex1
        bbox[i,:] = bbox1
        shape[i,0] = ny1
        shape[i,1] = nx1
        ndata[i] = n1
    return xdata,ydata,ravelindex,bbox,shape,ndata

# @njit(cache=True)
# def unpackstar(xdata,ydata,bbox,shape,istar):
#     """ Return unpacked data for one star."""
#     xdata1 = xdata[istar,:].copy()
#     ydata1 = ydata[istar,:].copy()
#     bbox1 = bbox[istar,:]
#     shape1 = shape[istar,:]
#     n = len(imdata1)
#     npix = int(np.sqrt(n))
#     # Convert to 2D arrays
#     xdata1 = xdata1.reshape(npix,npix)
#     ydata1 = ydata1.reshape(npix,npix)
#     # Trim values
#     if shape1[0] < npix or shape1[1] < npix:
#         xdata1 = xdata1[:shape1[0],:shape1[1]]
#         ydata1 = ydata1[:shape1[0],:shape1[1]]
#     return xdata1,ydata1,bbox1,shape1

@njit(cache=True)
def getfitstar(imshape,xcen,ycen,fitradius):
    """ Get the fitting pixel information for a single star."""
    npix = int(np.floor(2*fitradius))+2
    bbox = gnb.starbbox((xcen,ycen),imshape,fitradius)
    nx = bbox[1]-bbox[0]
    ny = bbox[3]-bbox[2]
    xdata = np.zeros(npix*npix,np.int32)-1
    ydata = np.zeros(npix*npix,np.int32)-1
    ravelindex = np.zeros(npix*npix,np.int64)-1
    mask = np.zeros(npix*npix,np.int32)
    count = 0
    for j in range(ny):
        y = j + bbox[2]
        for i in range(nx):
            x = i + bbox[0]
            r = np.sqrt((x-xcen)**2 + (y-ycen)**2)
            if r <= fitradius:
                xdata[count] = x
                ydata[count] = y
                multi_index = (np.array([y]),np.array([x]))
                ravelindex[count] = utils.ravel_multi_index(multi_index,imshape)[0]
                mask[count] = 1
                count += 1
    return xdata,ydata,ravelindex,count,mask
        
@njit(cache=True)
def collatefitstars(imshape,starx,stary,fitradius):
    """ Get the fitting pixel information for all stars."""
    nstars = len(starx)
    npix = int(np.floor(2*fitradius))+2
    # Get xdata, ydata, error
    maxpix = nstars*(npix)**2
    xdata = np.zeros((nstars,npix*npix),np.int32)
    ydata = np.zeros((nstars,npix*npix),np.int32)
    ravelindex = np.zeros((nstars,npix*npix),np.int64)
    mask = np.zeros((nstars,npix*npix),np.int32)
    ndata = np.zeros(nstars,np.int32)
    bbox = np.zeros((nstars,4),np.int32)
    for i in range(nstars):
        xcen = starx[i]
        ycen = stary[i]
        bb = gnb.starbbox((xcen,ycen),imshape,fitradius)
        xdata1,ydata1,ravelindex1,n1,mask1 = getfitstar(imshape,xcen,ycen,fitradius)
        xdata[i,:] = xdata1
        ydata[i,:] = ydata1
        ravelindex[i,:] = ravelindex1
        mask[i,:] = mask1
        ndata[i] = n1
        bbox[i,:] = bb
    return xdata,ydata,ravelindex,bbox,ndata,mask

# @njit(cache=True)
# def unpackfitstar(xdata,ydata,bbox,ndata,istar):
#     """ Return unpacked fitting pixel data for one star."""
#     xdata1 = xdata[istar,:]
#     ydata1 = ydata[istar,:]
#     bbox1 = bbox[istar,:]
#     n1 = ndata[istar]
#     # Trim to the values we want
#     xdata1 = xdata1[:n1]
#     ydata1 = ydata1[:n1]
#     return xdata1,ydata1,bbox1,n1

spec = [
    ('x', types.float64[:]),
    ('y', types.int64[:]),
    ('bbox', types.int64[:]),
    ('ravelindex', types.int64[:]),
    ('shape', types.int64[:]),
    ('n', types.int64),
    ('fit_x', types.float64[:]),
    ('fit_y', types.int64[:]),
    ('fit_bbox', types.int64[:]),
    ('fit_ravelindex', types.int64[:]),
    ('fit_shape', types.int64[:]),
    ('fit_n', types.int64),
    ('fit_invindex', types.int64[:]),
]
@jitclass(spec)
class StarData(object):
    # Just a container for star data
    def __init__(self,x,y,bbox,ravelindex,shape,n,fit_x,fit_y,fit_bbox,
                 fit_ravelindex,fit_shape,fit_n,fit_invindex):
        self.x = x
        self.y = y
        self.bbox = bbox
        self.ravelindex = ravelindex
        self.shape = shape
        self.n = n
        self.fit_x = fit_x
        self.fit_y = fit_y
        self.fit_bbox = fit_bbox
        self.fit_ravelindex = fit_ravelindex
        self.fit_shape = fit_fit_shape
        self.fit_n = fit_n
        self.fit_invindex = fit_invindex

    
kv_ty = (types.int64, types.unicode_type)
spec = [
    ('psftype', types.int32),
    ('psfparams', types.float64[:]),
    ('verbose', types.boolean),
    ('psflookup', types.float64[:,:,:]),
    ('psforder', types.int32),
    #('fwhm', types.float64),
    ('image', types.float64[:,:]),
    ('error', types.float64[:,:]),
    ('skyim', types.float64[:,:]),
    ('tab', types.float64[:,:]),
    ('initpars', types.float64[:,:]),
    ('pars', types.float64[:]),
    ('nstars', types.int32),
    ('niter', types.int32),    
    ('npsfpix', types.int32),    
    ('nx', types.int32),
    ('ny', types.int32),
    ('imshape', types.int64[:]),
    ('npix', types.int32),    
    ('fitradius', types.float64),
    ('nfitpix', types.int32),
    ('radius', types.int32),
    ('starsky', types.float64[:]),
    ('starniter', types.int32[:]),
    ('njaciter', types.int32),
    ('freezestars', types.boolean[:]),
    ('freezepars', types.boolean[:]),
    ('pixused', types.int32),
    ('starchisq', types.float64[:]),
    ('starrms', types.float64[:]),
    ('star_xdata', types.int64[:,:]),
    ('star_ydata', types.int64[:,:]),
    ('star_ravelindex', types.int64[:,:]),
    ('star_bbox', types.int32[:,:]),
    ('star_shape', types.int32[:,:]),
    ('star_ndata', types.int32[:]),
    ('starfit_xdata', types.int64[:,:]),
    ('starfit_ydata', types.int64[:,:]),
    ('starfit_ravelindex', types.int64[:,:]),
    ('starfit_mask', types.int8[:,:]),
    ('starfit_bbox', types.int32[:,:]),
    ('starfit_ndata', types.int32[:]),
    ('starfit_invindex', types.int64[:,:]),
    ('starflat_index', types.int64[:,:]),
    ('starflat_ndata', types.int64[:]),
    ('ntotpix', types.int32),
    ('imflat', types.float64[:]),
    ('resflat', types.float64[:]),
    ('errflat', types.float64[:]),
    ('xflat', types.int64[:]),
    ('yflat', types.int64[:]),
    ('indflat', types.int64[:]),
    ('invindex', types.int64[:]),
    ('overlap_nused', types.int64[:]),
    ('overlap_invindex', types.int64[:,:]),
    ('overlap_xdata', types.int64[:,:]),
    ('overlap_ydata', types.int64[:,:]),
    ('bbox', types.int32[:]),
    ('usepix', types.boolean[:]),
  #  ('usepix', types.int8[:]),
]

@jitclass(spec)
class GroupFitter(object):

    def __init__(self,psftype,psfparams,image,error,tab,fitradius,
                 psflookup,npix,verbose):
        #         psflookup=np.zeros((1,1,1),np.float64),npix=51,verbose=False):
        # Save the input values
        self.psftype = psftype
        self.psfparams = psfparams
        self.psflookup = psflookup
        _,_,npsforder = psflookup.shape
        self.psforder = 0
        if npsforder==4:
            self.psforder = 1
        if psflookup.ndim != 3:
            raise Exception('psflookup must have 3 dimensions')
        psforder = psflookup.shape[2]
        if psforder>1:
            self.psforder = 1
        self.verbose = verbose
        self.image = image.astype(np.float64)
        self.error = error.astype(np.float64)
        self.skyim = np.zeros(image.shape,np.float64)
        self.tab = tab                                  # ID, amp, xcen, ycen
        self.initpars = tab[:,1:4]                      # amp, xcen, ycen
        self.nstars = len(tab)                          # number of stars
        self.niter = 0                                  # number of iterations in the solver
        self.npsfpix = npix                             # shape of PSF
        ny,nx = image.shape                             # save image dimensions, python images are (Y,X)
        self.nx = nx
        self.ny = ny
        self.imshape = np.array([ny,nx])
        self.npix = npix
        self.fitradius = fitradius
        self.nfitpix = int(np.ceil(fitradius))  # +/- nfitpix
        # Star amps
        # if 'amp' in cat.colnames:
        #     staramp = cat['amp'].copy()
        # else:
        #     # estimate amp from flux and fwhm
        #     # area under 2D Gaussian is 2*pi*A*sigx*sigy
        #     if 'fwhm' in cat.columns:
        #         amp = cat['flux']/(2*np.pi*(cat['fwhm']/2.35)**2)
        #     else:
        #         amp = cat['flux']/(2*np.pi*(psf.fwhm()/2.35)**2)                
        #     staramp = np.maximum(amp,0)   # make sure it's positive
        # Initialize the parameter array
        pars = np.zeros(self.nstars*3+1,float) # amp, xcen, ycen
        pars[0:-1:3] = self.initpars[:,0]
        pars[1:-1:3] = self.initpars[:,1]
        pars[2:-1:3] = self.initpars[:,2]
        pars[-1] = 0.0  # sky offset
        self.pars = pars
        # Sky and Niter arrays for the stars
        self.starsky = np.zeros(self.nstars,float)
        self.starniter = np.zeros(self.nstars,np.int32)
        self.njaciter = 0  # initialize njaciter
        # Initialize the freezepars and freezestars arrays
        self.freezestars = np.zeros(self.nstars,np.bool_)
        self.freezepars = np.zeros(self.nstars*3+1,np.bool_)
        self.pixused = -1   # initialize pixused

        #(fxdata,fydata,fravelindex,fbbox,fshape,fndata,
        #    xdata,ydata,ravelindex,bbox,ndata,mask)
        
        # Get information for all the stars
        xcen = self.pars[1::3]
        ycen = self.pars[2::3]
        hpsfnpix = self.npsfpix//2
        out = collatestars(self.imshape,xcen,ycen,hpsfnpix,fitradius)
        fxdata,fydata,fravelindex,fsbbox,fsshape,fsndata = out[:6]
        self.star_xdata = fxdata
        self.star_ydata = fydata
        self.star_ravelindex = fravelindex
        self.star_bbox = fsbbox
        self.star_shape = fsshape
        self.star_ndata = fsndata
        # Fitting arrays
        xdata,ydata,ravelindex,sbbox,sndata,smask = out[6:]
        #fxdata,fydata,fravelindex,fbbox,fndata,fmask = collatefitstars(self.imshape,xcen,ycen,fitradius)
        self.starfit_xdata = xdata
        self.starfit_ydata = ydata
        self.starfit_ravelindex = ravelindex
        self.starfit_bbox = sbbox
        self.starfit_ndata = sndata
        self.starfit_mask = smask

        # Combine all of the X and Y values (of the pixels we are fitting) into one array
        ntotpix = np.sum(self.starfit_ndata)
        xall = np.zeros(ntotpix,np.int32)
        yall = np.zeros(ntotpix,np.int32)
        count = 0
        for i in range(self.nstars):
            n1 = self.starfit_ndata[i]
            xdata1 = self.starfit_xdata[i,:n1]
            ydata1 = self.starfit_ydata[i,:n1]
            xall[count:count+n1] = xdata1
            yall[count:count+n1] = ydata1
            count += n1
            
        # Create 1D unraveled indices, python images are (Y,X)
        ind1 = utils.ravel_multi_index((yall,xall),self.imshape)
        # Get unique indices and inverse indices
        #   the inverse index list takes you from the full/duplicated pixels
        #   to the unique ones
        uind1,uindex1,invindex = utils.unique_index(ind1)
        ntotpix = len(uind1)
        ucoords = utils.unravel_index(uind1,image.shape)
        yflat = ucoords[:,0]
        xflat = ucoords[:,1]
        # x/y coordinates of the unique fitted pixels
        
        # Save information on the "flattened" and unique fitted pixel arrays
        imflat = np.zeros(len(uind1),np.float64)
        imflat[:] = image.ravel()[uind1]
        errflat = np.zeros(len(uind1),np.float64)
        errflat[:] = error.ravel()[uind1]
        self.ntotpix = ntotpix
        self.imflat = imflat
        self.resflat = imflat.copy()
        self.errflat = errflat
        self.xflat = xflat
        self.yflat = yflat
        self.indflat = uind1
        self.invindex = invindex

        self.usepix = np.ones(self.ntotpix,np.bool_)
        
        # Add inverse index for the fitted pixels
        #  to be used with imflat/resflat/errflat/xflat/yflat/indflat
        maxfitpix = np.max(self.starfit_ndata)
        self.starfit_invindex = np.zeros((self.nstars,maxfitpix),np.int64)-1
        for i in range(self.nstars):
            n1 = self.starfit_ndata[i]
            if i==0:
                invlo = 0
            else:
                invlo = np.sum(self.starfit_ndata[:i])
            invindex1 = invindex[invlo:invlo+n1]
            self.starfit_invindex[i,:n1] = invindex1

        # For all of the fitting pixels, find the ones that
        # a given star contributes to (basically within its PSF radius)
        starflat_index = np.zeros((self.nstars,self.ntotpix),np.int64)-1
        starflat_ndata = np.zeros(self.nstars,np.int64)
        for i in range(self.ntotpix):
            x1 = self.xflat[i]
            y1 = self.yflat[i]
            for j in range(self.nstars):
                r = np.sqrt((self.xflat[i]-self.starxcen[j])**2 + (self.yflat[i]-self.starycen[j])**2)
                if r <= self.npsfpix:
                    starflat_index[j,starflat_ndata[j]] = i
                    starflat_ndata[j] += 1
        maxndata = np.max(starflat_ndata)
        starflat_index = starflat_index[:,:maxndata]
        self.starflat_index = starflat_index
        self.starflat_ndata = starflat_ndata
                    
            
        # We want to know for each star which pixels (that are being fit)
        # are affected by other stars (within their full footprint, not just
        # their "fitted pixels").
        # How does a star's full PSF footprint overlap neighboring star's fitted pixels?
        pixim = np.zeros(self.imshape[0]*self.imshape[1],np.int32)
        onused = np.zeros(self.nstars,np.int64)
        overlap_invindex = np.zeros((self.nstars,self.npsfpix*self.npsfpix),np.int64)-1
        overlap_xdata = np.zeros((self.nstars,self.npsfpix*self.npsfpix),np.int64)-1
        overlap_ydata = np.zeros((self.nstars,self.npsfpix*self.npsfpix),np.int64)-1
        for i in range(self.nstars):
            pars1,xind1,yind1,ravelindex1 = self.getstar(i)
            # which of these are contained within the final, unique
            # list of fitted pixels (X,Y)?
            ind1 = utils.ravel_multi_index((yind1,xind1),self.imshape)
            pixim[ind1] = 1
            used, = np.where(pixim[self.indflat]==1)
            onused[i] = len(used)
            overlap_invindex[i,:onused[i]] = used
            overlap_xdata[i,:onused[i]] = self.xflat[used]
            overlap_ydata[i,:onused[i]] = self.yflat[used]
            pixim[ind1] = 0   # reset
        maxused = np.max(onused)
        overlap_invindex = overlap_invindex[:,:maxused]
        overlap_xdata = overlap_xdata[:,:maxused]
        overlap_ydata = overlap_ydata[:,:maxused]
        self.overlap_nused = onused
        self.overlap_invindex = overlap_invindex
        self.overlap_xdata = overlap_xdata
        self.overlap_ydata = overlap_ydata

        # Bounding box of all the stars
        xmin = np.min(self.star_bbox[:,0])  # xmin
        xmax = np.max(self.star_bbox[:,1])  # xmax
        ymin = np.min(self.star_bbox[:,2])  # ymin
        ymax = np.max(self.star_bbox[:,3])  # ymax
        bb = np.array([xmin,xmax,ymin,ymax])
        self.bbox = bb

        # Create initial sky image
        self.skyim = utils.sky(self.image)
        #self.sky()
        
        return

        
    @property
    def starpars(self):
        """ Return the [amp,xcen,ycen] parameters in [Nstars,3] array.
            You can GET a star's parameters like this:
            pars = self.starpars[4]
            You can also SET a star's parameters a similar way:
            self.starpars[4] = pars
        """
        return self.pars[0:3*self.nstars].copy().reshape(self.nstars,3)
    
    @property
    def staramp(self):
        """ Return the best-fit amps for all stars."""
        return self.pars[0:-1:3]

    @staramp.setter
    def staramp(self,val):
        """ Set staramp values."""
        self.pars[0:-1:3] = val
    
    @property
    def starxcen(self):
        """ Return the best-fit X centers for all stars."""        
        return self.pars[1::3]

    @starxcen.setter
    def starxcen(self,val):
        """ Set starxcen values."""
        self.pars[1::3] = val
    
    @property
    def starycen(self):
        """ Return the best-fit Y centers for all stars."""        
        return self.pars[2::3]    

    @starycen.setter
    def starycen(self,val):
        """ Set starycen values."""
        self.pars[2::3] = val
    
    def sky(self,method='sep',rin=None,rout=None):
        """ (Re)calculate the sky."""
        # Remove the current best-fit model
        resid = self.image-self.modelim    # remove model
        self.skyim = utils.sky(resid)
        # Calculate sky value for each star
        #  use center position
        for i in range(self.nstars):
            self.starsky[i] = self.skyim[int(np.round(self.starycen[i])),
                                         int(np.round(self.starxcen[i]))]
        
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

    @property
    def skyflat(self):
        """ Return the sky values for the pixels that we are fitting."""
        return self.skyim.ravel()[self.indflat]

    def starx(self,i):
        """ Return star's full circular footprint x array."""
        n = self.star_ndata[i]
        return self.star_xdata[i,:n]

    def stary(self,i):
        """ Return star's full circular footprint y array."""
        n = self.star_ndata[i]
        return self.star_ydata[i,:n]

    def starbbox(self,i):
        """ Return star's full circular footprint bounding box."""
        return self.star_bbox[i,:]

    def starnpix(self,i):
        """ Return number of pixels in a star's full circular footprint."""
        return self.star_ndata[i]

    def starravelindex(self,i):
        """ Return ravel index (into full image) of the star's full footprint """
        n = self.star_ndata[i]
        return self.star_ravelindex[i,:n]

    def starfitx(self,i):
        """ Return star's fitting pixels x values."""
        n = self.starfit_ndata[i]
        return self.starfit_xdata[i,:n]

    def starfity(self,i):
        """ Return star's fitting pixels y values."""
        n = self.starfit_ndata[i]
        return self.starfit_ydata[i,:n]

    def starfitbbox(self,i):
        """ Return star's fitting pixels bounding box."""
        return self.starfit_bbox[i,:]

    def starfitnpix(self,i):
        """ Return number of fitted pixels for a star."""
        return self.starfit_ndata[i]

    def starfitravelindex(self,i):
        """ Return ravel index (into full image) of the star's fitted pixels."""
        n = self.starfit_ndata[i]
        return self.starfit_ravelindex[i,:n]

    def starfitinvindex(self,i):
        """ Return inverse index of the star's fitted pixels."""
        n = self.starfit_ndata[i]
        return self.starfit_invindex[i,:n]

    def starflatnpix(self,i):
        """ Return the number of flat pixels for a star."""
        return self.starflat_ndata[i]
    
    def starflatindex(self,i):
        """ Return (reverse) index for a star's flat pixels."""
        n = self.starflat_ndata[i]
        return self.starflat_index[i,:n]
        
    def starflatx(self,i):
        """ Return star's flat pixel's x values."""
        return self.xflat[self.starflatindex(i)]

    def starflaty(self,i):
        """ Return star's flat pixel's y values."""
        return self.yflat[self.starflatindex(i)]
    
    def starfitchisq(self,i):
        """ Return chisq of current best-fit for one star."""
        pars1,xind1,yind1,ravelind1,invind1 = self.getstarfit(i)
        n1 = self.starfitnpix(i)
        flux1 = self.image.ravel()[ravelind1].copy()
        err1 = self.error.ravel()[ravelind1].copy()
        model1 = self.psf(xind1,yind1,pars1)
        sky1 = self.pars[-1]
        chisq1 = np.sum((flux1-sky1-model1)**2/err1**2)/n1
        return chisq1
    
    def starfitrms(self,i):
        """ Return rms of current best-fit for one star."""
        pars1,xind1,yind1,ravelind1,invind1 = self.getstarfit(i)
        n1 = self.starfitnpix(i)
        flux1 = self.image.ravel()[ravelind1].copy()
        err1 = self.error.ravel()[ravelind1].copy()
        model1 = self.psf(xind1,yind1,pars1)
        sky1 = self.pars[-1]
        # chi value, RMS of the residuals as a fraction of the amp
        rms1 = np.sqrt(np.mean(((flux1-sky1-model1)/pars1[0])**2))
        return rms1
    
    def getstar(self,i):
        """ Get star full footprint information."""
        pars = self.pars[i*3:(i+1)*3]
        n = self.star_ndata[i]
        xind = self.star_xdata[i,:n]
        yind = self.star_ydata[i,:n]
        ravelindex = self.star_ravelindex[i,:n]
        return pars,xind,yind,ravelindex
    
    def getstarfit(self,i):
        """ Get a star fitting pixel information."""
        pars = self.pars[i*3:(i+1)*3]
        n = self.starfit_ndata[i]
        xind = self.starfit_xdata[i,:n]
        yind = self.starfit_ydata[i,:n]
        ravelindex = self.starfit_ravelindex[i,:n]
        invindex = self.starfit_invindex[i,:n]
        return pars,xind,yind,ravelindex,invindex

    @property
    def residfull(self):
        """ Return the residual values of the fitted pixels in full 2D image format."""
        resid = np.zeros(self.imshape[0]*self.imshape[1],np.float64)
        resid[self.indflat] = self.resflat
        resid = resid.reshape((self.imshape[0],self.imshape[1]))
        return resid

    @property
    def imfitfull(self):
        """ Return the flux values of the fitted pixels in full 2D image format."""
        im = np.zeros(self.imshape[0]*self.imshape[1],np.float64)
        im[self.indflat] = self.imflat
        im = im.reshape((self.imshape[0],self.imshape[1]))
        return im

    @property
    def imfull(self):
        """ Return the flux values of the full footprint pixels in full 2D image format."""
        im = np.zeros((self.imshape[0],self.imshape[1]),np.float64)
        for i in range(self.nstars):
            pars1,xind1,yind1,ravelindex1 = self.getstar(i)
            bbox1 = self.star_bbox[i,:]
            im[bbox1[2]:bbox1[3],bbox1[0]:bbox1[1]] = self.image[bbox1[2]:bbox1[3],bbox1[0]:bbox1[1]].copy()
        return im
    
    @property
    def nfreezepars(self):
        """ Return the number of frozen parameters."""
        return np.sum(self.freezepars)

    @property
    def freepars(self):
        """ Return the free parameters."""
        return ~self.freezepars
    
    @property
    def nfreepars(self):
        """ Return the number of free parameters."""
        return np.sum(self.freepars)
    
    @property
    def nfreezestars(self):
        """ Return the number of frozen stars."""
        return np.sum(self.freezestars)

    @property
    def freestars(self):
        """ Return the free stars."""
        return ~self.freezestars
    
    @property
    def nfreestars(self):
        """ Return the number of free stars."""
        return np.sum(self.freestars) 
    
    
    def freeze(self,pars,frzpars):
        """ Freeze stars and parameters"""
        # PARS: best-fit values of free parameters
        # FRZPARS: boolean array of which "free" parameters
        #            should now be frozen

        # Update all the free parameters
        self.pars[self.freepars] = pars

        # Update freeze values for "free" parameters
        self.freezepars[np.where(self.freepars==True)] = frzpars   # stick in the new values for the "free" parameters
        
        # Check if we need to freeze any new parameters
        nfrz = np.sum(frzpars)
        if nfrz==0:
            return pars
        
        # Freeze new stars
        oldfreezestars = self.freezestars.copy()
        self.freezestars = np.sum(self.freezepars[0:3*self.nstars].copy().reshape(self.nstars,3),axis=1)==3
        # Subtract model for newly frozen stars
        newfreezestars, = np.where((oldfreezestars==False) & (self.freezestars==True))
        if len(newfreezestars)>0:
            # add models to a full image
            # WHY FULL IMAGE??
            newmodel = np.zeros(self.imshape[0]*self.imshape[1],np.float64)
            for i in newfreezestars:
                # Save on what iteration this star was frozen
                self.starniter[i] = self.niter+1
                #print('freeze: subtracting model for star ',i)
                pars1,xind1,yind1,ravelind1,invind1 = self.getstarfit(i)
                n1 = len(xind1)
                im1 = self.psf(xind1,yind1,pars1)
                newmodel[ravelind1] += im1
            # Only keep the pixels being fit
            #  and subtract from the residuals
            newmodel1 = newmodel[self.indflat]
            self.resflat -= newmodel1
  
        # Return the new array of free parameters
        frzind = np.arange(len(frzpars))[np.where(frzpars==True)]
        pars = np.delete(pars,frzind)
        return pars

    def unfreeze(self):
        """ Unfreeze all parameters and stars."""
        self.freezestars = np.zeros(self.nstars,np.bool_)
        self.freezepars = np.zeros(self.nstars*3+1,np.bool_)
        self.resflat = self.imflat.copy()


    def mkbounds(self,pars,imshape,xoff=10):
        """ Make bounds for a set of input parameters."""
        # is [amp1,xcen1,ycen1,amp2,xcen2,ycen2, ...]

        npars = len(pars)
        ny,nx = imshape
        
        # Make bounds
        lbounds = np.zeros(npars,float)
        ubounds = np.zeros(npars,float)
        lbounds[0:-1:3] = 0
        lbounds[1::3] = np.maximum(pars[1::3]-xoff,0)
        lbounds[2::3] = np.maximum(pars[2::3]-xoff,0)
        lbounds[-1] = -np.inf
        ubounds[0:-1:3] = np.inf
        ubounds[1::3] = np.minimum(pars[1::3]+xoff,nx-1)
        ubounds[2::3] = np.minimum(pars[2::3]+xoff,ny-1)
        ubounds[-1] = np.inf
        
        bounds = (lbounds,ubounds)
                 
        return bounds

    def checkbounds(self,pars,bounds):
        """ Check the parameters against the bounds."""
        # 0 means it's fine
        # 1 means it's beyond the lower bound
        # 2 means it's beyond the upper bound
        npars = len(pars)
        lbounds,ubounds = bounds
        check = np.zeros(npars,np.int32)
        badlo, = np.where(pars<=lbounds)
        if len(badlo)>0:
            check[badlo] = 1
        badhi, = np.where(pars>=ubounds)
        if len(badhi):
            check[badhi] = 2
        return check
        
    def limbounds(self,pars,bounds):
        """ Limit the parameters to the boundaries."""
        lbounds,ubounds = bounds
        outpars = np.minimum(np.maximum(pars,lbounds),ubounds)
        return outpars

    def limsteps(self,steps,maxsteps):
        """ Limit the parameter steps to maximum step sizes."""
        signs = np.sign(steps)
        outsteps = np.minimum(np.abs(steps),maxsteps)
        outsteps *= signs
        return outsteps

    def steps(self,pars,bounds=None,dx=0.5):
        """ Return step sizes to use when fitting the stellar parameters."""
        npars = len(pars)
        fsteps = np.zeros(npars,float)
        fsteps[0:-1:3] = np.maximum(np.abs(pars[0:-1:3])*0.25,1)
        fsteps[1::3] = dx        
        fsteps[2::3] = dx
        fsteps[-1] = 10
        return fsteps
        
    def newpars(self,pars,steps,bounds,maxsteps):
        """ Get new parameters given initial parameters, steps and constraints."""

        # Limit the steps to maxsteps
        limited_steps = self.limsteps(steps,maxsteps)
        # Make sure that these don't cross the boundaries
        lbounds,ubounds = bounds
        check = self.checkbounds(pars+limited_steps,bounds)
        # Reduce step size for any parameters to go beyond the boundaries
        badpars = (check!=0)
        # reduce the step sizes until they are within bounds
        newsteps = limited_steps.copy()
        count = 0
        maxiter = 2
        while (np.sum(badpars)>0 and count<=maxiter):
            newsteps[badpars] /= 2
            newcheck = self.checkbounds(pars+newsteps,bounds)
            badpars = (newcheck!=0)
            count += 1
            
        # Final parameters
        newpars = pars + newsteps
            
        # Make sure to limit them to the boundaries
        check = self.checkbounds(newpars,bounds)
        badpars = (check!=0)
        if np.sum(badpars)>0:
            # add a tiny offset so it doesn't fit right on the boundary
            newpars = np.minimum(np.maximum(newpars,lbounds+1e-30),ubounds-1e-30)
        return newpars

    def psf(self,x,y,pars):
        """ Return the PSF model for a single star."""
        g,_ = mnb.psf(x,y,pars,self.psftype,self.psfparams,self.psflookup,self.imshape,
                      deriv=False,verbose=False)
        return g

    def psfjac(self,x,y,pars):
        """ Return the PSF model and derivatives/jacobian for a single star."""
        return mnb.psf(x,y,pars,self.psftype,self.psfparams,self.psflookup,self.imshape,
                      deriv=True,verbose=False)
    
    def modelstar(self,i,full=False):
        """ Return model of one star (full footprint) with the current best values."""
        pars,xind,yind,ravelind = self.getstar(i)
        m = self.psf(xind,yind,pars)        
        if full==True:
            modelim = np.zeros((self.imshape[0]*self.imshape[1]),np.float64)
            modelim[ravelind] = m
            modelim = modelim.reshape((self.imshape[0],self.imshape[1]))
        else:
            bbox = self.star_bbox[i,:]
            nx = bbox[1]-bbox[0]+1
            ny = bbox[3]-bbox[2]+1
            xind1 = xind-bbox[0]
            yind1 = yind-bbox[2]
            ind1 = utils.ravel_multi_index((xind1,yind1),(ny,nx))
            modelim = np.zeros(nx*ny,np.float64)
            modelim[ind1] = m
            modelim = modelim.reshape((ny,nx))
        return modelim

    def modelstarfit(self,i,full=False):
        """ Return model of one star (only fitted pixels) with the current best values."""
        pars,xind,yind,ravelind,invind = self.getstarfit(i)
        m = self.psf(xind,yind,pars)        
        if full==True:
            modelim = np.zeros((self.imshape[0]*self.imshape[1]),np.float64)
            modelim[ravelind] = m
            modelim = modelim.reshape((self.imshape[0],self.imshape[1]))
        else:
            bbox = self.starfit_bbox[i,:]
            nx = bbox[1]-bbox[0]+1
            ny = bbox[3]-bbox[2]+1
            xind1 = xind-bbox[0]
            yind1 = yind-bbox[2]
            ind1 = utils.ravel_multi_index((xind1,yind1),(ny,nx))
            modelim = np.zeros(nx*ny,np.float64)
            modelim[ind1] = m
            modelim = modelim.reshape((ny,nx))
        return modelim
    
    @property
    def modelim(self):
        """ This returns the full image of the current best model (no sky)
            using the PARS values."""
        modelim = np.zeros((self.imshape[0],self.imshape[1]),np.float64).ravel()
        for i in range(self.nstars):
            pars1,xind1,yind1,ravelindex1 = self.getstar(i)
            modelim1 = self.psf(xind1,yind1,pars1)
            modelim[ravelindex1] += modelim1
        modelim = modelim.reshape((self.imshape[0],self.imshape[1]))
        return modelim
        
    # @property
    # def modelflatten(self):
    #     """ This returns the current best model (no sky) for only the "flatten" pixels
    #         using the PARS values."""
    #     return self.model(np.arange(10),*self.pars,allparams=True,verbose=False)
        
    def model(self,args,trim=False,allparams=False,verbose=False):
        """ Calculate the model for the stars and pixels we are fitting."""
        # ALLPARAMS:  all of the parameters were input

        if verbose==True or self.verbose:
            print('model: ',self.niter,args)

        # Args are [amp,xcen,ycen] for all Nstars + sky offset
        # so 3*Nstars+1 parameters

        psftype = self.psftype
        psfparams = self.psfparams
        # lookup
        
        # Figure out the parameters of ALL the stars
        #  some stars and parameters are FROZEN
        if self.nfreezepars>0 and allparams==False:
            allpars = self.pars
            allpars[self.freepars] = args
        else:
            allpars = args
        
        x0,x1,y0,y1 = self.bbox
        nx = x1-x0-1
        ny = y1-y0-1
        #im = np.zeros((nx,ny),float)    # image covered by star
        allim = np.zeros(self.ntotpix,np.float64)
        usepix = np.zeros(self.ntotpix,np.bool_)

        # Loop over the stars and generate the model image        
        # ONLY LOOP OVER UNFROZEN STARS
        if allparams==False:
            dostars = np.arange(self.nstars)[self.freestars]
        else:
            dostars = np.arange(self.nstars)
        for i in dostars:
            pars1 = allpars[i*3:(i+1)*3]
            n1 = self.starflat_ndata[i]
            invind1 = self.starflat_index[i,:n1]
            xind1 = self.xflat[invind1]
            yind1 = self.yflat[invind1]
            # we need the inverse index to the unique fitted pixels
            im1 = self.psf(xind1,yind1,pars1)
            allim[invind1] += im1
            usepix[invind1] = True
        
        allim += allpars[-1]  # add sky offset
            
        self.usepix = usepix
        nusepix = np.sum(usepix)
        
        # if trim and nusepix<self.ntotpix:            
        #     unused = np.arange(self.ntotpix)[~usepix]
        #     allim = np.delete(allim,unused)
        
        # self.niter += 1
        
        return allim

    
    def jac(self,args,trim=False,allparams=False):
        #,retmodel=False,trim=False,allparams=False,verbose=None):
        """ Calculate the jacobian for the pixels and parameters we are fitting"""

        # if verbose is None and self.verbose:
        #     print('jac: ',self.njaciter,args)

        # Args are [amp,xcen,ycen] for all Nstars + sky offset
        # so 3*Nstars+1 parameters
        
        psftype = self.psftype
        psfparams = self.psfparams

        # Figure out the parameters of ALL the stars
        #  some stars and parameters are FROZEN
        if self.nfreezepars>0 and allparams==False:
            allpars = self.pars
            if len(args) != (len(self.pars)-self.nfreezepars):
                print('problem')
                return np.zeros(1,np.float64)+np.nan,np.zeros((1,1),np.float64)+np.nan
            allpars[np.where(self.freepars==True)] = args
        else:
            allpars = args
        
        x0,x1,y0,y1 = self.bbox
        nx = x1-x0-1
        ny = y1-y0-1
        im = np.zeros(self.ntotpix,np.float64)
        jac = np.zeros((self.ntotpix,len(self.pars)),np.float64)    # image covered by star
        usepix = np.zeros(self.ntotpix,np.bool_)

        # Loop over the stars and generate the model image        
        # ONLY LOOP OVER UNFROZEN STARS
        if allparams==False:
            dostars = np.arange(self.nstars)[np.where(self.freestars==True)]
        else:
            dostars = np.arange(self.nstars)
        for i in dostars:
            pars1 = allpars[i*3:(i+1)*3]
            n1 = self.starflat_ndata[i]
            invind1 = self.starflat_index[i,:n1]
            xind1 = self.xflat[invind1]
            yind1 = self.yflat[invind1]
            # we need the inverse index to the unique fitted pixels
            im1,jac1 = self.psfjac(xind1,yind1,pars1)
            jac[invind1,i*3] = jac1[:,0]
            jac[invind1,i*3+1] = jac1[:,1]
            jac[invind1,i*3+2] = jac1[:,2]
            im[invind1] += im1
            usepix[invind1] = True
            
        # Sky gradient
        jac[:,-1] = 1
            
        # Remove frozen columns
        if self.nfreezepars>0 and allparams==False:
            #jac = np.delete(jac,np.arange(len(self.pars))[self.freezepars],axis=1)
            origjac = jac
            jac = np.zeros((self.ntotpix,self.nfreepars),np.float64)
            freeparind, = np.where(self.freepars==True)
            for count,i in enumerate(freeparind):
                jac[:,count] = origjac[:,i]

        self.usepix = usepix
        nusepix = np.sum(usepix)

        # Trim out unused pixels
        if trim and nusepix<self.ntotpix:
            unused = np.arange(self.ntotpix)[~usepix]
            origjac = jac
            jac = np.zeros((nusepix,self.nfreepars),np.float64)
            usepixind, = np.where(usepix==True)
            im = im[usepixind]
            for count,i in enumerate(usepixind):
                for j in range(self.nfreepars):
                    jac[count,j] = origjac[i,j]
        
        self.njaciter += 1
        
        return im,jac

    def chisq(self,pars):
        """ Return chi-squared """
        flux = self.resflat[self.usepix]-self.skyflat[self.usepix]
        wt = 1/self.errflat[self.usepix]**2
        bestmodel = self.model(pars,False,True)   # allparams,Trim
        resid = flux-bestmodel[self.usepix]
        chisq1 = np.sum(resid**2/self.errflat[self.usepix]**2)
        return chisq1
    
    def linesearch(self,bestpar,dbeta,m,jac):
        """ Perform line search along search gradient """
        # Residuals
        flux = self.resflat[self.usepix]-self.skyflat[self.usepix]
        # Weights
        wt = 1/self.errflat[self.usepix]**2
        # Inside model() the parameters are limited to the PSF bounds()
        f0 = self.chisq(bestpar)
        f1 = self.chisq(bestpar+0.5*dbeta)
        f2 = self.chisq(bestpar+dbeta)
        print('linesearch:',f0,f1,f2)
        alpha = utils.quadratic_bisector(np.array([0.0,0.5,1.0]),np.array([f0,f1,f2]))
        alpha = np.minimum(np.maximum(alpha,0.0),1.0)  # 0<alpha<1
        if np.isfinite(alpha)==False:
            alpha = 1.0
        pars_new = bestpar + alpha * dbeta
        new_dbeta = alpha * dbeta
        return alpha,new_dbeta

    def score(self,x,y,err,im,pars):
        """
        The score is the partial derivative of the ln likelihood
        """

        m,j = self.psfjac(x,y,pars)

        # ln likelihood is
        # ln likelihood = -0.5 * Sum( (y_i - m_i)**2/err_i**2 + ln(2*pi*err_i**2))
        #                 -0.5 * Sum( (1/err_i**2) * (y_i**2 - 2*y_i*m_i + m_i**2) + ln(2*pi*err_i**2))
        # only one pixel
        # d/dtheta []  = -0.5 * ( (1/err_i**2) * ( 0 - 2*y_i*dm_i/dtheta + 2*m_i*dm_i/dtheta) + 0)
        #              = -0.5 * (  (1/err_i**2) * (-2*y_i*dm_i/dtheta + 2*m_i*dm_i/dtheta) )
        scr = np.zeros(j.shape,float)
        scr[:,0] = - (1/err**2) * (-im + m)*j[:,0]
        scr[:,1] = - (1/err**2) * (-im + m)*j[:,1]
        scr[:,2] = - (1/err**2) * (-im + m)*j[:,2]
        return scr
 
    def information(self,x,y,err,pars):
        """
        This calculates the "information" in pixels for a given star.
        x/y/err for a set of pixels
        pars are [amp,xc,yc] for a star
        """

        m,j = self.psfjac(x,y,pars)

        # |dm/dtheta| * (S/N)
        # where dm/dtheta is given by the Jacobian
        # and we use the model for the "signal"

        # since there are 3 parameters, we are going to add up
        # partial derivatives for all three
        info = (np.abs(j[:,0])+np.abs(j[:,1])+np.abs(j[:,2])) * (m/err)**2

        return info
        
        
    def ampfit(self,trim=True):
        """ Fit the amps only for the stars."""

        # linear least squares problem
        # Ax = b
        # A is the set of models pixel values for amp, [Npix, Nstar]
        # x is amps [Nstar] we are solving for
        # b is pixel values, or residflat values
        
        # All parameters
        allpars = self.pars
        
        A = np.zeros((self.ntotpix,self.nstars),float)
        usepix = np.zeros(self.ntotpix,np.int8)

        # Loop over the stars and generate the model image        
        # ONLY LOOP OVER UNFROZEN STARS
        dostars = np.arange(self.nstars)[self.freestars]
        guess = np.zeros(self.nfreestars,float)
        for count,i in enumerate(dostars):
            pars = allpars[i*3:(i+1)*3].copy()
            guess[count] = pars[0]
            pars[0] = 1.0  # unit amp
            n1 = self.starfitnpix(i)
            xind1 = self.starfitx(i)
            yind1 = self.starfity(i)
            invind1 = self.starfitinvindex(i)
            # we need the inverse index to the unique fitted pixels
            im1 = self.psf(xind1,yind1,pars)
            A[invind1,i] = im1
            usepix[invind1] = 1

        nusepix = np.sum(usepix)

        # Residual data
        dy = self.resflat-self.skyflat
        
        # if trim and nusepix<self.ntotpix:
        #     unused = np.arange(self.ntotpix)[~usepix]
        #     A = np.delete(A,unused,axis=0)
        #     dy = dy[usepix]

        # from scipy import sparse
        # A = sparse.csc_matrix(A)   # make sparse

        # # Use guess to get close to the solution
        dy2 = dy - A.dot(guess)
        #par = sparse.linalg.lsqr(A,dy2,atol=1e-4,btol=1e-4)
        #dbeta = utils.qr_jac_solve(A,dy2,weight=wt1d)
        par = np.linalg.lstsq(A,dy2)
        
        damp = par[0]
        amp = guess+damp
        
        # preconditioning!
        
        return amp

    
    def centroid(self):
        """ Centroid all of the stars."""

        # Start with the residual image and all stars subtracted.
        # Then add in the model of one star at a time and centroid it with the best-fitting model.

        # All parameters
        allpars = self.pars

        #resid = self.image.copy()-self.skyim
        resid1d = self.image.copy().ravel()-self.skyim.copy().ravel()

        # Generate full models 
        # Loop over the stars and generate the model image        
        # ONLY LOOP OVER UNFROZEN STARS
        dostars = np.arange(self.nstars)[self.freestars]
        maxndata = np.max(self.star_ndata)
        fmodels = np.zeros((len(dostars),maxndata),np.float64)    # full footprint
        maxfitndata = np.max(self.starfit_ndata)
        models = np.zeros((len(dostars),maxfitndata),np.float64)  # fit pixels
        jac = np.zeros((len(dostars),maxfitndata,3),np.float64)  # fit pixels
        usepix = np.zeros(self.ntotpix,np.int8)        
        for count,i in enumerate(dostars):
            # Full star footprint model
            pars1,fxind1,fyind1,fravelind1 = self.getstar(i)
            fim1 = self.psf(fxind1,fyind1,pars1)
            fmodels[i,:len(fim1)] = fim1
            resid1d[fravelind1] -= fim1        # subtract model
            # Model for fitting pixels
            pars1,xind1,yind1,ravelind1,invind1 = self.getstarfit(i)
            im1,jac1 = self.psfjac(xind1,yind1,pars1)
            models[i,:len(im1)] = im1
            jac[i,:len(im1),0] = jac1[:,0]
            jac[i,:len(im1),1] = jac1[:,1]
            jac[i,:len(im1),2] = jac1[:,2]

        # Loop over all free stars and fit centroid
        xnew = np.zeros(self.nfreestars,np.float64)
        ynew = np.zeros(self.nfreestars,np.float64)
        for count,i in enumerate(dostars):
            pars = self.starpars[i]
            freezepars = self.freezepars[i*3:(i+1)*3]
            xnew[count] = pars[1]
            ynew[count] = pars[2]

            # Both x/y frozen
            if freezepars[1]==True and freezepars[2]==True:
                continue
            
            # Add the model for this star back in
            fxind = self.starx(i)
            fyind = self.stary(i)
            fn1 = self.star_ndata[i]
            fmodel1 = fmodels[i,:fn1]
            
            # crowdsource, does a sum
            #  y = 2 / integral(P*P*W) * integral(x*(I-P)*W)
            #  where x = x/y coordinate, I = isolated stamp, P = PSF model, W = weight

            # Use the derivatives instead
            pars1,xind1,yind1,ravelind1,invind1 = self.getstarfit(i)
            n1 = self.starfit_ndata[i]
            jac1 = jac[i,:,:]
            jac1 = jac1[:n1,:]   # trim extra pixels
            jac1 = jac1[:,1:]     # trim amp column
            resid1 = resid1d[ravelind1]

            # CHOLESKY_JAC_SOLVE NEEDS THE ACTUAL RESIDUALS!!!!
            #  with the star removed!!
            
            # Use cholesky to solve
            # If only one position frozen, solve for the free one
            # X frozen, solve for Y
            if freezepars[1]==True:
                oldjac1 = jac1
                jac1 = np.zeros((n1,1),np.float64)
                jac1[:,0] = oldjac1[:,1]
                dbeta = utils.qr_jac_solve(jac1,resid1)
                ynew[count] += dbeta[0]
            # Y frozen, solve for X
            elif freezepars[2]==True:
                oldjac1 = jac1
                jac1 = np.zeros((n1,1),np.float64)
                jac1[:,0] = oldjac1[:,0]
                dbeta = utils.qr_jac_solve(jac1,resid1)
                xnew[count] += dbeta[0]
            # Solve for both X and Y
            else:
                dbeta = utils.qr_jac_solve(jac1,resid1)
                xnew[count] += dbeta[0]
                ynew[count] += dbeta[1]            

            # Remove the star again
            #resid[fxind,fyind] -= fmodel1
                
        return xnew,ynew
        
        
    def cov(self):
        """ Determine the covariance matrix."""

        # https://stats.stackexchange.com/questions/93316/parameter-uncertainty-after-non-linear-least-squares-estimation
        # more background here, too: http://ceres-solver.org/nnls_covariance.html        
        xdata = np.arange(self.ntotpix)
        # Hessian = J.T * T, Hessian Matrix
        #  higher order terms are assumed to be small
        # https://www8.cs.umu.se/kurser/5DA001/HT07/lectures/lsq-handouts.pdf
        #mjac = self.jac(xdata,self.pars,allparams=True,trim=False,verbose=False)
        _,mjac = self.jac(self.pars)
        # Weights
        #   If weighted least-squares then
        #   J.T * W * J
        #   where W = I/sig_i**2
        wt = np.diag(1/self.errflat**2)
        hess = mjac.T @ (wt @ mjac)
        #hess = mjac.T @ mjac  # not weighted
        # cov = H-1, covariance matrix is inverse of Hessian matrix
        cov_orig = utils.inverse(hess)
        # Rescale to get an unbiased estimate
        # cov_scaled = cov * (RSS/(m-n)), where m=number of measurements, n=number of parameters
        # RSS = residual sum of squares
        #  using rss gives values consistent with what curve_fit returns
        #bestmodel = self.model(xdata,*self.pars,allparams=True,trim=False,verbose=False)
        #resid = self.imflatten-self.skyflatten-bestmodel
        bestmodel = self.model(self.pars,False,True,False)   # trim,allparams
        resid = self.imflat-bestmodel-self.skyflat
        #cov = cov_orig * (np.sum(resid**2)/(self.ntotpix-len(self.pars)))
        # Use chi-squared, since we are doing the weighted least-squares and weighted Hessian
        chisq = np.sum(resid**2/self.errflat**2)        
        cov = cov_orig * (chisq/(self.ntotpix-len(self.pars)))  # what MPFITFUN suggests, but very small

        # cov = lqr.jac_covariange(mjac,resid,wt)
        
        return cov

        
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

#!/usr/bin/env python

"""GROUPFIT.PY - Fit groups of stars in an image

"""

__authors__ = 'David Nidever <dnidever@montana.edu?'
__version__ = '20210826'  # yyyymmdd


import os
import sys
import numpy as np
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


# Fit a PSF model to multiple stars in an image


# For each star's footprint, save the indices into the whole image

@njit
def getstar(imshape,xcen,ycen,fitradius):
    """ Return the entire footprint image/error/x/y arrays for one star."""
    # always return the same size
    # a distance of 6.2 pixels spans 6 full pixels but you could have
    # 0.1 left on one side and 0.1 left on the other side
    # that's why we have to add 2 pixels
    npix = int(np.floor(2*fitradius))+2
    bbox = gnb.starbbox((xcen,ycen),imshape,fitradius)
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

@njit
def collatestars(imshape,starx,stary,fitradius):
    """ Get the entire footprint image/error/x/y for all of the stars."""
    nstars = len(starx)
    npix = int(np.floor(2*fitradius))+2
    # Get xdata, ydata, error
    xdata = np.zeros((nstars,npix*npix),np.int32)
    ydata = np.zeros((nstars,npix*npix),np.int32)
    ravelindex = np.zeros((nstars,npix*npix),np.int64)
    bbox = np.zeros((nstars,4),np.int32)
    shape = np.zeros((nstars,2),np.int32)
    ndata = np.zeros(nstars,np.int32)
    for i in range(nstars):
        xdata1,ydata1,ravelindex1,bbox1,nx1,ny1,n1 = getstar(imshape,starx[i],stary[i],fitradius)
        xdata[i,:] = xdata1
        ydata[i,:] = ydata1
        ravelindex[i,:] = ravelindex1
        bbox[i,:] = bbox1
        shape[i,0] = ny1
        shape[i,1] = nx1
        ndata[i] = n1
    return xdata,ydata,ravelindex,bbox,shape,ndata

@njit
def unpackstar(xdata,ydata,bbox,shape,istar):
    """ Return unpacked data for one star."""
    xdata1 = xdata[istar,:].copy()
    ydata1 = ydata[istar,:].copy()
    bbox1 = bbox[istar,:]
    shape1 = shape[istar,:]
    n = len(imdata1)
    npix = int(np.sqrt(n))
    # Convert to 2D arrays
    xdata1 = xdata1.reshape(npix,npix)
    ydata1 = ydata1.reshape(npix,npix)
    # Trim values
    if shape1[0] < npix or shape1[1] < npix:
        xdata1 = xdata1[:shape1[0],:shape1[1]]
        ydata1 = ydata1[:shape1[0],:shape1[1]]
    return xdata1,ydata1,bbox1,shape1

@njit
def getfitstar(imshape,xcen,ycen,fitradius):
    """ Get the fitting pixels for a single star."""
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
        
@njit
def collatefitstars(imshape,starx,stary,fitradius):
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

@njit
def unpackfitstar(xdata,ydata,bbox,ndata,istar):
    """ Return unpacked fitting data for one star."""
    xdata1 = xdata[istar,:]
    ydata1 = ydata[istar,:]
    bbox1 = bbox[istar,:]
    n1 = ndata[istar]
    # Trim to the values we want
    xdata1 = xdata1[:n1]
    ydata1 = ydata1[:n1]
    return xdata1,ydata1,bbox1,n1

    
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
    ('starnpix', types.int32[:]),
    ('star_xdata', types.int32[:,:]),
    ('star_ydata', types.int32[:,:]),
    ('star_ravelindex', types.int64[:,:]),
    ('star_bbox', types.int32[:,:]),
    ('star_shape', types.int32[:,:]),
    ('star_ndata', types.int32[:]),
    ('starfit_xdata', types.int32[:,:]),
    ('starfit_ydata', types.int32[:,:]),
    ('starfit_ravelindex', types.int64[:,:]),
    ('starfit_mask', types.int32[:,:]),
    ('starfit_bbox', types.int32[:,:]),
    ('starfit_ndata', types.int32[:]),
    ('starfit_invindex', types.int32[:,:]),
    ('ntotpix', types.int32),
    ('imflat', types.float64[:]),
    ('resflat', types.float64[:]),
    ('errflat', types.float64[:]),
    ('xflat', types.int64[:]),
    ('yflat', types.int64[:]),
    ('indflat', types.int64[:]),
    ('bbox', types.int32[:]),
    #('_unitfootflux', types.float64),
    #('_bounds', types.float64[:,:]),
    #('_steps', types.float64[:]),
    #('coords', types.float64[:]),
    #('imshape', types.int32[:]),
    #('order', types.int32),
]

@jitclass(spec)
class GroupFitter(object):

    def __init__(self,psftype,psfparams,image,error,tab,fitradius,
                 psflookup=np.zeros((1,1,1),np.float64),npix=51,verbose=False):
        # Save the input values
        self.psftype = psftype
        self.psfparams = psfparams
        self.psflookup = psflookup
        self.psforder = 0
        if psflookup.ndim != 3:
            raise Exception('psflookup must have 3 dimensions')
        psforder = psflookup.shape[2]
        if psforder>1:
            self.psforder = 1
        self.verbose = verbose
        self.image = image.astype(np.float64)
        self.error = error.astype(np.float64)
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

        # Get information for all the stars
        xcen = self.pars[1::3]
        ycen = self.pars[2::3]
        xdata,ydata,ravelindex,sbbox,sshape,sndata = collatestars(self.imshape,xcen,ycen,fitradius)
        self.star_xdata = xdata
        self.star_ydata = ydata
        self.star_ravelindex = ravelindex
        self.star_bbox = sbbox
        self.star_shape = sshape
        self.star_ndata = sndata

        # Get fitting information for all the stars
        fxdata,fydata,fravelindex,fbbox,fndata,fmask = collatefitstars(self.imshape,xcen,ycen,fitradius)
        self.starfit_xdata = fxdata
        self.starfit_ydata = fydata
        self.starfit_ravelindex = fravelindex
        self.starfit_mask = fmask
        self.starfit_bbox = fbbox
        self.starfit_ndata = fndata

        # Get the unique pixels we are fitting

        # Combine all of the X and Y values (of the pixels we are fitting) into one array
        ntotpix = np.sum(self.starfit_ndata)
        xall = np.zeros(ntotpix,np.int32)
        yall = np.zeros(ntotpix,np.int32)
        count = 0
        for i in range(self.nstars):
            n1 = self.starfit_ndata[i]
            xdata1 = self.starfit_xdata[i,:]
            ydata1 = self.starfit_ydata[i,:]
            xall[count:count+n1] = xdata1[:n1]
            yall[count:count+n1] = ydata1[:n1]
            count += n1
        
        # Create 1D unraveled indices, python images are (Y,X)
        ind1 = utils.ravel_multi_index((yall,xall),self.imshape)
        # Get unique indexes and inverse indices
        #   the inverse index list takes you from the duplicated pixels
        #   to the unique ones
        #uind1,invindex = np.unique(ind1,return_inverse=True)
        uind1,uindex1,invindex = utils.unique_index(ind1)
        ntotpix = len(uind1)
        ucoords = utils.unravel_index(uind1,image.shape)
        yflat = ucoords[:,0]
        xflat = ucoords[:,1]
        # x/y coordinates of the unique fitted pixels
        
        # Save information on the "flattened" arrays
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

        # Add inverse index
        #  to be used with imflat/resflat/errflat/xflat/yflat/indflat
        self.starfit_invindex = np.zeros((self.nstars,npix*npix),np.int32)
        for i in range(self.nstars):
            n1 = self.starfit_ndata[i]
            if i==0:
                invlo = 0
            else:
                invlo = np.sum(self.starfit_ndata[:i])
            invindex1 = invindex[invlo:invlo+n1]
            self.starfit_invindex[i,:n1] = invindex1
        
        # Bounding box of all the stars
        xmin = np.min(self.star_bbox[:,0])  # xmin
        xmax = np.max(self.star_bbox[:,1])  # xmax
        ymin = np.min(self.star_bbox[:,2])  # ymin
        ymax = np.max(self.star_bbox[:,3])  # ymax
        bb = np.array([xmin,xmax,ymin,ymax])
        self.bbox = bb
        
        return
        


        # # We want to know for each star which pixels (that are being fit)
        # # are affected by it (within it's full pixel list, not just
        # # "its fitted pixels").
        # invindexlist = []
        # xlist = []
        # ylist = []
        # pixim = np.zeros(image.shape,bool)        
        # for i in range(self.nstars):
        #     fx,fy = fxlist[i],fylist[i]
        #     # which of these are contained within the final, unique
        #     # list of fitted pixels (X,Y)?
        #     pixim[fy,fx] = True
        #     used, = np.where(pixim[y,x]==True)
        #     invindexlist.append(used)
        #     xlist.append(x[used])
        #     ylist.append(y[used])
        #     pixim[fy,fx] = False  # reset
        # self.invindexlist = invindexlist
        # self.xlist = xlist
        # self.ylist = ylist
            
        # #self.invindex = invindex  # takes you from duplicates to unique pixels
        # #invindexlist = []
        # #count = 0
        # #for i in range(self.nstars):
        # #    n = len(self.xlist[i])
        # #    invindexlist.append(invindex[count:count+n])
        # #    count += n
        # #self.invindexlist = invindexlist
        
        # self.bbox = BoundingBox(np.min(x),np.max(x)+1,np.min(y),np.max(y)+1)
        # #self.bboxdata = bboxdata

        # # Create initial sky image
        # self.sky()

        
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
    
    # def sky(self,method='sep',rin=None,rout=None):
    #     """ (Re)calculate the sky."""
    #     # Remove the current best-fit model
    #     resid = self.image.data-self.modelim  # remove model
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

    # @property
    # def skyflatten(self):
    #     """ Return the sky values for the pixels that we are fitting."""
    #     return self.skyim.ravel()[self.ind1]

    def getstar(self,i):
        """ Get star full footprint information."""
        pars = self.pars[i*3:(i+1)*3]
        n = self.starfit_ndata[i]
        xind = self.starfit_xdata[i,:n]
        yind = self.starfit_ydata[i,:n]
        ravelindex = self.starfit_ravelindex[i,:n]
        invindex = self.starfit_invindex[i,:n]
        return pars,xind,yind,ravelindex,invindex
    
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
        self.freezepars[self.freepars] = frzpars   # stick in the new values for the "free" parameters
        
        # Check if we need to freeze any new parameters
        nfrz = np.sum(frzpars)
        if nfrz==0:
            return pars
        
        # Freeze new stars
        oldfreezestars = self.freezestars.copy()
        self.freezestars = np.sum(self.freezepars[0:3*self.nstars].reshape(self.nstars,3),axis=1)==3
        # Subtract model for newly frozen stars
        newfreezestars, = np.where((oldfreezestars==False) & (self.freezestars==True))
        if len(newfreezestars)>0:
            # add models to a full image
            newmodel = np.zeros((self.imshape[0],self.imshape[1]),np.float64).ravel()
            for i in newfreezestars:
                # Save on what iteration this star was frozen
                self.starniter[i] = self.niter+1
                #print('freeze: subtracting model for star ',i)
                pars1 = self.pars[i*3:(i+1)*3]
                n1 = self.starfit_ndata[i]
                xind = self.starfit_xdata[i,:n1]
                yind = self.starfit_ydata[i,:n1]
                ravelindex = self.starfit_ravelindex[i,:n1]
                im1 = self.psf(xind,yind,pars1)
                newmodel[ravelindex] += im1
            # Only keep the pixels being fit
            #  and subtract from the residuals
            newmodel1 = newmodel[self.indflat]
            self.resflatten -= newmodel1
                
        # Return the new array of free parameters
        frzind = np.arange(len(frzpars))[frzpars]
        pars = np.delete(pars,frzind)
        return pars
                         
    def unfreeze(self):
        """ Unfreeze all parameters and stars."""
        self.freezestars = np.zeros(self.nstars,bool)
        self.freezepars = np.zeros(self.nstars*3+1,bool)
        self.resflatten = self.imflatten.copy()


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
        ubounds[1::3] = np.maximum(pars[1::3]+xoff,nx-1)
        ubounds[2::3] = np.maximum(pars[2::3]+xoff,ny-1)
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

    def steps(self,pars,bounds=None,dx=0.2):
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
        
    @property
    def modelim(self):
        """ This returns the full image of the current best model (no sky)
            using the PARS values."""
        modelim = np.zeros((self.imshape[0],self.imshape[1]),np.float64).ravel()
        for i in range(self.nstars):
            pars = self.pars[i*3:(i+1)*3]
            n1 = self.star_ndata[i]
            xdata1 = self.star_xdata[i,:n1]
            ydata1 = self.star_ydata[i,:n1]
            ravelindex1 = self.star_ravelindex[i,:n1]
            modelim1 = self.psf(xdata1,ydata1,pars)
            modelim[ravelindex1] += modelim1
        modelim = modelim.reshape((self.imshape[0],self.imshape[1]))
        return modelim
        
    # @property
    # def modelflatten(self):
    #     """ This returns the current best model (no sky) for only the "flatten" pixels
    #         using the PARS values."""
    #     return self.model(np.arange(10),*self.pars,allparams=True,verbose=False)
        
    def model(self,args,trim=False,allparams=False,verbose=False):
        """ Calculate the model for the pixels we are fitting."""
        # ALLPARAMS:  all of the parameters were input

        if verbose==True or self.verbose:
            print('model: ',self.niter,args)

        # Args are [amp,xcen,ycen] for all Nstars + sky offset
        # so 3*Nstars+1 parameters

        psftype = self.psftype
        psfparams = self.psfparams
        
        # Figure out the parameters of ALL the stars
        #  some stars and parameters are FROZEN
        if self.nfreezepars>0 and allparams is False:
            allpars = self.pars
            allpars[self.freepars] = args
        else:
            allpars = args
        
        x0,x1,y0,y1 = self.bbox
        nx = x1-x0-1
        ny = y1-y0-1
        #im = np.zeros((nx,ny),float)    # image covered by star
        allim = np.zeros(self.ntotpix,float)
        usepix = np.zeros(self.ntotpix,bool)

        # Loop over the stars and generate the model image        
        # ONLY LOOP OVER UNFROZEN STARS
        if allparams is False:
            dostars = np.arange(self.nstars)[self.freestars]
        else:
            dostars = np.arange(self.nstars)
        for i in dostars:
            pars = allpars[i*3:(i+1)*3]
            n1 = self.star_ndata[i]
            xind1 = self.starfit_xdata[i,:n1]
            yind1 = self.starfit_ydata[i,:n1]
            if i==0:
                invlo = 0
            else:
                invlo = np.sum(self.starfit_ndata[:i])
            invindex = self.indflat[invlo:invlo+n1]
            # ravelindex is 1d ravel index for the whole image
            #ravelindex1 = self.starfit_ravelindex[i,:n1]
            # we need the inverse index to the unique fitted pixels
            im1 = self.psf(xind,yind,pars)
            allim[invindex] += im1
            usepix[invindex] = True

        # allim += allpars[-1]  # add sky offset
            
        # self.usepix = usepix
        # nusepix = np.sum(usepix)
        
        # if trim and nusepix<self.ntotpix:            
        #     unused = np.arange(self.ntotpix)[~usepix]
        #     allim = np.delete(allim,unused)
        
        # # self.niter += 1
        
        # return allim

    
    def jac(self,x,*args,retmodel=False,trim=False,allparams=False,verbose=None):
        """ Calculate the jacobian for the pixels and parameters we are fitting"""

        if verbose is None and self.verbose:
            print('jac: ',self.njaciter,args)

        # Args are [amp,xcen,ycen] for all Nstars + sky offset
        # so 3*Nstars+1 parameters
        
        psftype = self.psftype
        psfparams = self.psfparams

        # Figure out the parameters of ALL the stars
        #  some stars and parameters are FROZEN
        if self.nfreezepars>0 and allparams is False:
            allpars = self.pars
            if len(args) != (len(self.pars)-self.nfreezepars):
                print('problem')
                import pdb; pdb.set_trace()
            allpars[self.freepars] = args
        else:
            allpars = args
        
    #     x0,x1 = self.bbox.xrange
    #     y0,y1 = self.bbox.yrange
    #     nx = x1-x0-1
    #     ny = y1-y0-1
    #     #jac = np.zeros((nx,ny,len(args)),float)    # image covered by star
    #     jac = np.zeros((self.ntotpix,len(self.pars)),float)    # image covered by star
    #     usepix = np.zeros(self.ntotpix,bool)
    #     if retmodel:
    #         im = np.zeros(self.ntotpix,float)

        # # Loop over the stars and generate the model image        
        # # ONLY LOOP OVER UNFROZEN STARS
        # if allparams is False:
        #     dostars = np.arange(self.nstars)[self.freestars]
        # else:
        #     dostars = np.arange(self.nstars)
        # for i in dostars:
        #     pars = allpars[i*3:(i+1)*3]
        #     #bbox = self.bboxdata[i]
        #     xind = self.xlist[i]
        #     yind = self.ylist[i]
        #     invindex = self.invindexlist[i]
        #     xdata = (xind,yind)
        #     if retmodel:
        #         m,jac1 = psf.jac(xdata,*pars,retmodel=True)
        #     else:
        #         jac1 = psf.jac(xdata,*pars)
        #     jac[invindex,i*3] = jac1[:,0]
        #     jac[invindex,i*3+1] = jac1[:,1]
        #     jac[invindex,i*3+2] = jac1[:,2]
        #     #jac[invindex,i*4+3] = jac1[:,3]
        #     if retmodel:
        #         im[invindex] += m
        #     usepix[invindex] = True

        # # Sky gradient
        # jac[:,-1] = 1
            
        # # Remove frozen columns
        # if self.nfreezepars>0 and allparams is False:
    #     #     jac = np.delete(jac,np.arange(len(self.pars))[self.freezepars],axis=1)


    #     self.usepix = usepix
    #     nusepix = np.sum(usepix)

    #     # Trim out unused pixels
    #     if trim and nusepix<self.ntotpix:
    #         unused = np.arange(self.ntotpix)[~usepix]
    #         jac = np.delete(jac,unused,axis=0)
    #         if retmodel:
    #             im = np.delete(im,unused)
        
    #     self.njaciter += 1
        
    #     if retmodel:
    #         return im,jac
    #     else:
    #         return jac

    # def linesearch(self,xdata,bestpar,dbeta,m,jac):
    #     # Perform line search along search gradient
    #     # Residuals
    #     flux = self.resflatten[self.usepix]-self.skyflatten[self.usepix]
    #     # Weights
    #     wt = 1/self.errflatten[self.usepix]**2

    #     start_point = bestpar
    #     search_gradient = dbeta
    #     def obj_func(pp,m=None):
    #         """ chisq given the parameters."""
    #         if m is None:
    #             m = self.model(xdata,*pp,trim=True)
    #         chisq = np.sum((flux.ravel()-m.ravel())**2 * wt.ravel())
    #         return chisq
    #     def obj_grad(pp,m=None,jac=None):
    #         """ Gradient of chisq wrt the parameters."""
    #         if m is None and jac is None:
    #             m,jac = self.jac(xdata,*pp,retmodel=True)
    #         # d chisq / d parj = np.sum( 2*jac_ij*(m_i-d_i))/sig_i**2)
    #         dchisq = np.sum( 2*jac * (m.ravel()-flux.ravel()).reshape(-1,1)
    #                          * wt.ravel().reshape(-1,1),axis=0)
    #         return dchisq

    #     f0 = obj_func(start_point,m=m)
    #     # Do our own line search with three points and a quadratic fit.
    #     f1 = obj_func(start_point+0.5*search_gradient)
    #     f2 = obj_func(start_point+search_gradient)
    #     alpha = dln.quadratic_bisector(np.array([0.0,0.5,1.0]),np.array([f0,f1,f2]))
    #     alpha = np.minimum(np.maximum(alpha,0.0),1.0)  # 0<alpha<1
    #     if ~np.isfinite(alpha):
    #         alpha = 1.0
    #     # Use scipy.optimize.line_search()
    #     #grad0 = obj_grad(start_point,m=m,jac=jac)        
    #     #alpha,fc,gc,new_fval,old_fval,new_slope = line_search(obj_func, obj_grad, start_point, search_gradient, grad0,f0,maxiter=3)
    #     #if alpha is None:  # did not converge
    #     #    alpha = 1.0
    #     pars_new = start_point + alpha * search_gradient
    #     new_dbeta = alpha * search_gradient
    #     return alpha,new_dbeta
    
        
    def ampfit(self,trim=True):
        """ Fit the amps only for the stars."""

        # linear least squares problem
        # Ax = b
        # A is the set of models pixel values for amp, [Npix, Nstar]
        # x is amps [Nstar] we are solving for
        # b is pixel values, oru residflatten values
        
        # All parameters
        allpars = self.pars
        
        A = np.zeros((self.ntotpix,self.nstars),float)
        # usepix = np.zeros(self.ntotpix,bool)

        ## Loop over the stars and generate the model image        
        ## ONLY LOOP OVER UNFROZEN STARS
        # dostars = np.arange(self.nstars)[self.freestars]
        # guess = np.zeros(self.nfreestars,float)
        # for count,i in enumerate(dostars):
        #     pars = allpars[i*3:(i+1)*3].copy()
        #     guess[count] = pars[0]
        #     pars[0] = 1.0  # unit amp
        #     xind = self.xlist[i]
        #     yind = self.ylist[i]
        #     invindex = self.invindexlist[i]
        #     im1 = self.psf(xind,yind,pars)
        #     A[invindex,i] = im1
        #     usepix[invindex] = True

        # nusepix = np.sum(usepix)

        # # Residual data
        # dy = self.resflatten-self.skyflatten
        
        # if trim and nusepix<self.ntotpix:
        #     unused = np.arange(self.ntotpix)[~usepix]
        #     A = np.delete(A,unused,axis=0)
        #     dy = dy[usepix]

        # from scipy import sparse
        # A = sparse.csc_matrix(A)   # make sparse

        # # # Use guess to get close to the solution
        # # dy2 = dy - A.dot(guess)
        # par = sparse.linalg.lsqr(A,dy2,atol=1e-4,btol=1e-4)
        # damp = par[0]
        # amp = guess+damp
        
        # # preconditioning!
        
        # return amp

    
    def centroid(self):
        """ Centroid all of the stars."""

        # Start with the residual image and all stars subtracted.
        # Then add in the model of one star at a time and centroid it with the best-fitting model.

        # All parameters
        allpars = self.pars

        resid = self.image.copy()-self.skyim
        
    #     # Generate full models 
    #     # Loop over the stars and generate the model image        
    #     # ONLY LOOP OVER UNFROZEN STARS
    #     fmodels = []
    #     #fjac = []
    #     models = []
    #     jac = []
    #     dostars = np.arange(self.nstars)[self.freestars]
    #     usepix = np.zeros(self.ntotpix,bool)        
    #     for count,i in enumerate(dostars):
    #         pars = self.starpars[i]
    #         #pars = allpars[i*3:(i+1)*3]
    #         # Full models
    #         fxind = self.fxlist[i]
    #         fyind = self.fylist[i]
    #         fim1 = self.psf(fxind,fyind,pars)
    #         resid[fyind,fxind] -= fim1
        #     fmodels.append(fim1)            
        #     #fjac.append(fjac1)
        #     #fusepix[finvindex] = True
        #     # Only fitting models
        #     xind = self.xlist[i]
        #     yind = self.ylist[i]
        #     invindex = self.invindexlist[i]
        #     im1,jac1 = self.psf.jac((xind,yind),*pars,retmodel=True)
        #     models.append(im1)
        #     jac.append(jac1)
        #     #usepix[invindex] = True

        # # Loop over all free stars and fit centroid
        # xnew = np.zeros(self.nfreestars,float)
        # ynew = np.zeros(self.nfreestars,float)
        # for count,i in enumerate(dostars):
        #     pars = self.starpars[i]
        #     freezepars = self.freezepars[i*3:(i+1)*3]
        #     xnew[count] = pars[1]
        #     ynew[count] = pars[2]

        #     # Both x/y frozen
        #     if freezepars[1]==True and freezepars[2]==True:
        #         continue
            
        #     # Add the model for this star back in
        #     fxind = self.fxlist[i]
        #     fyind = self.fylist[i]
        #     fmodel1 = fmodels[count]
        #     #resid[fyind,fxind] += fmodel1
            
        #     # crowdsource, does a sum
        #     #  y = 2 / integral(P*P*W) * integral(x*(I-P)*W)
        #     #  where x = x/y coordinate, I = isolated stamp, P = PSF model, W = weight

        #     # Use the derivatives instead
        #     xind = self.xlist[i]
        #     yind = self.ylist[i]
        #     jac1 = jac[count]
        #     jac1 = np.delete(jac1,0,axis=1)  # delete amp column
        #     resid1 = resid[yind,xind] 

        #     # CHOLESKY_JAC_SOLVE NEEDS THE ACTUAL RESIDUALS!!!!
        #     #  with the star removed!!
            
        #     # Use cholesky to solve
        #     # If only one position frozen, solve for the free one
        #     # X frozen, solve for Y
        #     if freezepars[1]==True:
        #         jac1 = np.delete(jac1,0,axis=1)
        #         dbeta = cholesky_jac_solve(jac1,resid1)
        #         ynew[count] += dbeta[0]
        #     # Y frozen, solve for X
        #     elif freezepars[2]==True:
        #         jac1 = np.delete(jac1,1,axis=1)
        #         dbeta = cholesky_jac_solve(jac1,resid1)
        #         xnew[count] += dbeta[0]
        #     # Solve for both X and Y
        #     else:
        #         dbeta = cholesky_jac_solve(jac1,resid1)
        #         xnew[count] += dbeta[0]
        #         ynew[count] += dbeta[1]            

        #     # Remove the star again
        #     #resid[fxind,fyind] -= fmodel1
                
        # return xnew,ynew
        
        
    def cov(self):
        """ Determine the covariance matrix."""

        # https://stats.stackexchange.com/questions/93316/parameter-uncertainty-after-non-linear-least-squares-estimation
        # more background here, too: http://ceres-solver.org/nnls_covariance.html        
        xdata = np.arange(self.ntotpix)
        # Hessian = J.T * T, Hessian Matrix
        #  higher order terms are assumed to be small
        # https://www8.cs.umu.se/kurser/5DA001/HT07/lectures/lsq-handouts.pdf
    #     mjac = self.jac(xdata,*self.pars,allparams=True,trim=False,verbose=False)
    #     # Weights
    #     #   If weighted least-squares then
    #     #   J.T * W * J
    #     #   where W = I/sig_i**2
    #     wt = np.diag(1/self.errflatten**2)
    #     hess = mjac.T @ (wt @ mjac)
    #     #hess = mjac.T @ mjac  # not weighted
    #     # cov = H-1, covariance matrix is inverse of Hessian matrix
    #     cov_orig = lsq.inverse(hess)
    #     # Rescale to get an unbiased estimate
    #     # cov_scaled = cov * (RSS/(m-n)), where m=number of measurements, n=number of parameters
    #     # RSS = residual sum of squares
    #     #  using rss gives values consistent with what curve_fit returns
    #     bestmodel = self.model(xdata,*self.pars,allparams=True,trim=False,verbose=False)
    #     resid = self.imflatten-self.skyflatten-bestmodel
    #     #cov = cov_orig * (np.sum(resid**2)/(self.ntotpix-len(self.pars)))
    #     # Use chi-squared, since we are doing the weighted least-squares and weighted Hessian
    #     chisq = np.sum(resid**2/self.errflatten**2)        
    #     cov = cov_orig * (chisq/(self.ntotpix-len(self.pars)))  # what MPFITFUN suggests, but very small

    #     # cov = lqr.jac_covariange(mjac,resid,wt)
        
    #     return cov

        
    
def fit(psf,image,cat,method='qr',fitradius=None,recenter=True,maxiter=10,minpercdiff=0.5,
        reskyiter=2,nofreeze=False,skyfit=True,absolute=False,verbose=False):
    """
    Fit PSF to group of stars in an image.

    Parameters
    ----------
    psf : PSF object
       PSF object with initial parameters to use.
    image : CCDData object
       Image to use to fit PSF model to stars.
    cat : table
       Catalog with initial amp/x/y values for the stars to use to fit the PSF.
    method : str, optional
       Method to use for solving the non-linear least squares problem: "cholesky",
       "qr", "svd", and "curve_fit".  Default is "cholesky".
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

    outcat,model,sky = fit(psf,image,cat)

    """

    start = time.time()
    print = utils.getprintfunc() # Get print function to be used locally, allows for easy logging   
    
    # Check input catalog
    for n in ['x','y']:
        if n not in cat.keys():
            raise ValueError('Cat must have x and y columns')

    # Check the method
    method = str(method).lower()    
    if method not in ['cholesky','svd','qr','sparse','htcen','curve_fit']:
        raise ValueError('Only cholesky, svd, qr, sparse, htcen or curve_fit methods currently supported')

    # Make sure image is CCDData
    if isinstance(image,CCDData) is False:
        image = CCDData(image)
    
    nstars = np.array(cat).size
    
    # Image offsets
    if absolute:
        imx0 = image.bbox.xrange[0]
        imy0 = image.bbox.yrange[0]
        cat['x'] -= imx0
        cat['y'] -= imy0        
        
    # Start the Group Fitter
    gf = GroupFitter(psf,image,cat,fitradius=fitradius,verbose=(verbose>=2))
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
    initpar[1::3] = cat['x']
    initpar[2::3] = cat['y']
    initpar[-1] = 0.0
    
    # Make bounds
    #  this requires all 3*Nstars parameters to be input
    bounds = gf.mkbounds(initpar,image.shape,xoff=2)    
        
    # Curve_fit
    #   dealt with separately
    if method=='curve_fit':
        # Perform the fitting
        bounds = [np.zeros(gf.nstars*3+1,float)-np.inf,
                  np.zeros(gf.nstars*3+1,float)+np.inf]
        bounds[0][0:-1:3] = 0
        if recenter:
            bounds[0][1::3] = cat['x']-2
            bounds[1][1::3] = cat['x']+2
            bounds[0][2::3] = cat['y']-2
            bounds[1][2::3] = cat['y']+2
        else:
            bounds[0][1::3] = cat['x']-1e-7
            bounds[1][1::3] = cat['x']+1e-7
            bounds[0][2::3] = cat['y']-1e-7
            bounds[1][2::3] = cat['y']+1e-7         
        bestpar,cov = curve_fit(gf.model,xdata,gf.imflatten-gf.skyflatten,bounds=bounds,
                                sigma=gf.errflatten,p0=initpar,jac=gf.jac)
        bestmodel = gf.model(xdata,*bestpar)
        perror = np.sqrt(np.diag(cov))
        chisq = np.sum((gf.imflatten-gf.skyflatten-bestmodel)**2/gf.errflatten**2)
        gf.pars = bestpar
        gf.chisq = chisq

        
    # All other fitting methods
    else:

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
            start0 = time.time()

            # Jacobian solvers
            if method != 'htcen':
                # Get the Jacobian and model
                #  only for pixels that are affected by the "free" parameters
                m,jac = gf.jac(xdata,*bestpar,retmodel=True,trim=True)
                # Residuals
                dy = gf.resflatten[gf.usepix]-gf.skyflatten[gf.usepix]-m
                # Weights
                wt = 1/gf.errflatten[gf.usepix]**2
                # Solve Jacobian
                dbeta_free = lsq.jac_solve(jac,dy,method=method,weight=wt)
                dbeta_free[~np.isfinite(dbeta_free)] = 0.0  # deal with NaNs, shouldn't happen
                dbeta = np.zeros(len(gf.pars),float)
                #import pdb; pdb.set_trace()
                dbeta[gf.freepars] = dbeta_free
                                
            #  htcen, crowdsource method of solving amps/fluxes first
            #      and then centroiding to get x/y
            else:
                dbeta = np.zeros(3*gf.nfreestars+1,float)
                # Solve for the amps
                newht = gf.ampfit()
                dbeta[0:-1:3] = gf.staramp[gf.freestars] - newht
                gf.staramp[gf.freestars] = newht  # update the amps
                # Solve for the positions
                newx, newy = gf.centroid()
                dbeta[1::3] = gf.starxcen[~gf.freezestars] - newx
                dbeta[2::3] = gf.starycen[~gf.freezestars] - newy
                gf.starxcen[gf.freestars] = newx
                gf.starycen[gf.freestars] = newy
                # trim frozen parameters from free stars
                freestars = np.arange(gf.nstars)[gf.freestars]
                freepars = np.zeros(3*gf.nstars,bool)
                for count,i in enumerate(freestars):
                    freepars1 = gf.freepars[i*3:(i+1)*3]
                    freepars[count*3:(count+1)*3] = freepars1
                dbeta_free = dbeta[freepars]

            chisq = np.sum(dy**2 * wt.ravel())/len(dy)
            
            # Perform line search
            alpha,new_dbeta_free = gf.linesearch(xdata,bestpar,dbeta_free,m,jac)
            new_dbeta = np.zeros(len(gf.pars),float)
            new_dbeta[gf.freepars] = new_dbeta_free
            
            # Update parameters
            oldpar = bestpar.copy()
            oldpar_all = bestpar_all.copy()
            #bestpar_all = gf.newpars(gf.pars,dbeta,bounds,maxsteps)
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
            if not nofreeze:
                frzpars = percdiff<=minpercdiff
                freeparsind, = np.where(~gf.freezepars)
                bestpar = gf.freeze(bestpar,frzpars)
                # If the only free parameter is the sky offset,
                #  then freeze it as well
                if gf.nfreepars==1 and gf.freepars[-1]==True:
                    gf.freezepars[-1] = True
                    bestpar = []
                npar = len(bestpar)
                if verbose:
                    print('Nfrozen pars = '+str(gf.nfreezepars))
                    print('Nfrozen stars = '+str(gf.nfreezestars))
                    print('Nfree pars = '+str(npar))
            else:
                gf.pars = bestpar            
            maxpercdiff = np.max(percdiff)

            # Get model and chisq
            if method != 'curve_fit':
                bestmodel = gf.model(xdata,*gf.pars,allparams=True)
                resid = gf.imflatten-gf.skyflatten-bestmodel
                chisq = np.sum(resid**2/gf.errflatten**2)
                gf.chisq = chisq
              
            if verbose:
                print('Iter = '+str(gf.niter))
                print('Pars = '+str(gf.pars))
                print('Percent diff = '+str(percdiff))
                print('Diff = '+str(diff))
                print('chisq = '+str(chisq))
                
            # Re-estimate the sky
            if gf.niter % reskyiter == 0:
                print('Re-estimating the sky')
                gf.sky()

            if verbose:
                print('iter dt =  %.2f sec' % (time.time()-start0))
                
            gf.niter += 1     # increment counter
        
    # Check that all starniter are set properly
    #  if we stopped "prematurely" then not all stars were frozen
    #  and didn't have starniter set
    gf.starniter[gf.starniter==0] = gf.niter
    
    # Make final model
    gf.unfreeze()
    model = CCDData(gf.modelim,bbox=image.bbox,unit=image.unit)
        
    # Estimate uncertainties
    if method != 'curve_fit':
        # Calculate covariance matrix
        cov = gf.cov()
        perror = np.sqrt(np.diag(cov))
        
    pars = gf.pars
    if verbose:
        print('Best-fitting parameters: '+str(pars))
        print('Errors: '+str(perror))

        
    # Put in catalog
    # Initialize catalog
    dt = np.dtype([('id',int),('amp',float),('amp_error',float),('x',float),
                   ('x_error',float),('y',float),('y_error',float),('sky',float),
                   ('flux',float),('flux_error',float),('mag',float),('mag_error',float),
                   ('rms',float),('chisq',float),('niter',int)])
    outcat = np.zeros(nstars,dtype=dt)
    if 'id' in cat.keys():
        outcat['id'] = cat['id']
    else:
        outcat['id'] = np.arange(nstars)+1
    outcat['amp'] = pars[0:-1:3]
    outcat['amp_error'] = perror[0:-1:3]
    outcat['x'] = pars[1::3]
    outcat['x_error'] = perror[1::3]
    outcat['y'] = pars[2::3]
    outcat['y_error'] = perror[2::3]
    outcat['sky'] = gf.starsky + pars[-1]
    outcat['flux'] = outcat['amp']*psf.flux()
    outcat['flux_error'] = outcat['amp_error']*psf.flux()    
    outcat['mag'] = -2.5*np.log10(np.maximum(outcat['flux'],1e-10))+25.0
    outcat['mag_error'] = (2.5/np.log(10))*outcat['flux_error']/outcat['flux']
    outcat['niter'] = gf.starniter  # what iteration it converged on
    outcat = Table(outcat)

    # Relculate chi-squared and RMS of fit
    for i in range(nstars):
        xlist = gf.xlist0[i]
        ylist = gf.ylist0[i]
        flux = image.data[ylist,xlist].copy()
        err = image.error[ylist,xlist]
        xdata = (xlist,ylist)
        model1 = psf(xlist,ylist,pars=[outcat['amp'][i],outcat['x'][i],outcat['y'][i]])
        chisq = np.sum((flux-outcat['sky'][i]-model1.ravel())**2/err**2)/len(xlist)
        outcat['chisq'][i] = chisq
        # chi value, RMS of the residuals as a fraction of the amp
        rms = np.sqrt(np.mean(((flux-outcat['sky'][i]-model1.ravel())/outcat['amp'][i])**2))
        outcat['rms'][i] = rms
        
    # Image offsets for absolute X/Y coordinates
    if absolute:
        outcat['x'] += imx0
        outcat['y'] += imy0
        cat['x'] += imx0
        cat['y'] += imy0        

    if verbose:
        print('dt = %.2f sec' % (time.time()-start))        
        
    return outcat,model,gf.skyim
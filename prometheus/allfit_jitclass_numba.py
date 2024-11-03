#!/usr/bin/env python

"""ALLFIT_NUMBA.PY - Fit PSF to all stars in an image

"""

__authors__ = 'David Nidever <dnidever@montana.edu?'
__version__ = '20241013'  # yyyymmdd


import os
import numpy as np
import time
from numba import njit,types,from_dtype
from numba.experimental import jitclass
from . import utils_numba as utils, groupfit_numba as gfit, models_numba as mnb
from .clock_numba import clock

@njit(cache=True)
def skyval(array,sigma):
    """  Estimate sky value from sky pixels."""
    wt = 1/sigma**2
    med = np.median(array)
    sig = utils.mad(array)
    # reweight outlier pixels using Stetson's method
    resid = array-med
    wt2 = wt/(1+np.abs(resid)**2/np.median(sigma))
    xmn = np.sum(wt2*array)/np.sum(wt2)
    return xmn

    
kv_ty = (types.int64, types.unicode_type)
spec = [
    ('psftype', types.int32),
    ('psfparams', types.float64[:]),
    ('verbose', types.boolean),
    ('nofreeze', types.boolean),
    ('psflookup', types.float64[:,:,:]),
    ('psforder', types.int32),
    #('fwhm', types.float64),
    ('image', types.float64[:,:]),
    ('error', types.float64[:,:]),
    ('mask', types.boolean[:,:]),
    ('skyim', types.float64[:]),
    ('modelim', types.float64[:]),
    ('resid', types.float64[:]),
    ('xx', types.int64[:]),
    ('yy', types.int64[:]),
    ('tab', types.float64[:,:]),
    ('initpars', types.float64[:,:]),
    ('pars', types.float64[:]),
    ('perror', types.float64[:]),
    ('nstars', types.int32),
    ('niter', types.int32),    
    ('npsfpix', types.int32),    
    ('nx', types.int32),
    ('ny', types.int32),
    ('imshape', types.int64[:]),
    ('npix', types.int32),    
    ('fitradius', types.float64),
    ('skyradius', types.float64),
    ('nfitpix', types.int32),
    ('radius', types.int32),
    ('_starsky', types.float64[:]),
    ('starniter', types.int32[:]),
    ('njaciter', types.int32),
    ('freezestars', types.boolean[:]),
    ('starchisq', types.float64[:]),
    ('starrms', types.float64[:]),
    ('star_ravelindex', types.int64[:,:]),
    ('star_ndata', types.int32[:]),
    ('starfit_ravelindex', types.int64[:,:]),
    ('starfit_ndata', types.int32[:]),
    ('starsky_ravelindex', types.int64[:,:]),
    ('starsky_ndata', types.int32[:]),
    ('ntotpix', types.int32),
    ('ravelindex', types.int64[:]),
]

@jitclass(spec)
#@jitclass(spec, cache=True)
class AllFitter(object):

    def __init__(self,psftype,psfparams,npix,psflookup,image,error,mask,
                 tab,fitradius,verbose=False,nofreeze=False):
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
        self.nofreeze = nofreeze
        ny,nx = image.shape                             # save image dimensions, python images are (Y,X)
        self.nx = nx
        self.ny = ny
        self.imshape = np.array([ny,nx])
        self.image = image.copy().astype(np.float64)
        self.error = error.astype(np.float64)
        self.mask = mask.astype(np.bool_)
        xx,yy = utils.meshgrid(np.arange(nx),np.arange(ny))
        self.xx = xx.flatten()
        self.yy = yy.flatten()
        # Order stars by flux, brightest first
        si = np.argsort(tab[:,1])[::-1]  # largest amp first
        self.tab = tab[si]                              # ID, amp, xcen, ycen
        self.initpars = self.tab[:,1:4]                 # amp, xcen, ycen
        self.nstars = len(self.tab)                     # number of stars
        self.niter = 1                                  # number of iterations in the solver
        self.npsfpix = npix                             # shape of PSF
        self.npix = npix
        self.fitradius = fitradius
        self.nfitpix = int(np.ceil(fitradius))  # +/- nfitpix
        self.skyradius = npix//2 + 10
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
        pars = np.zeros(self.nstars*3,float) # amp, xcen, ycen
        pars[0::3] = self.initpars[:,0]
        pars[1::3] = self.initpars[:,1]
        pars[2::3] = self.initpars[:,2]
        self.pars = pars
        self.perror = np.zeros(len(pars),np.float64)
        # Sky and Niter arrays for the stars
        self._starsky = np.zeros(self.nstars,float)
        self.starniter = np.zeros(self.nstars,np.int32)
        self.starchisq = np.zeros(self.nstars,np.float64)
        self.starrms = np.zeros(self.nstars,np.float64)
        self.njaciter = 0  # initialize njaciter
        # Initialize the freezestars array
        self.freezestars = np.zeros(self.nstars,np.bool_)
        
        # Get information for all the stars
        xcen = self.pars[1::3]
        ycen = self.pars[2::3]
        hpsfnpix = self.npsfpix//2
        out = self.collatestars(self.imshape,xcen,ycen,hpsfnpix,fitradius,self.skyradius)
        fravelindex,fsndata,ravelindex,sndata,skyravelindex,skyndata = out
        self.star_ravelindex = fravelindex
        self.star_ndata = fsndata
        # Fitting arrays
        self.starfit_ravelindex = ravelindex
        self.starfit_ndata = sndata
        # Sky arrays
        self.starsky_ravelindex = skyravelindex
        self.starsky_ndata = skyndata
        
        # Put indices of all the unique fitted pixels into one array
        ntotpix = np.sum(self.starfit_ndata)
        allravelindex = np.zeros(ntotpix,np.int64)
        count = 0
        for i in range(self.nstars):
            n1 = self.starfit_ndata[i]
            ravelind1 = self.starfit_ravelindex[i,:n1]
            allravelindex[count:count+n1] = ravelind1
            count += n1
        allravelindex = np.unique(allravelindex)
        self.ravelindex = allravelindex
        self.ntotpix = len(allravelindex)
        
        # Create initial smooth sky image
        self.skyim = utils.sky(self.image).flatten()
        # Subtract the initial models from the residual array
        self.resid = image.copy().astype(np.float64).flatten()   # flatten makes it easier to modify
        self.resid[:] -= self.skyim   # subtract smooth sky
        self.modelim = np.zeros(self.imshape[0]*self.imshape[1],np.float64)
        for i in range(self.nstars):
            pars1 = self.starpars[i]
            n1 = self.star_ndata[i]
            ravelind1 = self.starravelindex(i)
            xind1 = self.xx[ravelind1]
            yind1 = self.yy[ravelind1]
            m = self.psf(xind1,yind1,pars1)
            self.resid[ravelind1] -= m
            self.modelim[ravelind1] += m
            

    def getstar(self,imshape,xcen,ycen,hpsfnpix,fitradius,skyradius):
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
                if self.mask[y,x]:   # exclude bad pixels
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

    def collatestars(self,imshape,starx,stary,hpsfnpix,fitradius,skyradius):
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
            out = self.getstar(imshape,starx[i],stary[i],hpsfnpix,fitradius,skyradius)
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
        return self.pars[0::3]

    @staramp.setter
    def staramp(self,val):
        """ Set staramp values."""
        self.pars[0::3] = val
    
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
    
    def sky(self):
        """ (Re)calculate the smooth sky."""
        # Remove the current best-fit model
        resid = self.image-self.modelim.copy().reshape(self.imshape[0],self.imshape[1])    # remove model
        prevsky = self.skyim.copy()
        self.skyim[:] = utils.sky(resid).flatten()
        # Update resid
        self.resid[:] += prevsky
        self.resid[:] -= self.skyim
        
    def starsky(self,i):
        """ Calculate the local sky for a single star."""
        n = self.starsky_ndata[i]
        skyravelindex = self.starsky_ravelindex[i,:n]
        resid = self.resid[skyravelindex]
        err = self.error.ravel()[skyravelindex]
        sky = skyval(resid,err)
        self._starsky[i] = sky
        return sky

    def starnpix(self,i):
        """ Return number of pixels in a star's full circular footprint."""
        return self.star_ndata[i]

    def starravelindex(self,i):
        """ Return ravel index (into full image) of the star's full footprint """
        n = self.star_ndata[i]
        return self.star_ravelindex[i,:n]

    def starfitnpix(self,i):
        """ Return number of fitted pixels for a star."""
        return self.starfit_ndata[i]

    def starfitravelindex(self,i):
        """ Return ravel index (into full image) of the star's fitted pixels."""
        n = self.starfit_ndata[i]
        return self.starfit_ravelindex[i,:n]

    def starfitchisq(self,i):
        """ Return chisq of current best-fit for one star."""
        pars1,xind1,yind1,ravelind1 = self.starfitdata(i)
        n1 = self.starfitnpix(i)
        flux1 = self.image.ravel()[ravelind1].copy()
        err1 = self.error.ravel()[ravelind1].copy()
        model1 = self.psf(xind1,yind1,pars1)
        sky1 = self._starsky[i]
        chisq1 = np.sum((flux1-sky1-model1)**2/err1**2)/n1
        return chisq1
    
    def starfitrms(self,i):
        """ Return rms of current best-fit for one star."""
        pars1,xind1,yind1,ravelind1 = self.starfitdata(i)
        n1 = self.starfitnpix(i)
        flux1 = self.image.ravel()[ravelind1].copy()
        err1 = self.error.ravel()[ravelind1].copy()
        model1 = self.psf(xind1,yind1,pars1)
        sky1 = self._starsky[i]
        # chi value, RMS of the residuals as a fraction of the amp
        rms1 = np.sqrt(np.mean(((flux1-sky1-model1)/pars1[0])**2))
        return rms1
    
    def stardata(self,i):
        """ Return a star's full footprint information."""
        pars = self.pars[i*3:(i+1)*3]
        n = self.star_ndata[i]
        ravelindex = self.star_ravelindex[i,:n]        
        xind = self.xx[ravelindex]
        yind = self.yy[ravelindex]
        return pars,xind,yind,ravelindex
    
    def starfitdata(self,i):
        """ Return a star's fitting pixel information."""
        pars = self.pars[i*3:(i+1)*3]
        n = self.starfit_ndata[i]
        ravelindex = self.starfit_ravelindex[i,:n]
        xind = self.xx[ravelindex]
        yind = self.yy[ravelindex]
        return pars,xind,yind,ravelindex

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

    def unfreeze(self):
        """ Unfreeze all parameters and stars."""
        self.freezestars = np.zeros(self.nstars,np.bool_)
        ##self.resflat = self.imflat.copy()


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

    def steps(self,pars,dx=0.5):
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

    def starmodelfull(self,i):
        """ Calculate the model for a star for the full footprint."""
        # This is for a SINGLE star
        # Get all of the star data
        pars,xind,yind,ravelindex = self.stardata(i)
        n = len(xind)
        m = self.psf(xind,yind,pars)        
        return m

    def starmodel(self,i):
        """ Calculate the model for a star for the fitted pixels."""
        # This is for a SINGLE star
        # Get all of the star data
        pars,xind,yind,ravelindex = self.starfitdata(i)
        m = self.psf(xind,yind,pars)        
        return m
    
    def starjac(self,i):
        """ Calculate the jacobian for a star for the fitted pixels."""
        # This is for a SINGLE star
        # Get all of the star data
        pars,xind,yind,ravelindex = self.starfitdata(i)
        m,j = self.psfjac(xind,yind,pars)        
        return m,j

    def starfitim(self,i):
        """ Return the image pixels for a star."""
        n = self.starfit_ndata[i]
        ravelind = self.starfit_ravelindex[i,:n]
        return self.image.ravel()[ravelind]
    
    def starfiterr(self,i):
        """ Return the error pixels for a star."""
        n = self.starfit_ndata[i]
        ravelind = self.starfit_ravelindex[i,:n]
        return self.error.ravel()[ravelind]
    
    def starfitresid(self,i):
        """ Return the resid pixels for a star."""
        n = self.starfit_ndata[i]
        ravelind = self.starfit_ravelindex[i,:n]
        return self.resid[ravelind]
    
    # def starchisq(self,i,pars):
    #     """ Return chi-squared """
    #     fravelindex = self.starfitravelindex(i)
    #     resid = self.resid[fravelindex]
    #     # Subtract the sky?        
    #     err = self.error.ravel()[fravelindex]
    #     #flux = self.resid-self.skyflat
    #     #wt = 1/self.errflat**2
    #     wt = 1/wt**2
    #     bestmodel = self.model(pars,False,True)   # allparams,Trim
    #     resid = flux-bestmodel[self.usepix]
    #     chisq1 = np.sum(resid**2/err**2)
    #     return chisq1

        
    def starcov(self,i):
        """ Determine the covariance matrix for a single star."""

        # https://stats.stackexchange.com/questions/93316/parameter-uncertainty-after-non-linear-least-squares-estimation
        # more background here, too: http://ceres-solver.org/nnls_covariance.html

        # Get fitting pixel information
        pars,fxind,fyind,fravelindex = self.starfitdata(i)
        resid = self.resid[fravelindex]
        err = self.error.ravel()[fravelindex]
        wt = 1/err**2    # weights
        n = len(fxind)
        # Hessian = J.T * T, Hessian Matrix
        #  higher order terms are assumed to be small
        # https://www8.cs.umu.se/kurser/5DA001/HT07/lectures/lsq-handouts.pdf
        m,j = self.psfjac(fxind,fyind,pars)
        # Weights
        #   If weighted least-squares then
        #   J.T * W * J
        #   where W = I/sig_i**2
        hess = j.T @ (np.diag(wt) @ j)
        #hess = mjac.T @ mjac  # not weighted
        # cov = H-1, covariance matrix is inverse of Hessian matrix
        cov_orig = utils.inverse(hess)
        # Rescale to get an unbiased estimate
        # cov_scaled = cov * (RSS/(m-n)), where m=number of measurements, n=number of parameters
        # RSS = residual sum of squares
        #  using rss gives values consistent with what curve_fit returns
        #cov = cov_orig * (np.sum(resid**2)/(self.ntotpix-len(self.pars)))
        # Use chi-squared, since we are doing the weighted least-squares and weighted Hessian
        chisq = np.sum(resid**2/err**2)
        cov = cov_orig * (chisq/(n-3))  # what MPFITFUN suggests, but very small

        # cov = lqr.jac_covariange(mjac,resid,wt)
        
        return cov

    def starfit(self,i,minpercdiff=0.5):
        """
        Fit a single star
        single iteration of the non-linear least-squares loop
        """

        if i > self.nstars-1:
            raise IndexError('Index',i,' is out of bounds.',self.nstars,' stars')
        
        if self.freezestars[i]:
            return

        # Get fitting pixel information
        pars,fxind,fyind,fravelindex = self.starfitdata(i)
        resid = self.resid[fravelindex]
        err = self.error.ravel()[fravelindex]
        wt = 1/err**2    # weights

        # Calculate the local sky for this star (relative to the smooth sky image)
        sky = self.starsky(i)
        # subtract sky from the residuals
        resid -= sky
        
        # Jacobian solvers
        # Get the Jacobian and model
        #  only for pixels that are affected by the "free" parameters
        model0,j = self.psfjac(fxind,fyind,pars)
        # Solve Jacobian
        dbeta = utils.qr_jac_solve(j,resid,weight=wt)
        dbeta[~np.isfinite(dbeta)] = 0.0  # deal with NaNs, shouldn't happen
            
        # Perform line search
        # This has the previous best-fit model subtracted
        #  add it back in
        data = resid + model0
        # Models
        model1 = self.psf(fxind,fyind,pars+0.5*dbeta)
        model2 = self.psf(fxind,fyind,pars+dbeta)
        chisq0 = np.sum((data-model0)**2/err**2)
        chisq1 = np.sum((data-model1)**2/err**2)
        chisq2 = np.sum((data-model2)**2/err**2)
        if self.verbose:
            print('linesearch:',chisq0,chisq1,chisq2)
        alpha = utils.quadratic_bisector(np.array([0.0,0.5,1.0]),
                                         np.array([chisq0,chisq1,chisq2]))
        alpha = np.minimum(np.maximum(alpha,0.0),1.0)  # 0<alpha<1
        if np.isfinite(alpha)==False:
            alpha = 1.0
        pars_new = pars + alpha * dbeta
        new_dbeta = alpha * dbeta

        # Update parameters
        oldpars = pars.copy()
        bounds = self.mkbounds(pars,self.imshape)
        maxsteps = self.steps(pars)
        bestpars = self.newpars(pars,new_dbeta,bounds,maxsteps)
        # Check differences and changes
        diff = np.abs(bestpars-oldpars)
        percdiff = diff.copy()*0
        percdiff[0] = diff[0]/np.maximum(oldpars[0],0.0001)*100  # amp
        percdiff[1] = diff[1]*100               # x
        percdiff[2] = diff[2]*100               # y
        
        # Freeze parameters/stars that converged
        #  also subtract models of fixed stars
        #  also return new free parameters
        if self.nofreeze==False and np.sum(percdiff<=minpercdiff)==3:
            # frzpars = percdiff<=minpercdiff
            # freeparsind, = np.where(self.freezepars==False)
            # bestpar = self.freeze(bestpar,frzpars)

            # set NITER for this star

            self.freezestars[i] = True
            
            
            # # If the only free parameter is the sky offset,
            # #  then freeze it as well
            # if self.nfreepars==1 and self.freepars[-1]==True:
            #     self.freezepars[-1] = True
            #     bestpar = np.zeros((1),np.float64)
            # npar = len(bestpar)
            if self.verbose:
                print('Star ',i,' frozen')
        else:
            self.pars[3*i:3*i+3] = bestpars
        maxpercdiff = np.max(percdiff)
                
        # Update the residuals for a star's full footprint
        #  add in the previous full footprint model
        #  and subtract the new full footprint model
        _,xind,yind,ravelindex = self.stardata(i)        
        prevmodel = self.psf(xind,yind,oldpars)
        newmodel = self.psf(xind,yind,bestpars)
        self.resid[ravelindex] += prevmodel
        self.resid[ravelindex] -= newmodel

        # Update the model image
        self.modelim[ravelindex] += prevmodel
        self.modelim[ravelindex] -= newmodel
        
        # Calculate chisq with updated resid array
        bestchisq = np.sum(self.resid[fravelindex]**2/err**2)
        rms = np.sqrt(np.mean((self.resid[fravelindex]/bestpars[0])**2))
        if self.verbose:
            print('best chisq =',bestchisq)
        self.starchisq[i] = bestchisq
        self.starrms[i] = rms
        
        if self.verbose:
            print('Iter = ',self.niter)
            print('Pars = ',self.starpars[i])
            print('Percent diff = ',percdiff)
            print('Diff = ',diff)
            print('chisq = ',bestchisq)

    def chisq(self):
        """ Compute total chi-square of the current best-fit solution for all
        fitting pixels."""
        ravelindex = self.ravelindex    # all fitting pixels
        chisq = np.sum(self.resid[ravelindex]**2/self.error.ravel()[ravelindex]**2)
        return chisq
    
    def fit(self,maxiter=10,minpercdiff=0.5,reskyiter=2,nofreeze=False,verbose=False):
        """
        Fit PSF to all stars iteratively

        Parameters
        ----------
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

        Example
        -------

        outtab,model,sky = gf.fit()

        """

        # Iterate
        self.niter = 1
        maxpercdiff = 1e10
        while (self.niter<maxiter and self.nfreestars>0):
            start0 = clock()
            
            # Star loop
            for i in range(self.nstars):
                # Fit the single star (if not frozen)
                if self.freezestars[i]==False:
                    self.starfit(i)

            # Re-estimate the sky
            if self.niter % reskyiter == 0:
                if self.verbose:
                    print('Re-estimating the sky')
                self.sky()

            if verbose:
                print('iter dt =',(clock()-start0)/1e9,'sec.')
                
            self.niter += 1     # increment counter

            if self.verbose:
                print('niter=',self.niter)

        # Check that all starniter are set properly
        #  if we stopped "prematurely" then not all stars were frozen
        #  and didn't have starniter set
        self.starniter[np.where(self.starniter==0)] = self.niter

        # Calculate parameter uncertainties
        # estimate uncertainties
        # Calculate covariance matrix
        perror = np.zeros(len(self.pars),np.float64)
        for i in range(self.nstars):
            cov1 = self.starcov(i)
            perror1 = np.sqrt(np.diag(cov1))
            perror[3*i:3*i+3] = perror1
        self.perror[:] = perror

@njit(cache=True)
def numba_allfit(psftype,psfparams,psfnpix,psflookup,psfflux,
                 image,error,mask,tab,fitradius,maxiter=10,
                 minpercdiff=0.5,reskyiter=2,verbose=False,
                 nofreeze=False):
    """
    This is a thin wrapper around AllFitter()
    so we can cache it.
    """
    af = AllFitter(psftype,psfparams,psfnpix,psflookup,
                   image,error,mask,tab,fitradius,verbose,nofreeze)
    # Fit the stars iteratively
    #af.fit()
    af.fit(maxiter,minpercdiff,reskyiter,nofreeze,verbose)
    
    # Return the information that we want
    model = af.modelim.copy() #.reshape(af.imshape)
    skyim = af.skyim.copy()  #.reshape(af.imshape)
    pars = af.pars
    perror = af.perror
    # Put in catalog
    outtab = np.zeros((af.nstars,15),np.float64)
    outtab[:,0] = np.arange(af.nstars)+1           # id
    outtab[:,1] = pars[0:-1:3]                       # amp
    outtab[:,2] = perror[0:-1:3]                     # amp_error
    outtab[:,3] = pars[1::3]                         # x
    outtab[:,4] = perror[1::3]                       # x_error
    outtab[:,5] = pars[2::3]                         # y
    outtab[:,6] = perror[2::3]                       # y_error
    outtab[:,7] = af._starsky                        # sky
    outtab[:,8] = outtab[:,1]*psfflux                # flux
    outtab[:,9] = outtab[:,2]*psfflux                # flux_error
    outtab[:,10] = -2.5*np.log10(np.maximum(outtab[:,8],1e-10))+25.0   # mag
    outtab[:,11] = (2.5/np.log(10))*outtab[:,9]/outtab[:,8]            # mag_error
    outtab[:,12] = af.starrms
    outtab[:,13] = af.starchisq
    outtab[:,14] = af.starniter                      # niter, what iteration it converged on
    return outtab,model,skyim
    

def allfit(psf,image,tab,fitradius=0.0,maxiter=10,minpercdiff=0.5,
           reskyiter=2,nofreeze=False,verbose=False):
    """
    Fit PSF to all stars in an image iteratively.

    Parameters
    ----------
    psf : PSF object
       PSF object to use for the fitting.
    image : CCDData object
       Image to use to fit PSF model to stars.
    tab : table
       Catalog with initial amp/x/y values for the stars to use to fit the PSF.
       To pre-group the stars, add a "group_id" in the catalog.
    fitradius : float, optional
       The fitting radius in pixels.  By default the PSF FWHM is used.
           reskyiter=2,nofreeze=False,skyfit=True,verbose=False):
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

    Example
    -------

    outtab,model,sky = gf.fit()

    """

    t0 = time.time()

    # if skyfit==False:
    #     self.freezepars[-1] = True  # freeze sky value
        
    # # Centroids fixed
    # if recenter==False:
    #     self.freezepars[1::3] = True  # freeze X values
    #     self.freezepars[2::3] = True  # freeze Y values

    # Create AllFitter object
    fitradius = 3.0
    if psf.haslookup:
        psflookup = psf.lookup.copy().astype(np.float64)
    else:
        psflookup = np.zeros((1,1,1),np.float64)
    psfflux = psf.flux()
    out = numba_allfit(psf.psftype,psf.params,psf.npix,psflookup,psfflux,
                       image.data,image.error,image.mask,tab,fitradius,
                       maxiter,minpercdiff,reskyiter,verbose,nofreeze)
    outtab,model,skyim = out
    model = model.reshape(image.shape)
    skyim = skyim.reshape(image.shape)
    return outtab,model,skyim
        
    #af = AllFitter(psf.psftype,psf.params,psf.npix,psflookup,
    #               image.data,image.error,image.mask,tab,
    #               fitradius,verbose,nofreeze)
    ## Fit the stars iteratively
    #af.fit()
    
    # Make final model
    # self.unfreeze()
    model = af.modelim.copy().reshape(af.imshape)
    skyim = af.skyim.copy().reshape(af.imshape)

    # Parameters and uncertainties
    pars = af.pars
    perror = af.perror

    # Put in catalog
    outtab = np.zeros((af.nstars,15),np.float64)
    outtab[:,0] = np.arange(af.nstars)+1           # id
    outtab[:,1] = pars[0:-1:3]                       # amp
    outtab[:,2] = perror[0:-1:3]                     # amp_error
    outtab[:,3] = pars[1::3]                         # x
    outtab[:,4] = perror[1::3]                       # x_error
    outtab[:,5] = pars[2::3]                         # y
    outtab[:,6] = perror[2::3]                       # y_error
    outtab[:,7] = af._starsky                        # sky
    psfflux = psf.flux()
    outtab[:,8] = outtab[:,1]*psfflux                # flux
    outtab[:,9] = outtab[:,2]*psfflux                # flux_error
    outtab[:,10] = -2.5*np.log10(np.maximum(outtab[:,8],1e-10))+25.0   # mag
    outtab[:,11] = (2.5/np.log(10))*outtab[:,9]/outtab[:,8]            # mag_error
    outtab[:,12] = af.starrms
    outtab[:,13] = af.starchisq
    outtab[:,14] = af.starniter                      # niter, what iteration it converged on

    if verbose:
        print('dt =',(clock()-start)/1e9,'sec.')

    return outtab,model,skyim

        

#@njit
def fit(psf,image,tab,fitradius=0.0,recenter=True,maxiter=10,minpercdiff=0.5,
        reskyiter=2,nofreeze=False,skyfit=True,verbose=False):
    """
    Fit PSF to all stars in an image.

    To pre-group the stars, add a "group_id" in the input catalog.

    Parameters
    ----------
    psf : PSF object
       PSF object with initial parameters to use.
    image : CCDData object
       Image to use to fit PSF model to stars.
    tab : table
       Catalog with initial amp/x/y values for the stars to use to fit the PSF.
       To pre-group the stars, add a "group_id" in the catalog.
    fitradius : float, optional
       The fitting radius in pixels.  By default the PSF FWHM is used.
    recenter : boolean, optional
       Allow the centroids to be fit.  Default is True.
    maxiter : int, optional
       Maximum number of iterations to allow.  Only for methods "qr" or "svd".
       Default is 10.
    minpercdiff : float, optional
       Minimum percent change in the parameters to allow until the solution is
       considered converged and the iteration loop is stopped.  Only for methods
       "qr" and "svd".  Default is 0.5.
    reskyiter : int, optional
       After how many iterations to re-calculate the sky background. Default is 2.
    nofreeze : boolean, optional
       Do not freeze any parameters even if they have converged.  Default is False.
    skyfit : boolean, optional
       Fit a constant sky offset with the stellar parameters.  Default is True.
    verbose : boolean, optional
       Verbose output.

    Returns
    -------
    results : table
       Table of best-fitting parameters for each star.
       id, amp, amp_error, x, x_err, y, y_err, sky
    model : numpy array
       Best-fitting model of the stars and sky background.

    Example
    -------

    results,model = fit(psf,image,tab,groups)

    """

    start = clock()
    nstars = len(tab)
    ny,nx = image.shape

    # Groups
    if 'group_id' not in tab.keys():
        daogroup = DAOGroup(crit_separation=2.5*psf.fwhm())
        starlist = tab.copy()
        starlist['x_0'] = tab['x']
        starlist['y_0'] = tab['y']
        # THIS TAKES ~4 SECONDS!!!!!! WAY TOO LONG!!!!
        star_groups = daogroup(starlist)
        tab['group_id'] = star_groups['group_id']

    # Star index
    starindex = utils.index(np.array(tab['group_id']))
    #starindex = dln.create_index(np.array(tab['group_id'])) 
    groups = starindex['value']
    ngroups = len(groups)
    if verbose:
        print(ngroups,'star groups')

    # Initialize catalog
    #dt = np.dtype([('id',int),('amp',float),('amp_error',float),('x',float),
    #               ('x_error',float),('y',float),('y_error',float),('sky',float),
    #               ('flux',float),('flux_error',float),('mag',float),('mag_error',float),
    #               ('niter',int),('group_id',int),('ngroup',int),('rms',float),('chisq',float)])
    outtab = np.zeros((nstars,17),dtype=dt)
    outtab[:,0] = tab[:,0]  # copy ID


    # Group Loop
    #---------------
    resid = image.copy()
    outmodel = np.zeros(image.shape,np.float64)
    outsky = np.zeros(image.shape,np.float64)
    for g,grp in enumerate(groups):
        ind = starindex['index'][starindex['lo'][g]:starindex['hi'][g]+1]
        nind = len(ind)
        inptab = tab[ind].copy()
        if 'amp' not in inptab.columns:
            # Estimate amp from flux and fwhm
            # area under 2D Gaussian is 2*pi*A*sigx*sigy
            if 'fwhm' in inptab.columns:
                amp = inptab['flux']/(2*np.pi*(inptab['fwhm']/2.35)**2)
            else:
                amp = inptab['flux']/(2*np.pi*(psf.fwhm()/2.35)**2)                
            staramp = np.maximum(amp,0)   # make sure it's positive
            inptab['amp'] = staramp
        
        if verbose:
            print('-- Group '+str(grp)+'/'+str(len(groups))+' : '+str(nind)+' star(s) --')

        # Single Star
        if nind==1:
            inptab = np.array([inptab[1],inptab[2],inptab[3]])
            out,model = psf.fit(resid,inptab,niter=3,verbose=verbose,retfullmodel=True,recenter=recenter)
            model.data -= out['sky']   # remove sky
            outmodel.data[model.bbox.slices] += model.data
            outsky.data[model.bbox.slices] = out['sky']

        # Group
        else:
            bbox = cutoutbbox(image,psf,inptab)
            out,model,sky = gfit.fit(psf,resid[bbox.slices],inptab,fitradius=fitradius,
                                     recenter=recenter,maxiter=maxiter,minpercdiff=minpercdiff,
                                     reskyiter=reskyiter,nofreeze=nofreeze,verbose=verbose,
                                     skyfit=skyfit,absolute=True)
            outmodel.data[model.bbox.slices] += model.data
            outsky.data[model.bbox.slices] = sky
            
        # Subtract the best model for the group/star
        resid[model.bbox.slices].data -= model.data

        # Put in catalog
        cols = ['amp','amp_error','x','x_error','y','y_error',
                'sky','flux','flux_error','mag','mag_error','niter','rms','chisq']
        for c in cols:
            outtab[c][ind] = out[c]
        outtab['group_id'][ind] = grp
        outtab['ngroup'][ind] = nind
        outtab = Table(outtab)
        
    if verbose:
        print('dt = {:.2f} sec.'.format(time.time()-t0))
    
    return outtab,outmodel,outsky
        
            

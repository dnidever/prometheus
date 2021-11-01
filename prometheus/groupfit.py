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
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
from scipy.optimize import curve_fit, least_squares
from scipy.interpolate import interp1d
from scipy import sparse
#from astropy.nddata import CCDData,StdDevUncertainty
from dlnpyutils import utils as dln, bindata
import copy
import logging
import time
import matplotlib
import sep
from photutils.aperture import CircularAnnulus
from astropy.stats import sigma_clipped_stats
from . import leastsquares as lsq
from .ccddata import CCDData,BoundingBox

# Fit a PSF model to multiple stars in an image

    
class GroupFitter(object):

    def __init__(self,psf,image,cat,fitradius=None,verbose=False):
        # Save the input values
        self.verbose = verbose
        self.psf = psf
        self.image = image
        self.cat = cat
        self.nstars = np.size(cat)  # number of stars
        self.niter = 0              # number of iterations in the solver
        self.npsfpix = psf.npix     # shape of PSF
        ny,nx = image.data.shape    # save image dimensions, python images are (Y,X)
        self.nx = nx
        self.ny = ny
        if fitradius is None:       # PSF fitting radius
            fitradius = psf.fwhm()
        self.fitradius = fitradius
        self.nfitpix = int(np.ceil(fitradius))  # +/- nfitpix
        # Star heights
        if 'height' in cat.colnames:
            starheight = cat['height'].copy()
        else:
            # estimate height from flux and fwhm
            # area under 2D Gaussian is 2*pi*A*sigx*sigy
            if 'fwhm' in cat.columns:
                height = cat['flux']/(2*np.pi*(cat['fwhm']/2.35)**2)
            else:
                height = cat['flux']/(2*np.pi*(psf.fwhm()/2.35)**2)                
            starheight = np.maximum(height,0)   # make sure it's positive
        # Initialize the parameter array
        pars = np.zeros(self.nstars*3,float) # height, xcen, ycen
        pars[0::3] = starheight
        pars[1::3] = cat['x']
        pars[2::3] = cat['y']
        self.pars = pars
        # Sky and Niter arrays for the stars
        self.starsky = np.zeros(self.nstars,float)
        self.starniter = np.zeros(self.nstars,int)
        self.njaciter = 0  # initialize njaciter
        # Initialize the freezepars and freezestars arrays
        self.freezestars = np.zeros(self.nstars,bool)
        self.freezepars = np.zeros(self.nstars*3,bool)
        self.pixused = None   # initialize pixused
        
        # Get xdata, ydata
        bboxdata = []
        xlist = []
        ylist = []
        fbboxdata = []
        fxlist = []
        fylist = []
        ntotpix = 0
        hpsfnpix = self.psf.npix//2
        for i in range(self.nstars):
            xcen = self.starxcen[i]
            ycen = self.starycen[i]
            # Full PSF region
            fbbox = psf.starbbox((xcen,ycen),image.shape,hpsfnpix)
            fx,fy = psf.bbox2xy(fbbox)
            frr = np.sqrt( (fx-xcen)**2 + (fy-ycen)**2 )
            # Use image mask
            #  mask=True for bad values
            if image.mask is not None:
                fmask = (frr<=hpsfnpix) & (image.mask[fy,fx]==False)
            else:
                fmask = frr<=hpsfnpix                
            fx = fx[fmask]  # raveled
            fy = fy[fmask]
            fbboxdata.append(fbbox)
            fxlist.append(fx)
            fylist.append(fy)
            # Fitting region
            bbox = psf.starbbox((xcen,ycen),image.shape,self.nfitpix)
            x,y = psf.bbox2xy(bbox)
            rr = np.sqrt( (x-xcen)**2 + (y-ycen)**2 )
            # Use image mask
            #  mask=True for bad values
            if image.mask is not None:           
                mask = (rr<=self.fitradius) & (image.mask[y,x]==False)
            else:
                mask = rr<=self.fitradius                
            x = x[mask]  # raveled
            y = y[mask]
            ntotpix += x.size
            bboxdata.append(bbox)  # this still includes the corners
            xlist.append(x)
            ylist.append(y)

        self.fxlist = fxlist
        self.fylist = fylist
        self.xlist = xlist
        self.ylist = ylist
            
        # Combine all of the X and Y values into one array
        xall = np.zeros(ntotpix,int)
        yall = np.zeros(ntotpix,int)
        count = 0
        for i in range(self.nstars):
            n = len(xlist[i])
            xall[count:count+n] = xlist[i]
            yall[count:count+n] = ylist[i]
            count += n
            
        # Create 1D unraveled indices, python images are (Y,X)
        ind1 = np.ravel_multi_index((yall,xall),image.shape)
        # Get unique indexes and inverse indices
        #   the inverse index list takes you from the duplicated pixels
        #   to the unique ones
        uind1,invindex = np.unique(ind1,return_inverse=True)
        ntotpix = len(uind1)
        y,x = np.unravel_index(uind1,image.shape)
        
        imflatten = image.data.ravel()[uind1]
        errflatten = image.error.ravel()[uind1]
        # Save information on the "flattened" arrays
        self.ntotpix = ntotpix
        self.imflatten = imflatten
        self.resflatten = imflatten.copy()
        self.errflatten = errflatten
        self.ind1 = uind1
        self.x = x
        self.y = y
        self.invindex = invindex  # takes you from duplicates to unique pixels
        invindexlist = []
        count = 0
        for i in range(self.nstars):
            n = len(self.xlist[i])
            invindexlist.append(invindex[count:count+n])
            count += n
        self.invindexlist = invindexlist
        self.bbox = BoundingBox(np.min(x),np.max(x),np.min(y),np.max(y))
        self.bboxdata = bboxdata

        # Create initial sky image
        self.sky()

    @property
    def starpars(self):
        """ Return the [height,xcen,ycen] parameters in [Nstars,3] array.
            You can GET a star's parameters like this:
            pars = self.starpars[4]
            You can also SET a star's parameters a similar way:
            self.starpars[4] = pars
        """
        return self.pars.reshape(self.nstars,3)
    
    @property
    def starheight(self):
        """ Return the best-fit heights for all stars."""
        return self.pars[0::3]

    @starheight.setter
    def starheight(self,val):
        """ Set starheight values."""
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
    
    def sky(self,method='sep',rin=None,rout=None):
        """ (Re)calculate the sky."""
        # Remove the current best-fit model
        resid = self.image.data-self.modelim  # remove model
        # SEP smoothly varying background
        if method=='sep':
            bw = np.maximum(int(self.nx/10),64)
            bh = np.maximum(int(self.ny/10),64)
            bkg = sep.Background(resid, mask=None, bw=bw, bh=bh, fw=3, fh=3)
            self.skyim = bkg.back()
            # Calculate sky value for each star
            #  use center position
            self.starsky[:] = self.skyim[np.round(self.starycen).astype(int),np.round(self.starxcen).astype(int)]
        # Annulus aperture
        elif method=='annulus':
            if rin is None:
                rin = self.psf.fwhm()*1.5
            if rout is None:
                rout = self.psf.fwhm()*2.5
            positions = list(zip(self.starxcen,self.starycen))
            annulus = CircularAnnulus(positions,r_in=rin,r_out=rout)
            for i in range(self.nstars):
                annulus_mask = annulus[i].to_mask(method='center')
                annulus_data = annulus_mask.multiply(resid,fill_value=np.nan)
                data = annulus_data[(annulus_mask.data>0) & np.isfinite(annulus_data)]
                mean_sigclip, median_sigclip, _ = sigma_clipped_stats(data,stdfunc=dln.mad)
                self.starsky[i] = mean_sigclip
            if hasattr(self,'skyim') is False:
                self.skyim = np.zeros(self.image.shape,float)
            if self.skyim is None:
                self.skyim = np.zeros(self.image.shape,float)
            self.skyim += np.median(self.starsky)
        else:
            raise ValueError("Sky method "+method+" not supported")

    @property
    def skyflatten(self):
        """ Return the sky values for the pixels that we are fitting."""
        return self.skyim.ravel()[self.ind1]
        
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
        self.freezestars = np.sum(self.freezepars.reshape(self.nstars,3),axis=1)==3
        #self.nfreezestars = np.sum(self.freezestars)
        # Subtract model for newly frozen stars
        newfreezestars, = np.where((oldfreezestars==False) & (self.freezestars==True))
        if len(newfreezestars)>0:
            # add models to a full image
            newmodel = self.image.data.copy()*0
            for i in newfreezestars:
                # Save on what iteration this star was frozen
                self.starniter[i] = self.niter+1
                print('freeze: subtracting model for star ',i)
                pars1 = self.pars[i*3:(i+1)*3]
                #xind = self.xlist[i]
                #yind = self.ylist[i]
                #invindex = self.invindexlist[i]
                #im1 = self.psf(xind,yind,pars1)
                #self.resflatten[invindex] -= im1
                xind = self.fxlist[i]
                yind = self.fylist[i]
                im1 = self.psf(xind,yind,pars1)
                newmodel[yind,xind] += im1
            # Only keep the pixels being fit
            #  and subtract from the residuals
            newmodel1 = newmodel.ravel()[self.ind1]
            self.resflatten -= newmodel1
                
        # Return the new array of free parameters
        frzind = np.arange(len(frzpars))[frzpars]
        pars = np.delete(pars,frzind)
        return pars
                         
    def unfreeze(self):
        """ Unfreeze all parameters and stars."""
        self.freezestars = np.zeros(self.nstars,bool)
        self.freezepars = np.zeros(self.nstars*3,bool)
        self.resflatten = self.imflatten.copy()

    @property
    def modelim(self):
        """ This returns the full image of the current best model (no sky)
            using the PARS values."""
        im = np.zeros(self.image.shape,float)
        for i in range(self.nstars):
            pars = self.pars[i*3:(i+1)*3]
            fxind = self.fxlist[i]
            fyind = self.fylist[i]
            im1 = self.psf(fxind,fyind,pars)
            im[fyind,fxind] += im1
        return im
        
    @property
    def modelflatten(self):
        """ This returns the current best model (no sky) for only the "flatten" pixels
            using the PARS values."""
        return self.model(np.arange(10),*self.pars,allparams=True,verbose=False)
        
    def model(self,x,*args,trim=False,allparams=False,verbose=None):
        """ Calculate the model for the pixels we are fitting."""
        # ALLPARAMS:  all of the parameters were input

        if verbose is None and self.verbose:
            print('model: ',self.niter,args)

        # Args are [height,xcen,ycen,sky] for all Nstars
        # so 3*Nstars parameters

        psf = self.psf

        # Figure out the parameters of ALL the stars
        #  some stars and parameters are FROZEN
        if self.nfreezepars>0 and allparams is False:
            allpars = self.pars
            allpars[self.freepars] = args
        else:
            allpars = args
        
        x0,x1 = self.bbox.xrange
        y0,y1 = self.bbox.yrange
        nx = x1-x0
        ny = y1-y0
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
            xind = self.xlist[i]
            yind = self.ylist[i]
            invindex = self.invindexlist[i]
            im1 = psf(xind,yind,pars)
            allim[invindex] += im1
            usepix[invindex] = True

        self.usepix = usepix
        nusepix = np.sum(usepix)
        
        if trim and nusepix<self.ntotpix:            
            unused = np.arange(self.ntotpix)[~usepix]
            allim = np.delete(allim,unused)
        
        self.niter += 1
        
        return allim

    
    def jac(self,x,*args,retmodel=False,trim=False,allparams=False,verbose=None):
        """ Calculate the jacobian for the pixels and parameters we are fitting"""

        if verbose is None and self.verbose:
            print('jac: ',self.njaciter,args)

        # Args are [height,xcen,ycen,sky] for all Nstars
        # so 3*Nstars parameters
        
        psf = self.psf

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
        
        x0,x1 = self.bbox.xrange
        y0,y1 = self.bbox.yrange
        nx = x1-x0
        ny = y1-y0
        #jac = np.zeros((nx,ny,len(args)),float)    # image covered by star
        jac = np.zeros((self.ntotpix,len(self.pars)),float)    # image covered by star
        usepix = np.zeros(self.ntotpix,bool)
        if retmodel:
            im = np.zeros(self.ntotpix,float)

        # Loop over the stars and generate the model image        
        # ONLY LOOP OVER UNFROZEN STARS
        if allparams is False:
            dostars = np.arange(self.nstars)[self.freestars]
        else:
            dostars = np.arange(self.nstars)
        for i in dostars:
            pars = allpars[i*3:(i+1)*3]
            bbox = self.bboxdata[i]
            xind = self.xlist[i]
            yind = self.ylist[i]
            invindex = self.invindexlist[i]
            xdata = (xind,yind)
            if retmodel:
                m,jac1 = psf.jac(xdata,*pars,retmodel=True)
            else:
                jac1 = psf.jac(xdata,*pars)
            jac[invindex,i*3] = jac1[:,0]
            jac[invindex,i*3+1] = jac1[:,1]
            jac[invindex,i*3+2] = jac1[:,2]
            #jac[invindex,i*4+3] = jac1[:,3]
            if retmodel:
                im[invindex] += m
            usepix[invindex] = True

        # Remove frozen columns
        if self.nfreezepars>0 and allparams is False:
            jac = np.delete(jac,np.arange(len(self.pars))[self.freezepars],axis=1)


        self.usepix = usepix
        nusepix = np.sum(usepix)

        # Trim out unused pixels
        if trim and nusepix<self.ntotpix:
            unused = np.arange(self.ntotpix)[~usepix]
            jac = np.delete(jac,unused,axis=0)
            if retmodel:
                im = np.delete(im,unused)
        
        self.njaciter += 1
        
        if retmodel:
            return im,jac
        else:
            return jac

    def heightfit(self,trim=True):
        """ Fit the heights only for the stars."""

        # linear least squares problem
        # Ax = b
        # A is the set of models pixel values for height, [Npix, Nstar]
        # x is heights [Nstar] we are solving for
        # b is pixel values, oru residflatten values
        
        # All parameters
        allpars = self.pars
        
        A = np.zeros((self.ntotpix,self.nstars),float)
        usepix = np.zeros(self.ntotpix,bool)

        # Loop over the stars and generate the model image        
        # ONLY LOOP OVER UNFROZEN STARS
        dostars = np.arange(self.nstars)[self.freestars]
        guess = np.zeros(self.nfreestars,float)
        for count,i in enumerate(dostars):
            pars = allpars[i*3:(i+1)*3].copy()
            guess[count] = pars[0]
            pars[0] = 1.0  # unit height
            xind = self.xlist[i]
            yind = self.ylist[i]
            invindex = self.invindexlist[i]
            im1 = self.psf(xind,yind,pars)
            A[invindex,i] = im1
            usepix[invindex] = True

        nusepix = np.sum(usepix)

        # Residual data
        dy = self.resflatten-self.skyflatten
        
        if trim and nusepix<self.ntotpix:
            unused = np.arange(self.ntotpix)[~usepix]
            A = np.delete(A,unused,axis=0)
            dy = dy[usepix]

        from scipy import sparse
        A = sparse.csc_matrix(A)   # make sparse

        # Use guess to get close to the solution
        dy2 = dy - A.dot(guess)
        par = sparse.linalg.lsqr(A,dy2,atol=1e-4,btol=1e-4)
        dheight = par[0]
        height = guess+dheight
        
        # preconditioning!
        
        return height

    
    def centroid(self):
        """ Centroid all of the stars."""

        # Start with the residual image and all stars subtracted.
        # Then add in the model of one star at a time and centroid it with the best-fitting model.

        # All parameters
        allpars = self.pars

        resid = self.image.data.copy()-self.skyim
        
        # Generate full models 
        # Loop over the stars and generate the model image        
        # ONLY LOOP OVER UNFROZEN STARS
        fmodels = []
        #fjac = []
        models = []
        jac = []
        dostars = np.arange(self.nstars)[self.freestars]
        usepix = np.zeros(self.ntotpix,bool)        
        for count,i in enumerate(dostars):
            pars = self.starpars[i]
            #pars = allpars[i*3:(i+1)*3]
            # Full models
            fxind = self.fxlist[i]
            fyind = self.fylist[i]
            fim1 = self.psf(fxind,fyind,pars)
            resid[fyind,fxind] -= fim1
            fmodels.append(fim1)            
            #fjac.append(fjac1)
            #fusepix[finvindex] = True
            # Only fitting models
            xind = self.xlist[i]
            yind = self.ylist[i]
            invindex = self.invindexlist[i]
            im1,jac1 = self.psf.jac((xind,yind),*pars,retmodel=True)
            models.append(im1)
            jac.append(jac1)
            #usepix[invindex] = True

        # Loop over all free stars and fit centroid
        xnew = np.zeros(self.nfreestars,float)
        ynew = np.zeros(self.nfreestars,float)
        for count,i in enumerate(dostars):
            pars = self.starpars[i]
            freezepars = self.freezepars[i*3:(i+1)*3]
            xnew[count] = pars[1]
            ynew[count] = pars[2]

            # Both x/y frozen
            if freezepars[1]==True and freezepars[2]==True:
                continue
            
            # Add the model for this star back in
            fxind = self.fxlist[i]
            fyind = self.fylist[i]
            fmodel1 = fmodels[count]
            #resid[fyind,fxind] += fmodel1
            
            # crowdsource, does a sum
            #  y = 2 / integral(P*P*W) * integral(x*(I-P)*W)
            #  where x = x/y coordinate, I = isolated stamp, P = PSF model, W = weight

            # Use the derivatives instead
            xind = self.xlist[i]
            yind = self.ylist[i]
            jac1 = jac[count]
            jac1 = np.delete(jac1,0,axis=1)  # delete height column
            resid1 = resid[yind,xind] 

            # CHOLESKY_JAC_SOLVE NEEDS THE ACTUAL RESIDUALS!!!!
            #  with the star removed!!
            
            # Use cholesky to solve
            # If only one position frozen, solve for the free one
            # X frozen, solve for Y
            if freezepars[1]==True:
                jac1 = np.delete(jac1,0,axis=1)
                dbeta = cholesky_jac_solve(jac1,resid1)
                ynew[count] += dbeta[0]
            # Y frozen, solve for X
            elif freezepars[2]==True:
                jac1 = np.delete(jac1,1,axis=1)
                dbeta = cholesky_jac_solve(jac1,resid1)
                xnew[count] += dbeta[0]
            # Solve for both X and Y
            else:
                dbeta = cholesky_jac_solve(jac1,resid1)
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
        mjac = self.jac(xdata,*self.pars,allparams=True,trim=False,verbose=False)
        # Weights
        #   If weighted least-squares then
        #   J.T * W * J
        #   where W = I/sig_i**2
        wt = np.diag(1/self.errflatten**2)
        hess = mjac.T @ (wt @ mjac)
        #hess = mjac.T @ mjac  # not weighted
        # cov = H-1, covariance matrix is inverse of Hessian matrix
        cov_orig = np.linalg.inv(hess)
        # Rescale to get an unbiased estimate
        # cov_scaled = cov * (RSS/(m-n)), where m=number of measurements, n=number of parameters
        # RSS = residual sum of squares
        #  using rss gives values consistent with what curve_fit returns
        bestmodel = self.model(xdata,*self.pars,allparams=True,trim=False,verbose=False)
        resid = self.imflatten-self.skyflatten-bestmodel
        #cov = cov_orig * (np.sum(resid**2)/(self.ntotpix-len(self.pars)))
        # Use chi-squared, since we are doing the weighted least-squares and weighted Hessian
        chisq = np.sum(resid**2/self.errflatten**2)        
        cov = cov_orig * (chisq/(self.ntotpix-len(self.pars)))  # what MPFITFUN suggests, but very small

        # cov = lqr.jac_covariange(mjac,resid,wt)
        
        return cov

        
    
def fit(psf,image,cat,method='qr',fitradius=None,maxiter=10,minpercdiff=0.5,reskyiter=2,
        nofreeze=False,absolute=False,verbose=False):
    """
    Fit PSF to group of stars in an image.

    Parameters
    ----------
    psf : PSF object
       PSF object with initial parameters to use.
    image : CCDData object
       Image to use to fit PSF model to stars.
    cat : table
       Catalog with initial height/x/y values for the stars to use to fit the PSF.
    method : str, optional
       Method to use for solving the non-linear least squares problem: "cholesky",
       "qr", "svd", and "curve_fit".  Default is "cholesky".
    fitradius: float, optional
       The fitting radius in pixels.  By default the PSF FWHM is used.
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
    verbose : boolean, optional
       Verbose output.

    Returns
    -------
    out : table
       Table of best-fitting parameters for each star.
       id, height, height_error, x, x_err, y, y_err, sky
    model : numpy array
       Best-fitting model of the stars and sky background.
    sky : numpy array
       Best-fitting sky image.

    Example
    -------

    outcat,model,sky = fit(psf,image,cat)

    """

    start = time.time()
    
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
    gf = GroupFitter(psf,image,cat,fitradius=fitradius,verbose=verbose)
    xdata = np.arange(gf.ntotpix)


    # DO THE CONTRIBUTIONS OF WINGS OF THE PROFILE TO THE NEIGHBORING STARS FITTING PIXELS
    # NEED TO BE TAKEN INTO ACCOUNT???  I THINK SO
    print('TAKE WINGS INTO ACCOUNT IN FITTING PIXELS!!!')
    import pdb; pdb.set_trace()
    
    
    # Perform the fitting
    #--------------------
    
    # Initial estimates
    initpar = np.zeros(nstars*3,float)
    initpar[0::3] = cat['height']
    initpar[1::3] = cat['x']
    initpar[2::3] = cat['y']
    
    # Curve_fit
    #   dealt with separately
    if method=='curve_fit':
        # Perform the fitting
        bounds = [np.zeros(gf.nstars*3,float)-np.inf,
                  np.zeros(gf.nstars*3,float)+np.inf]
        bounds[0][0::3] = 0
        bounds[0][1::3] = cat['x']-2
        bounds[1][1::3] = cat['x']+2
        bounds[0][2::3] = cat['y']-2
        bounds[1][2::3] = cat['y']+2
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
        gf.niter = 0
        maxpercdiff = 1e10
        bestpar = initpar.copy()
        npars = len(bestpar)
        while (gf.niter<maxiter and maxpercdiff>minpercdiff):
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
                dbeta = lsq.jac_solve(jac,dy,method=method,weight=wt)
                
            #  htcen, crowdsource method of solving heights/fluxes first
            #      and then centroiding to get x/y
            else:
                dbeta = np.zeros(3*gf.nfreestars,float)
                # Solve for the heights
                newht = gf.heightfit()
                dbeta[0::3] = gf.starheight[gf.freestars] - newht
                gf.starheight[gf.freestars] = newht  # update the heights
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
                dbeta = dbeta[freepars]

            # Update parameters
            oldpar = bestpar.copy()
            bestpar += dbeta
            # Check differences and changes
            diff = np.abs(bestpar-oldpar)
            percdiff = diff.copy()*0
            percdiff[0::3] = diff[0::3]/oldpar[0::3]*100  # height
            percdiff[1::3] = diff[1::3]*100               # x
            percdiff[2::3] = diff[2::3]*100               # y

            # Freeze parameters/stars that converged
            #  also subtract models of fixed stars
            #  also return new free parameters
            if not nofreeze:
                frzpars = percdiff<=minpercdiff
                freeparsind, = np.where(~gf.freezepars)
                bestpar = gf.freeze(bestpar,frzpars)
                npar = len(bestpar)
                print('Nfrozen pars = ',gf.nfreezepars)
                print('Nfrozen stars = ',gf.nfreezestars)
                print('Nfree pars = ',npar)
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
                print('Iter = ',gf.niter)
                print('Pars = ',gf.pars)
                print('Percent diff = ',percdiff)
                print('Diff = ',diff)
                print('chisq = ',chisq)
                
            # Re-estimate the sky
            if gf.niter % reskyiter == 0:
                print('Re-estimating the sky')
                gf.sky()
        
            print('iter dt = ',time.time()-start0)

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
        print('Best-fitting parameters: ',pars)
        print('Errors: ',perror)

        
    # Put in catalog
    # Initialize catalog
    dt = np.dtype([('id',int),('height',float),('height_error',float),('x',float),
                   ('x_error',float),('y',float),('y_error',float),('sky',float),('niter',int)])
    outcat = np.zeros(nstars,dtype=dt)
    if 'id' in cat.keys():
        outcat['id'] = cat['id']
    else:
        outcat['id'] = np.arange(nstars)+1
    outcat['height'] = pars[0::3]
    outcat['height_error'] = perror[0::3]
    outcat['x'] = pars[1::3]
    outcat['x_error'] = perror[1::3]
    outcat['y'] = pars[2::3]
    outcat['y_error'] = perror[2::3]
    outcat['sky'] = gf.starsky
    outcat['niter'] = gf.starniter  # what iteration it converged on

    # Image offsets for absolute X/Y coordinates
    if absolute:
        outcat['x'] += imx0
        outcat['y'] += imy0
        cat['x'] += imx0
        cat['y'] += imy0        
    
    print('dt = ',time.time()-start)
    
    return outcat,model,gf.skyim

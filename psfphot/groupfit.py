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
from astropy.nddata import CCDData,StdDevUncertainty
from dlnpyutils import utils as dln, bindata
import copy
import logging
import time
import matplotlib
import sep
from photutils.aperture import CircularAnnulus
from astropy.stats import sigma_clipped_stats

# Fit a PSF model to multiple stars in an image

def jac_solve(jac,resid,method=None,weight=None):
    """ Thin wrapper for the various jacobian solver method."""

    if method=='qr':
        dbeta = qr_jac_solve(jac,resid,weight=weight)
    elif method=='svd':
        dbeta = svd_jac_solve(jac,resid,weight=weight)
    elif method=='cholesky':
        dbeta = cholesky_jac_solve(jac,resid,weight=weight)
    elif method=='sparse':
        dbeta = cholesky_jac_sparse_solve(jac,resid,weight=weight)        
    else:
        raise ValueError(method+' not supported')
    
    return dbeta

def qr_jac_solve(jac,resid,weight=None):
    """ Solve part of a non-linear least squares equation using QR decomposition
        using the Jacobian."""
    # jac: Jacobian matrix, first derivatives, [Npix, Npars]
    # resid: residuals [Npix]
    
    # QR decomposition
    q,r = np.linalg.qr(jac)
    rinv = np.linalg.inv(r)
    dbeta = rinv @ (q.T @ resid)
    return dbeta

def svd_jac_solve(jac,resid,weight=None):
    """ Solve part of a non-linear least squares equation using Single Value
        Decomposition (SVD) using the Jacobian."""
    # jac: Jacobian matrix, first derivatives, [Npix, Npars]
    # resid: residuals [Npix]

    # Precondition??
    
    # Singular Value decomposition (SVD)
    u,s,vt = np.linalg.svd(jac)
    #u,s,vt = sparse.linalg.svds(jac)
    # u: [Npix,Npix]
    # s: [Npars]
    # vt: [Npars,Npars]
    # dy: [Npix]
    sinv = s.copy()*0  # pseudo-inverse
    sinv[s!=0] = 1/s[s!=0]
    npars = len(s)
    dbeta = vt.T @ ((u.T @ resid)[0:npars]*sinv)
    return dbeta

def cholesky_jac_sparse_solve(jac,resid,weight=None):
    """ Solve part a non-linear least squares equation using Cholesky decomposition
        using the Jacobian, with sparse matrices."""
    # jac: Jacobian matrix, first derivatives, [Npix, Npars]
    # resid: residuals [Npix]

    # Precondition??
    
    # J * x = resid
    # J.T J x = J.T resid
    # A = (J.T @ J)
    # b = np.dot(J.T*dy)
    # J is [3*Nstar,Npix]
    # A is [3*Nstar,3*Nstar]
    from scipy import sparse
    jac = sparse.csc_matrix(jac)  # make it sparse
    A = jac.T @ jac
    b = jac.T.dot(resid)
    # Now solve linear least squares with sparse
    # Ax = b
    from sksparse.cholmod import cholesky
    factor = cholesky(A)
    dbeta = factor(b)
    
    # Precondition?

    return dbeta
    
def cholesky_jac_solve(jac,resid,weight=None):
    """ Solve part a non-linear least squares equation using Cholesky decomposition
        using the Jacobian."""
    # jac: Jacobian matrix, first derivatives, [Npix, Npars]
    # resid: residuals [Npix]
    
    # J * x = resid
    # J.T J x = J.T resid
    # A = (J.T @ J)
    # b = np.dot(J.T*dy)
    A = jac.T @ jac
    b = np.dot(jac.T,resid)

    # Now solve linear least squares with cholesky decomposition
    # Ax = b    
    return cholesky_solve(A,b)


def cholesky_solve(A,b):
    """ Solve linear least squares problem with Cholesky decomposition."""

    # Now solve linear least squares with cholesky decomposition
    # Ax = b
    # decompose A into L L* using cholesky decomposition
    #  L and L* are triangular matrices
    #  L* is conjugate transpose
    # solve Ly=b (where L*x=y) for y by forward substitution
    # finally solve L*x = y for x by back substitution

    L = np.linalg.cholesky(A)
    Lstar = L.T.conj()   # Lstar is conjugate transpose
    y = scipy.linalg.solve_triangular(L,b)
    x = scipy.linalg.solve_triangular(Lstar,y)
    return x

    
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
        nx,ny = image.data.shape    # save image dimensions
        self.nx = nx
        self.ny = ny
        if fitradius is None:       # PSF fitting radius
            fitradius = psf.fwhm()
        self.fitradius = fitradius
        self.nfitpix = int(np.ceil(fitradius))  # +/- nfitpix
        # Initialize the parameter array
        pars = np.zeros(self.nstars*3,float) # height, xcen, ycen
        pars[0::3] = cat['height']
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
        xydata = []
        xlist = []
        ylist = []
        fxydata = []
        fxlist = []
        fylist = []
        ntotpix = 0
        hpsfnpix = self.psf.npix//2
        for i in range(self.nstars):
            xcen = self.starxcen[i]
            ycen = self.starycen[i]
            # Full PSF region
            fxlo = np.maximum(int(np.round(xcen)-hpsfnpix),0)
            fxhi = np.minimum(int(np.round(xcen)+hpsfnpix),nx-1)
            fylo = np.maximum(int(np.round(ycen)-hpsfnpix),0)
            fyhi = np.minimum(int(np.round(ycen)+hpsfnpix),ny-1)
            fxy = [[fxlo,fxhi],[fylo,fyhi]]
            fx,fy = psf.xylim2xy(fxy)
            frr = np.sqrt( (fx-xcen)**2 + (fy-ycen)**2 )
            fmask = frr<=hpsfnpix
            fx = fx[fmask]  # raveled
            fy = fy[fmask]
            fxydata.append(fxy)
            fxlist.append(fx)
            fylist.append(fy)
            # Fitting region
            xlo = np.maximum(int(np.round(xcen)-self.nfitpix),0)
            xhi = np.minimum(int(np.round(xcen)+self.nfitpix),nx-1)
            ylo = np.maximum(int(np.round(ycen)-self.nfitpix),0)
            yhi = np.minimum(int(np.round(ycen)+self.nfitpix),ny-1)
            xy = [[xlo,xhi],[ylo,yhi]]
            x,y = psf.xylim2xy(xy)
            rr = np.sqrt( (x-xcen)**2 + (y-ycen)**2 )
            mask = rr<=self.fitradius
            x = x[mask]  # raveled
            y = y[mask]
            ntotpix += x.size
            xydata.append(xy)  # this still includes the corners
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
            
        # Create 1D unraveled indices
        ind1 = np.ravel_multi_index((xall,yall),image.shape)
        # Get unique indexes and inverse indices
        #   the inverse index list takes you from the duplicated pixels
        #   to the unique ones
        uind1,invindex = np.unique(ind1,return_inverse=True)
        ntotpix = len(uind1)
        x,y = np.unravel_index(uind1,image.shape)
        
        imflatten = image.data.ravel()[uind1]
        errflatten = image.uncertainty.array.ravel()[uind1]
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
        self.xylim = [[np.min(x),np.max(x)],[np.min(y),np.max(y)]]
        self.xydata = xydata

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
            self.starsky[:] = self.skyim[np.round(self.starxcen).astype(int),np.round(self.starycen).astype(int)]
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
                newmodel[xind,yind] += im1
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
            im[fxind,fyind] += im1
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
        
        x0,x1 = self.xylim[0]
        y0,y1 = self.xylim[1]
        nx = x1-x0+1
        ny = y1-y0+1
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
        
        x0,x1 = self.xylim[0]
        y0,y1 = self.xylim[1]
        nx = x1-x0+1
        ny = y1-y0+1
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
            xy = self.xydata[i]
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
            resid[fxind,fyind] -= fim1
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
            #resid[fxind,fyind] += fmodel1
            
            # crowdsource, does a sum
            #  y = 2 / integral(P*P*W) * integral(x*(I-P)*W)
            #  where x = x/y coordinate, I = isolated stamp, P = PSF model, W = weight

            # Use the derivatives instead
            xind = self.xlist[i]
            yind = self.ylist[i]
            jac1 = jac[count]
            jac1 = np.delete(jac1,0,axis=1)  # delete height column
            resid1 = resid[xind,yind] 

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
        mjac = self.jac(xdata,*self.pars,allparams=True,trim=False,verbose=False)
        hess = mjac.T @ mjac
        # cov = H-1, covariance matrix is inverse of Hessian matrix
        cov_orig = np.linalg.inv(hess)
        # Rescale to get an unbiased estimate
        # cov_scaled = cov * (RSS/(m-n)), where m=number of measurements, n=number of parameters
        # RSS = residual sum of squares
        #  using rss gives values consistent with what curve_fit returns
        bestmodel = self.model(xdata,*self.pars,allparams=True,trim=False,verbose=False)
        resid = self.imflatten-self.skyflatten-bestmodel
        cov = cov_orig * (np.sum(resid**2)/(self.ntotpix-len(self.pars)))
        #chisq = np.sum(resid**2/self.errflatten**2)        
        #cov = cov_orig * (chisq/(self.ntotpix-len(self.pars)))  # what MPFITFUN suggests, but very small
        
        return cov

        
    
def fit(psf,image,cat,method='qr',fitradius=None,maxiter=10,minpercdiff=0.5,reskyiter=2,
        nofreeze=False,verbose=False):
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
    maxiter : int, optional
       Maximum number of iterations to allow.  Only for methods "qr" or "svd".
       Default is 10.
    minpercdiff : float, optional
       Minimum percent change in the parameters to allow until the solution is
       considered converged and the iteration loop is stopped.  Only for methods
       "qr" and "svd".  Default is 0.5.
    reskyiter : int, optional
       After how many iterations to re-calculate the sky background. Default is 2.
    verbose : boolean, optional
       Verbose output.

    Returns
    -------
    out : table
       Table of best-fitting parameters for each star.
       id, height, height_error, x, x_err, y, y_err, sky
    model : numpy array
       Best-fitting model of the stars and sky background.

    Example
    -------

    outcat,model = fit(psf,image,cat)

    """

    start = time.time()
    
    # Check input catalog
    for n in ['height','x','y']:
        if n not in cat.keys():
            raise ValueError('Cat must have height, x, and y columns')

    # Check the method
    method = str(method).lower()    
    if method not in ['cholesky','svd','qr','sparse','htcen','curve_fit']:
        raise ValueError('Only cholesky, svd, qr, sparse, htcen or curve_fit methods currently supported')
        
    nstars = np.array(cat).size
    nx,ny = image.data.shape

    # Start the Group Fitter
    gf = GroupFitter(psf,image,cat,fitradius=fitradius,verbose=verbose)
    xdata = np.arange(gf.ntotpix)

    
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
                                sigma=gf.errflatten,p0=bestpar,jac=gf.jac)
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
                # Solve Jacobian
                dbeta = jac_solve(jac,dy,method=method)
            
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
                print('chisq = ',chisq)

            #print('min/max X: ',np.min(gf.pars[1::3]),np.max(gf.pars[1::3]))
            #print('min/max Y: ',np.min(gf.pars[2::3]),np.max(gf.pars[2::3]))        
            #print(dbeta)
            print('H: ',gf.starheight)
            print('X: ',gf.starxcen)
            print('Y: ',gf.starycen)        
        
            # Re-estimate the sky
            if gf.niter % reskyiter == 0:
                print('Re-estimating the sky')
                gf.sky()
        
            print('iter dt = ',time.time()-start0)

            gf.niter += 1     # increment counter
        
            #import pdb; pdb.set_trace()


    # Check that all starniter are set properly
    #  if we stopped "prematurely" then not all stars were frozen
    #  and didn't have starniter set
    gf.starniter[gf.starniter==0] = gf.niter
    
    # Make final model
    gf.unfreeze()
    model = gf.modelim+gf.skyim
        
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

    print('dt = ',time.time()-start)
    
    return outcat,model
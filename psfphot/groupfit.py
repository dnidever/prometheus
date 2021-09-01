#!/usr/bin/env python

"""GROUPFIT.PY - Fit groups of stars in an image

"""

__authors__ = 'David Nidever <dnidever@montana.edu?'
__version__ = '20210826'  # yyyymmdd


import os
import sys
import numpy as np
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


class GroupFitter(object):

    def __init__(self,psf,image,cat,fitradius=None,verbose=False):
        self.verbose = verbose
        self.psf = psf
        self.image = image
        self.cat = cat
        self.nstars = np.size(cat)
        self.niter = 0
        self.npsfpix = psf.npix
        nx,ny = image.data.shape
        self.nx = nx
        self.ny = ny
        if fitradius is None:
            fitradius = psf.fwhm()
        self.fitradius = fitradius
        self.nfitpix = int(np.ceil(fitradius))  # +/- nfitpix
        # Initialize the parameter array
        pars = np.zeros(self.nstars*3,float) # height, xcen, ycen
        pars[0::3] = cat['height']
        pars[1::3] = cat['x']
        pars[2::3] = cat['y']
        self.pars = pars
        self.starsky = np.zeros(self.nstars,float)
        self.njaciter = 0
        self.freezestars = np.zeros(self.nstars,bool)
        self.freezepars = np.zeros(self.nstars*3,bool)
        self.pixused = None
        
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
    def starheight(self):
        """ Return the best-fit heights for all stars."""
        return self.pars[0::3]

    @property
    def starxcen(self):
        """ Return the best-fit X centers for all stars."""        
        return self.pars[1::3]

    @property
    def starycen(self):
        """ Return the best-fit Y centers for all stars."""        
        return self.pars[2::3]    
        
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
        return self.skyim.ravel()[self.ind1]
        
    @property
    def nfreezepars(self):
        return np.sum(self.freezepars)

    @property
    def nfreepars(self):
        return np.sum(~self.freezepars)

    @property
    def nfreezestars(self):
        return np.sum(self.freezestars)

    @property
    def nfreestars(self):
        return np.sum(~self.freezestars)    
        
    def freeze(self,pars,frzpars):
        """ Freeze stars and parameters"""
        # PARS: best-fit values of free parameters
        # FRZPARS: boolean array of which "free" parameters
        #            should now be frozen

        # Update all the free parameters
        self.pars[~self.freezepars] = pars

        # Update freeze values for "free" parameters
        self.freezepars[~self.freezepars] = frzpars   # stick in the new values for the "free" parameters
        
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
        """ model function."""
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
            allpars[~self.freezepars] = args
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
            dostars = np.arange(self.nstars)[~self.freezestars]
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

    
    def jac(self,x,*args,retmodel=False,trim=False):
        """ jacobian."""

        if self.verbose:
            print('jac: ',self.njaciter,args)

        # Args are [height,xcen,ycen,sky] for all Nstars
        # so 3*Nstars parameters
        
        psf = self.psf

        # Figure out the parameters of ALL the stars
        #  some stars and parameters are FROZEN
        if self.nfreezepars>0:
            allpars = self.pars
            if len(args) != (len(self.pars)-self.nfreezepars):
                print('problem')
                import pdb; pdb.set_trace()
            allpars[~self.freezepars] = args
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
        for i in np.arange(self.nstars)[~self.freezestars]:
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
            
            #xindrel = xind-x0
            #yindrel = yind-y0
            #jac[xindrel,yindrel,i*4] = jac1[:,0]
            #jac[xindrel,yindrel,i*4+1] = jac1[:,1]
            #jac[xindrel,yindrel,i*4+2] = jac1[:,2]
            #jac[xindrel,yindrel,i*4+3] = jac1[:,3]            

        # Remove frozen columns
        if self.nfreezepars>0:
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

    
def fit(psf,image,cat,method='qr',fitradius=None,maxiter=10,minpercdiff=0.5,reskyiter=2,nofreeze=False,
        verbose=False):
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
       Method to use for solving the non-linear least squares problem: "qr",
       "svd", and "curve_fit".  Default is "qr".
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
    
    for n in ['height','x','y']:
        if n not in cat.keys():
            raise ValueError('Cat must have height, x, and y columns')
    
    # jac will be [Npix,Npars]
    # where Npars = 3*nstars for height,xcen,ycen,sky

    # SPARSE MATRIX OPERATIONS!!!!

    nstars = np.array(cat).size
    nx,ny = image.data.shape

    # Start the Group Fitter
    gf = GroupFitter(psf,image,cat,fitradius=fitradius,verbose=verbose)
    xdata = np.arange(gf.ntotpix)

    # Perform the fitting
    #  initial estimates
    initpar = np.zeros(nstars*3,float)
    initpar[0::3] = cat['height']
    initpar[1::3] = cat['x']
    initpar[2::3] = cat['y']

    # Iterate
    count = 0
    maxpercdiff = 1e10
    bestpar = initpar.copy()
    npars = len(bestpar)
    while (count<maxiter and maxpercdiff>minpercdiff):
        start0 = time.time()
        m,jac = gf.jac(xdata,*bestpar,retmodel=True,trim=True)
        dy = gf.resflatten[gf.usepix]-gf.skyflatten[gf.usepix]-m
        # QR decomposition
        if str(method).lower()=='qr':
            q,r = np.linalg.qr(jac)
            rinv = np.linalg.inv(r)
            dbeta = rinv @ (q.T @ dy)
        # SVD:
        elif str(method).lower()=='svd':
            u,s,vt = np.linalg.svd(jac)
            #u,s,vt = sparse.linalg.svds(jac)
            # u: [Npix,Npix]
            # s: [Npars]
            # vt: [Npars,Npars]
            # dy: [Npix]
            sinv = s.copy()*0  # pseudo-inverse
            sinv[s!=0] = 1/s[s!=0]
            npars = len(s)
            dbeta = vt.T @ ((u.T @ dy)[0:npars]*sinv)
        # Curve_fit
        elif str(method).lower()=='curve_fit':
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
            break
        else:
            raise ValueError('Only SVD, QR or curve_fit methods currently supported')
        
        oldpar = bestpar.copy()
        bestpar += dbeta
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
            
        count += 1        
        
        if verbose:
            print(count,bestpar,percdiff,chisq)

        print('min/max X: ',np.min(gf.pars[1::3]),np.max(gf.pars[1::3]))
        print('min/max Y: ',np.min(gf.pars[2::3]),np.max(gf.pars[2::3]))        

        # Re-estimate the sky
        if count % reskyiter == 0:
            print('Re-estimating the sky')
            gf.sky()
        
        print('iter dt = ',time.time()-start0)
            
        #import pdb; pdb.set_trace()

        # Maybe fit height of each star separately using just the central 4-9 pixels?

        # Maybe assume that the initial positions and sky subtraction are pretty good
        # and just solve for heights as crowdsource does?
        # can tweak positions and sky after that

    # Make final model
    gf.unfreeze()
    model = gf.modelim+gf.skyim
        
    # Estimate uncertainties
    if method != 'curve_fit':
        # https://stats.stackexchange.com/questions/93316/parameter-uncertainty-after-non-linear-least-squares-estimation
        # more background here, too: http://ceres-solver.org/nnls_covariance.html
        # Hessian = J.T * T, Hessian Matrix
        jac = gf.jac(xdata,*gf.pars)
        hess = jac.T @ jac
        # cov = H-1, covariance matrix is inverse of Hessian matrix
        cov_orig = np.linalg.inv(hess)
        # Rescale to get an unbiased estimate
        # cov_scaled = cov * (RSS/(m-n)), where m=number of measurements, n=number of parameters
        # RSS = residual sum of squares
        #  using rss gives values consistent with what curve_fit returns
        cov = cov_orig * (np.sum(resid**2)/(gf.ntotpix-len(gf.pars)))
        #cov = cov_orig * (chisq/(gf.ntotpix-len(gf.pars)))  # what MPFITFUN suggests, but very small
        perror = np.sqrt(np.diag(cov))
        
    pars = gf.pars
    if verbose:
        print('Best-fitting parameters: ',pars)
        print('Errors: ',perror)

        
    # Put in catalog
    # Initialize catalog
    dt = np.dtype([('id',int),('height',float),('height_error',float),('x',float),
                   ('x_error',float),('y',float),('y_error',float),('sky',float)])
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

    print('dt = ',time.time()-start)
    
    return outcat,model

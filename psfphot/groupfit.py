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
        self.starheight = np.zeros(self.nstars,float)
        self.starheight[:] = cat['height'].copy()
        self.starxcen = np.zeros(self.nstars,float)
        self.starxcen[:] = cat['x'].copy()
        self.starycen = np.zeros(self.nstars,float)
        self.starycen[:] = cat['y'].copy()
        self.starsky = np.zeros(self.nstars,float)
        self.starsky[:] = cat['sky'].copy()        
        self.njaciter = 0
        self.freezestars = np.zeros(self.nstars,bool)
        #self.nfreezestars = 0
        self.freezepars = np.zeros(self.nstars*4,bool)
        #self.nfreezepars = 0
        self.pars = np.zeros(self.nstars*4,float)
        self.perror = np.zeros(self.nstars*4,float)
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

        # Check if we need to freeze any new parameters
        nfrz = np.sum(frzpars)
        if nfrz==0:
            return
        
        # Update freeze values for "free" parameters
        self.freezepars[~self.freezepars] = frzpars   # stick in the new values for the "free" parameters
        #self.nfreezepars = np.sum(self.freezepars)

        # Freeze new stars
        oldfreezestars = self.freezestars.copy()
        self.freezestars = np.sum(self.freezepars.reshape(self.nstars,4),axis=1)==4
        #self.nfreezestars = np.sum(self.freezestars)
        # Subtract model for newly frozen stars
        newfreezestars, = np.where((oldfreezestars==False) & (self.freezestars==True))
        if len(newfreezestars)>0:
            # add models to a full image
            newmodel = self.image.data.copy()*0
            for i in newfreezestars:
                print('freeze: subtracting model for star ',i)
                pars1 = self.pars[i*4:(i+1)*4]
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
        #self.nfreezestars = 0
        self.freezepars = np.zeros(self.nstars*4,bool)
        #self.nfreezepars = 0
    
    def model(self,x,*args,trim=False):
        """ model function."""

        if self.verbose:
            print('model: ',self.niter,args)

        # Args are [height,xcen,ycen,sky] for all Nstars
        # so 4*Nstars parameters

        psf = self.psf

        # Figure out the parameters of ALL the stars
        #  some stars and parameters are FROZEN
        if self.nfreezepars>0:
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
        for i in np.arange(self.nstars)[~self.freezestars]:
            pars = allpars[i*4:(i+1)*4]
            #height,xcen,ycen,sky = pars
            #xy = self.xydata[i]
            xind = self.xlist[i]
            yind = self.ylist[i]
            invindex = self.invindexlist[i]
            im1 = psf(xind,yind,pars)
            #xindrel = xind-x0
            #yindrel = yind-y0
            #im[xindrel,yindrel] += im1
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
        # so 4*Nstars parameters
        
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
            pars = allpars[i*4:(i+1)*4]
            xy = self.xydata[i]
            xind = self.xlist[i]
            yind = self.ylist[i]
            invindex = self.invindexlist[i]
            xdata = (xind,yind)
            if retmodel:
                m,jac1 = psf.jac(xdata,*pars,retmodel=True)
            else:
                jac1 = psf.jac(xdata,*pars)
            jac[invindex,i*4] = jac1[:,0]
            jac[invindex,i*4+1] = jac1[:,1]
            jac[invindex,i*4+2] = jac1[:,2]
            jac[invindex,i*4+3] = jac1[:,3]
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

    
def fit(psf,image,cat,method='qr',fitradius=None,maxiter=10,minpercdiff=1.0,nofreeze=False,
        nobacksub=False,freezesky=False,verbose=False):
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
       "qr" and "svd".  Default is 1.0.
    freezesky : boolean, optional
       Do not fit the sky parameters for each star.  Default is False.
    nobacksub : boolean, optional
       Do not subtract a background.  Default is False.
    verbose : boolean, optional
       Verbose output.

    Returns
    -------
    pars : numpy array
       Array of best-fit model parameters
    perror : numpy array
       Uncertainties in "pars".
    newpsf : PSF object
       New PSF object with the best-fit model parameters.

    Example
    -------

    pars,perror,newpsf = getpsf(psf,image,cat)

    """

    start = time.time()
    
    for n in ['height','x','y','sky']:
        if n not in cat.keys():
            raise ValueError('Cat must have height, x, y and sky columns')
    
    # jac will be [Npix,Npars]
    # where Npars = 4*nstars for height,xcen,ycen,sky

    # once a star's parameters have stopped varying below some threshold, hold that
    #  star fixed and remove it from the variables/parameters being fit.
    #  basically just subtract the best-fit model from the image and continue with the other stars

    # should I use sparse matrix operations???

    # SPARSE MATRIX OPERATIONS!!!!

    nstars = np.array(cat).size
    
    nx,ny = image.data.shape

    # Local copy of image
    im = image.copy()
    
    # Subtract the background
    if not nobacksub:
        # Get the background using SEP
        bw = np.maximum(int(nx/10),64)
        bh = np.maximum(int(ny/10),64)
        bkg = sep.Background(im.data, bw=bw, bh=bh, fw=3, fh=3)
        bkg_image = bkg.back()
        # Subtract the background
        im.data -= bkg_image
        
    grpfitter = GroupFitter(psf,im,cat,fitradius=fitradius,verbose=verbose)
    xdata = np.arange(grpfitter.ntotpix)

    # Perform the fitting
    initpar = np.zeros(nstars*4,float)
    initpar[0::4] = cat['height']
    initpar[1::4] = cat['x']
    initpar[2::4] = cat['y']
    initpar[3::4] = cat['sky']
    if not nobacksub:
        initpar[3::4] = 0.0

    # Initialize catalog
    dt = np.dtype([('id',int),('height',float),('height_error',float),('x',float),
                   ('x_error',float),('y',float),('y_error',float),('sky',float),('sky_error',float)])
    outcat = np.zeros(nstars,dtype=dt)
    if 'id' in cat.keys():
        outcat['id'] = cat['id']
    else:
        outcat['id'] = np.arange(nstars)+1

    # Freeze the sky parameters to zero
    if freezesky:
        print('Freezing sky parameters to zero')
        initpar[3::4] = 0
        bestpar = initpar.copy()
        frzpars = np.zeros(len(initpar),bool)
        frzpars[3::4] = True
        bestpar = grpfitter.freeze(bestpar,frzpars)
    else:
        bestpar = initpar.copy()
        
    # Iterate
    count = 0
    maxpercdiff = 1e10
    npars = len(bestpar)
    while (count<maxiter and maxpercdiff>minpercdiff):
        start0 = time.time()
        m,jac = grpfitter.jac(xdata,*bestpar,retmodel=True,trim=True)
        dy = grpfitter.resflatten[grpfitter.usepix]-m
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
            bounds = [np.zeros(grpfitter.nstars*4,float)-np.inf,
                      np.zeros(grpfitter.nstars*4,float)+np.inf]
            bounds[0][0::4] = 0
            bounds[0][1::4] = cat['x']-2
            bounds[1][1::4] = cat['x']+2
            bounds[0][2::4] = cat['y']-2
            bounds[1][2::4] = cat['y']+2
            bounds[0][3::4] = -1e-7  # freeze sky to zero
            bounds[1][3::4] = 1e-7            
            bestpar,cov = curve_fit(grpfitter.model,xdata,grpfitter.imflatten,bounds=bounds,
                                     sigma=grpfitter.errflatten,p0=bestpar,jac=grpfitter.jac)
            perror = np.sqrt(np.diag(cov))
            grpfitter.pars = bestpar
            grpfitter.perror = perror
            break
        else:
            raise ValueError('Only SVD, QR or curve_fit methods currently supported')
            
        oldpar = bestpar.copy()
        bestpar += dbeta
        diff = np.abs(bestpar-oldpar)
        percdiff = diff.copy()*0
        percdiff[0::4] = diff[0::4]/oldpar[0::4]*100  # height
        percdiff[1::4] = diff[1::4]*100               # x
        percdiff[2::4] = diff[2::4]*100               # y
        percdiff[3::4] = diff[3::4]/oldpar[3::4]*100  # height
        #percdiff = diff/oldpar*100
        #starmaxpercdiff = np.max(percdiff.reshape(nstars,4),axis=1)

        # Freeze parameters/stars that converged
        #  also subtract models of fixed stars
        #  also return new free parameters
        if not nofreeze:
            frzpars = percdiff<=minpercdiff
            freeparsind, = np.where(~grpfitter.freezepars)
            grpfitter.perror[freeparsind[diff>0]] = diff[diff>0]
            bestpar = grpfitter.freeze(bestpar,frzpars)
            npar = len(bestpar)
            print('Nfrozen pars = ',grpfitter.nfreezepars)
            print('Nfrozen stars = ',grpfitter.nfreezestars)
            print('Nfree pars = ',npar)
        else:
            grpfitter.pars = bestpar
            grpfitter.perror = diff
            
        maxpercdiff = np.max(percdiff)
        #perror = diff  # rough estimate
        count += 1        
        
        if verbose:
            print(count,bestpar,percdiff)

        print('min/max X: ',np.min(grpfitter.pars[1::4]),np.max(grpfitter.pars[1::4]))
        print('min/max Y: ',np.min(grpfitter.pars[2::4]),np.max(grpfitter.pars[2::4]))        
            
        print('iter dt = ',time.time()-start0)
            
        #import pdb; pdb.set_trace()
            

        # MAYBE DON'T SOLVE ALL FOUR (HEIGHT/X/Y/SKY) SIMULTANEOUSLY!?

        # Maybe fit height of each star separately using just the central 4-9 pixels?

        # Maybe assume that the initial positions and sky subtraction are pretty good
        # and just solve for heights as crowdsource does?
        # can tweak positions and sky after that
        
    pars = grpfitter.pars
    perror = grpfitter.perror
    if verbose:
        print('Best-fitting parameters: ',pars)
        print('Errors: ',perror)

    # Make final model
    grpfitter.unfreeze()
    model1 = grpfitter.model(xdata,*pars)
    model = im.data.copy()*0
    model[grpfitter.x,grpfitter.y] = model1
        
    # Put in catalog
    outcat['height'] = pars[0::4]
    outcat['height_error'] = perror[0::4]
    outcat['x'] = pars[1::4]
    outcat['x_error'] = perror[1::4]
    outcat['y'] = pars[2::4]
    outcat['y_error'] = perror[2::4]
    outcat['sky'] = pars[3::4]
    outcat['sky_error'] = perror[3::4]        

    print('dt = ',time.time()-start)
    
    return outcat,model,grpfitter

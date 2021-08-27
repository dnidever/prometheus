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
        
        # Get xdata, ydata
        xydata = []
        xlist = []
        ylist = []
        ntotpix = 0
        for i in range(self.nstars):
            xcen = self.starxcen[i]
            ycen = self.starycen[i]
            xlo = np.maximum(int(np.round(xcen)-self.nfitpix),0)
            xhi = np.minimum(int(np.round(xcen)+self.nfitpix),nx-1)
            ylo = np.maximum(int(np.round(ycen)-self.nfitpix),0)
            yhi = np.minimum(int(np.round(ycen)+self.nfitpix),ny-1)
            xy = [[xlo,xhi],[ylo,yhi]]
            x,y = psf.xylim2xy(xy)
            rr = np.sqrt( (x-xcen)**2 + (y-ycen)**2 )
            mask = rr>self.fitradius
            x = x[mask]  # raveled
            y = y[mask]
            ntotpix += x.size
            xydata.append(xy)  # this still includes the corners
            xlist.append(x)
            ylist.append(y)
            
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

        
    def model(self,x,*args):
        """ model function."""

        if self.verbose:
            print('model: ',self.niter,args)

        # Args are [height,xcen,ycen,sky] for all Nstars
        # so 4*Nstars parameters

        psf = self.psf

        x0,x1 = self.xylim[0]
        y0,y1 = self.xylim[1]
        nx = x1-x0+1
        ny = y1-y0+1
        #im = np.zeros((nx,ny),float)    # image covered by star
        allim = np.zeros(self.ntotpix,float)
        
        # Loop over the stars and generate the model image
        for i in range(self.nstars):
            pars = args[i*4:(i+1)*4]
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
            
        # Pull out the pixels weneed
        #allim2 = im[self.x-x0,self.y-y0]

        self.niter += 1
        
        return allim

    
    def jac(self,x,*args,retmodel=False):
        """ jacobian."""

        if self.verbose:
            print('jac: ',self.niter,args)


        # Args are [height,xcen,ycen,sky] for all Nstars
        # so 4*Nstars parameters

        psf = self.psf

        x0,x1 = self.xylim[0]
        y0,y1 = self.xylim[1]
        nx = x1-x0+1
        ny = y1-y0+1
        #jac = np.zeros((nx,ny,len(args)),float)    # image covered by star
        jac = np.zeros((self.ntotpix,len(args)),float)    # image covered by star        
        if retmodel:
            im = np.zeros(self.ntotpix,float)
        
        # Loop over the stars and generate the derivatives
        for i in range(self.nstars):
            pars = args[i*4:(i+1)*4]
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
            
            #xindrel = xind-x0
            #yindrel = yind-y0
            #jac[xindrel,yindrel,i*4] = jac1[:,0]
            #jac[xindrel,yindrel,i*4+1] = jac1[:,1]
            #jac[xindrel,yindrel,i*4+2] = jac1[:,2]
            #jac[xindrel,yindrel,i*4+3] = jac1[:,3]            
            
            
        # Pull out the pixels weneed
        #alljac = np.zeros((self.ntotpix,len(args)),float)

        if retmodel:
            return im,jac
        else:
            return jac

    
def fit(psf,image,cat,method='qr',maxiter=10,minpercdiff=1.0,verbose=False):
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
    
    # Get the background using SEP
    bkg = sep.Background(image.data, bw=int(nx/10), bh=int(ny/10), fw=3, fh=3)
    bkg_image = bkg.back()
    
    # Subtract the background
    image0 = image.copy()
    image.data -= bkg_image
    
    grpfitter = GroupFitter(psf,image,cat,verbose=verbose)
    xdata = np.arange(grpfitter.ntotpix)

    # Perform the fitting
    initpar = np.zeros(nstars*4,float)
    initpar[0::4] = cat['height']
    initpar[1::4] = cat['x']
    initpar[2::4] = cat['y']
    initpar[3::4] = cat['sky']
    #bestpars,cov = curve_fit(grpfitter.model,xdata,grpfitter.imflatten,
    #                         sigma=grpfitter.errflatten,p0=initpar,jac=grpfitter.jac)
    #
    #import pdb; pdb.set_trace()
    
    # Iterate
    count = 0
    percdiff = 1e10
    bestpar = initpar.copy()
    while (count<maxiter and percdiff>minpercdiff):
        m,jac = grpfitter.jac(xdata,*bestpar,retmodel=True)
        dy = grpfitter.imflatten-m
        # QR decomposition
        if str(method).lower()=='qr':
            q,r = np.linalg.qr(jac)
            rinv = np.linalg.inv(r)
            dbeta = rinv @ (q.T @ dy)
        # SVD:
        elif str(method).lower()=='svd':
            u,s,vt = np.linalg.svd(jac)
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
            bestpars,cov = curve_fit(grpfitter.model,xdata,grpfitter.imflatten,
                                     sigma=grpfitter.errflatten,p0=bestpar,jac=grpfitter.jac)
            perror = np.sqrt(np.diag(cov))
            break
        else:
            raise ValueError('Only SVD or QR methods currently supported')
            
        oldpar = bestpar.copy()
        bestpar += dbeta
        diff = np.abs(bestpar-oldpar)
        percdiff = np.max(diff/oldpar*100)
        perror = diff  # rough estimate
        count += 1

        if verbose:
            print(count,bestpar,percdiff)

    pars = bestpar
    if verbose:
        print('Best-fitting parameters: ',pars)
        print('Errors: ',perror)

        
    #import pdb; pdb.set_trace()

        
    # Put in catalog
    dt = np.dtype([('id',int),('height',float),('height_error',float),('x',float),
                   ('x_error',float),('y',float),('y_error',float),('sky',float),('sky_error',float)])
    outcat = np.zeros(nstars,dtype=dt)
    if 'id' in cat.keys():
        outcat['id'] = cat['id']
    else:
        outcat['id'] = np.arange(nstars)+1
    outcat['height'] = pars[0::4]
    outcat['height_error'] = perror[0::4]
    outcat['x'] = pars[1::4]
    outcat['x_error'] = perror[1::4]
    outcat['y'] = pars[2::4]
    outcat['y_error'] = perror[2::4]
    outcat['sky'] = pars[3::4]
    outcat['sky_error'] = perror[3::4]        

    return outcat

#!/usr/bin/env python

"""ALLFIT.PY - Fit PSF to all stars in an image

"""

__authors__ = 'David Nidever <dnidever@montana.edu?'
__version__ = '20210908'  # yyyymmdd


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
from astropy.nddata import CCDData as CCD,StdDevUncertainty
from dlnpyutils import utils as dln, bindata
import copy
import logging
import time
import matplotlib
import sep
from photutils.aperture import CircularAnnulus
from astropy.stats import sigma_clipped_stats
from . import leastsquares as lsq
from . import groupfit
from .ccddata import CCDData,BoundingBox
from photutils.psf.groupstars import DAOGroup

# Fit a PSF model to all stars in an image

def cutoutbbox(image,psf,cat):
    """ Get image cutout that covers that catalog of stars."""
    nx,ny = image.shape
    hpsfpix = psf.npix//2
    xmin = np.min(cat['x'])
    xmax = np.max(cat['x'])
    ymin = np.min(cat['y'])
    ymax = np.max(cat['y'])    

    xlo = np.maximum(int(np.round(xmin)-hpsfpix),0)
    xhi = np.minimum(int(np.round(xmax)+hpsfpix),nx)
    ylo = np.maximum(int(np.round(ymin)-hpsfpix),0)
    yhi = np.minimum(int(np.round(ymax)+hpsfpix),ny)

    return BoundingBox(xlo,xhi,ylo,yhi)
    
    
def fit(psf,image,cat,method='qr',fitradius=None,maxiter=10,minpercdiff=0.5,reskyiter=2,
        nofreeze=False,verbose=False):
    """
    Fit PSF to all stars in an image.

    To pre-group the stars, add a "group_id" in the input catalog.

    Parameters
    ----------
    psf : PSF object
       PSF object with initial parameters to use.
    image : CCDData object
       Image to use to fit PSF model to stars.
    cat : table
       Catalog with initial height/x/y values for the stars to use to fit the PSF.
       To pre-group the stars, add a "group_id" in the catalog.
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

    outcat,model = fit(psf,image,cat,groups)

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

    # Groups
    if 'group_id' not in cat.keys():
        daogroup = DAOGroup(crit_separation=2.5*psf.fwhm())
        starlist = cat.copy()
        starlist['x_0'] = cat['x']
        starlist['y_0'] = cat['y']        
        star_groups = daogroup(starlist)
        cat['group_id'] = star_groups['group_id']

    # Star index
    starindex = dln.create_index(np.array(cat['group_id']))        
    groups = starindex['value']
    ngroups = len(groups)
    if verbose:
        print(str(ngroups)+' star groups')

        
    # Initialize catalog
    dt = np.dtype([('id',int),('height',float),('height_error',float),('x',float),
                   ('x_error',float),('y',float),('y_error',float),('sky',float),
                   ('niter',int),('group_id',int),('ngroup',int)])
    outcat = np.zeros(nstars,dtype=dt)
    if 'id' in cat.keys():
        outcat['id'] = cat['id']
    else:
        outcat['id'] = np.arange(nstars)+1
        
    # Group Loop
    #---------------
    resid = image.copy()
    outmodel = CCDData(np.zeros(image.shape),bbox=image.bbox,unit=image.unit)
    outsky = CCDData(np.zeros(image.shape),bbox=image.bbox,unit=image.unit)    
    for g,grp in enumerate(groups):
        ind = starindex['index'][starindex['lo'][g]:starindex['hi'][g]+1]
        nind = len(ind)
        cat1 = cat[ind]
        if verbose:
            print('Group '+str(grp)+': '+str(nind)+' stars')
        
        # Single Star
        if nind==1:
            out,model = psf.fit(resid,cat1,niter=3,verbose=verbose,retfullmodel=True)
            model.data -= out['sky']   # remove sky
            outmodel[model.bbox.slices].data += model.data
            outsky[model.bbox.slices].data = out['sky'] 
            
        # Group
        else:
            bbox = cutoutbbox(image,psf,cat1)
            out,model,sky = groupfit.fit(psf,resid[bbox.slices],cat1,method=method,fitradius=fitradius,
                                         maxiter=maxiter,minpercdiff=minpercdiff,reskyiter=reskyiter,
                                         nofreeze=nofreeze,verbose=verbose,absolute=True)
            outmodel[model.bbox.slices].data += model.data
            outsky[model.bbox.slices].data = sky
            

        # Subtract the best model for the group/star
        resid[model.bbox.slices].data -= model.data
                
        # Put in catalog
        cols = ['height','height_error','x','x_error','y','y_error','sky','niter']        
        for c in cols:
            outcat[c][ind] = out[c]
        outcat['group_id'] = grp
        outcat['ngroup'] = nind
        
    if verbose:
        print('dt = ',time.time()-start)
    
    return outcat,outmodel,outsky
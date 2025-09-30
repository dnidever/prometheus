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
#from astropy.nddata import CCDData as CCD,StdDevUncertainty
from dlnpyutils import utils as dln, bindata
import copy
import logging
import time
import matplotlib
import sep
from astropy.stats import sigma_clipped_stats
from . import leastsquares as lsq
from . import groupfit,utils
from .ccddata import CCDData,BoundingBox
from .groupstars import DAOGroup
#from photutils.psf.groupstars import DAOGroup

# Fit a PSF model to all stars in an image

def cutoutbbox(image,psf,cat):
    """ Get image cutout that covers that catalog of stars."""
    ny,nx = image.shape
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
    
    
def fit(psf,image,cat,method='qr',fitradius=None,recenter=True,maxiter=10,minpercdiff=0.5,
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
    cat : table
       Catalog with initial amp/x/y values for the stars to use to fit the PSF.
       To pre-group the stars, add a "group_id" in the catalog.
    method : str, optional
       Method to use for solving the non-linear least squares problem: "cholesky",
       "qr", "svd", and "curve_fit".  Default is "cholesky".
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
    out : table
       Table of best-fitting parameters for each star.
       id, amp, amp_error, x, x_err, y, y_err, sky
    model : numpy array
       Best-fitting model of the stars and sky background.

    Example
    -------

    outcat,model = fit(psf,image,cat,groups)

    """

    print = utils.getprintfunc() # Get print function to be used locally, allows for easy logging       
    start = time.time()
    
    # Check input catalog
    for n in ['x','y']:
        if n not in cat.keys():
            raise ValueError('Cat must have x and y columns')

    # Check the method
    method = str(method).lower()    
    if method not in ['cholesky','svd','qr','sparse','htcen','curve_fit']:
        raise ValueError('Only cholesky, svd, qr, sparse, htcen or curve_fit methods currently supported')
        
    nstars = np.array(cat).size
    ny,nx = image.data.shape
    
    # Groups
    if 'group_id' not in cat.keys():
        daogroup = DAOGroup(crit_separation=2.5*psf.fwhm())
        starlist = cat.copy()
        starlist['x_0'] = cat['x']
        starlist['y_0'] = cat['y']
        # THIS TAKES ~4 SECONDS!!!!!! WAY TOO LONG!!!!
        star_groups = daogroup(starlist)
        cat['group_id'] = star_groups['group_id']

    # Star index
    starindex = dln.create_index(np.array(cat['group_id']))        
    groups = starindex['value']
    ngroups = len(groups)
    if verbose:
        print(str(ngroups)+' star groups')
        
    # Initialize catalog
    dt = np.dtype([('id',int),('amp',float),('amp_error',float),('x',float),
                   ('x_error',float),('y',float),('y_error',float),('sky',float),
                   ('flux',float),('flux_error',float),('mag',float),('mag_error',float),
                   ('niter',int),('group_id',int),('ngroup',int),('rms',float),('chisq',float)])
    outcat = np.zeros(nstars,dtype=dt)
    outcat = Table(outcat)
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
        inpcat = cat[ind].copy()
        if 'amp' not in inpcat.columns:
            # Estimate amp from flux and fwhm
            # area under 2D Gaussian is 2*pi*A*sigx*sigy
            if 'fwhm' in inpcat.columns:
                amp = inpcat['flux']/(2*np.pi*(inpcat['fwhm']/2.35)**2)
            else:
                amp = inpcat['flux']/(2*np.pi*(psf.fwhm()/2.35)**2)                
            staramp = np.maximum(amp,0)   # make sure it's positive
            inpcat['amp'] = staramp
        
        if verbose:
            print('-- Group '+str(grp)+'/'+str(len(groups))+' : '+str(nind)+' star(s) --')
        
        # Single Star
        if nind==1:
            inpcat = [inpcat['amp'][0],inpcat['x'][0],inpcat['y'][0]]
            out,model = psf.fit(resid,inpcat,niter=3,verbose=verbose,retfullmodel=True,recenter=recenter)
            model.data -= out['sky']   # remove sky
            outmodel.data[model.bbox.slices] += model.data
            outsky.data[model.bbox.slices] = out['sky']

        # Group
        else:
            bbox = cutoutbbox(image,psf,inpcat)
            out,model,sky = groupfit.fit(psf,resid[bbox.slices],inpcat,method=method,fitradius=fitradius,
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
            outcat[c][ind] = out[c]
        outcat['group_id'][ind] = grp
        outcat['ngroup'][ind] = nind
        outcat = Table(outcat)
        
    if verbose:
        print('dt = %.2f sec' % (time.time()-start))
    
    return outcat,outmodel,outsky

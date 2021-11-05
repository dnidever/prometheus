#!/usr/bin/env python

"""UTILS.PY - Some PSF utility routines

"""

__authors__ = 'David Nidever <dnidever@montana.edu?'
__version__ = '20210915'  # yyyymmdd


import os
import sys
import numpy as np
import warnings
from astropy.io import fits
from astropy.table import Table
import logging
import time
from scipy.spatial import cKDTree
from . import detection, models, getpsf, allfit
from .ccddata import CCDData

try:
    import __builtin__ as builtins # Python 2
except ImportError:
    import builtins # Python 3
        
# Ignore these warnings, it's a bug
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

def getprintfunc(inplogger=None):
    """ Allows you to modify print() locally with a logger."""
    
    # Input logger
    if inplogger is not None:
        return inplogger.info  
    # Check if a global logger is defined
    elif hasattr(builtins,"logger"):
        return builtins.logger.info
    # Return the buildin print function
    else:
        return builtins.print

def splitfilename(filename):
    """ Split filename into directory, base and extensions."""
    fdir = os.path.dirname(filename)
    base = os.path.basename(filename)
    exten = ['.fit','.fits','.fit.gz','.fits.gz','.fit.fz','.fits.fz']
    for e in exten:
        if base[-len(e):]==e:
            base = base[0:-len(e)]
            ext = e
            break
    return (fdir,base,ext)

def estimatefwhm(objects,verbose=False):
    """ Estimate FWHM using objects."""

    print = getprintfunc() # Get print function to be used locally, allows for easy logging   
    
    # Check that we have all of the columns that we need
    for f in ['mag_auto','magerr_auto','flags','fwhm']:
        if f not in objects.colnames:
            raise ValueError('objects catalog must have mag_auto, magerr_auto, flags and fwhm columns')
    
    # Select good sources
    gdobjects = ((objects['mag_auto']< 50) & (objects['magerr_auto']<0.05) &
                 (objects['flags']==0))
    ngdobjects = np.sum(gdobjects)
    # Not enough good source, remove FLAGS cut
    if (ngdobjects<10):
        gdobjects = ((objects['mag_auto']< 50) & (objects['magerr_auto']<0.05))
        ngdobjects = np.sum(gdobjects)
    # Not enough sources, lower thresholds
    if (ngdobjects<10):
        gdobjects = ((objects['mag_auto']< 50) & (objects['magerr_auto']<0.08))
        ngdobjects = np.sum(gdobjects)            
    medfwhm = np.median(objects[gdobjects]['fwhm'])
    if verbose:
        print('FWHM = %5.2f pixels (%d sources)' % (medfwhm, ngdobjects))

    return medfwhm

def neighbors(objects,nnei=1):
    """ Find the closest neighbors to a star."""

    # Returns distance and index of closest neighbor
    
    # Use KD-tree
    X = np.vstack((objects['x'].data,objects['y'].data)).T
    kdt = cKDTree(X)
    # Get distance for 2 closest neighbors
    dist, ind = kdt.query(X, k=nnei+1)
    # closest neighbor is always itself, remove it
    dist = dist[:,1:]
    ind = ind[:,1:]
    magdiff = objects['mag_auto'][ind[:,0]]-objects['mag_auto']  # negative means neighbor is brighter
    if nnei==1:
        dist = dist.flatten()
        ind = ind.flatten()
    return dist,ind,magdiff
    
def pickpsfstars(objects,fwhm,nstars=100,logger=None,verbose=False):
    """ Pick PSF stars."""

    print = getprintfunc() # Get print function to be used locally, allows for easy logging   
    
    # -morph cuts
    # -magnitude limit (good S/N but not too bright due to saturation)
    # -no bad pixels in footprint
    # -no close neighbors

    # Use KD-tree to figure out closest neighbors
    neidist,neiind,neimagdiff = neighbors(objects)
    
    # Select good sources
    gdobjects1 = ((objects['mag_auto']< 50) & (objects['magerr_auto']<0.10))
    ngdobjects1 = np.sum(gdobjects1)
    # Bright and faint limit, use 5th and 95th percentile
    minmag, maxmag = np.sort(objects[gdobjects1]['mag_auto'])[[int(np.round(0.05*ngdobjects1)),int(np.round(0.95*ngdobjects1))]]
    # Select stars with
    # -good FWHM values
    # -good clas_star values (unless FWHM too large)
    # -good mag range, bright but not too bright
    # -no flags set
    gdobjects = ((objects['mag_auto']< 50) & (objects['magerr_auto']<0.1) & 
                 (objects['fwhm']>0.5*fwhm) & (objects['fwhm']<1.5*fwhm) &
                 (objects['mag_auto']>(minmag+1.0)) & (objects['mag_auto']<(maxmag-0.5)) &
                 (objects['flags']==0) & (neidist>15.0) & (neimagdiff>1.0))
    ngdobjects = np.sum(gdobjects)
    # No candidate, loosen cuts
    if ngdobjects<10:
        if verbose:
            print("Too few PSF stars on first try. Loosening cuts")
        gdobjects = ((objects['mag_auto']< 50) & (objects['magerr_auto']<0.15) & 
                     (objects['fwhm']>0.2*fwhm) & (objects['fwhm']<1.8*fwhm) &
                     (objects['mag_auto']>(minmag+0.5)) & (objects['mag_auto']<(maxmag-0.5)) &
                     (neidist>10) & (neimagdiff>1.0))
        ngdobjects = np.sum(gdobjects)
    # No candidate, loosen cuts again
    if ngdobjects<10:
        if verbose:
            print("Too few PSF stars on second try. Loosening cuts")
        gdobjects = ((objects['mag_auto']< 50) & (objects['magerr_auto']<0.15) & 
                     (objects['fwhm']>0.2*fwhm) & (objects['fwhm']<1.8*fwhm) &
                     (objects['mag_auto']>(minmag+0.5)) & (objects['mag_auto']<(maxmag-0.5)))
        ngdobjects = np.sum(gdobjects)
    # No candidates
    if ngdobjects==0:
        raise Exception('No good PSF stars found')
    
    # Candidate PSF stars, use only Nstars, and sort by magnitude
    si = np.argsort(objects[gdobjects]['mag_auto'])
    psfobjects = objects[gdobjects][si]
    if ngdobjects>nstars: psfobjects=psfobjects[0:nstars]
    if verbose:
        print(str(len(psfobjects))+" PSF stars found")
    
    return psfobjects

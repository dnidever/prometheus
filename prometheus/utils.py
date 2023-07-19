#!/usr/bin/env python

"""UTILS.PY - Some PSF utility routines

"""

__authors__ = 'David Nidever <dnidever@montana.edu?'
__version__ = '20210915'  # yyyymmdd


import os
import sys
import mmap
import numpy as np
import warnings
from astropy.io import fits
from astropy.table import Table
import logging
import time
from datetime import datetime
from scipy.spatial import cKDTree
from dlnpyutils import utils as dln,ladfit
from . import detection, models, getpsf, allfit, leastsquares as lsq
from .ccddata import CCDData
from . import __version__ as version

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


def poly2d(xdata,*pars):
    """ model of 2D linear polynomial."""
    x = xdata[0]
    y = xdata[1]
    return pars[0]+pars[1]*x+pars[2]*y+pars[3]*x*y

def jacpoly2d(xdata,*pars):
    """ jacobian of 2D linear polynomial."""
    x = xdata[0]
    y = xdata[1]
    nx = len(x)
    # Model
    m = pars[0]+pars[1]*x+pars[2]*y+pars[3]*x*y
    # Jacobian, partical derivatives wrt the parameters
    jac = np.zeros((nx,4),float)
    jac[:,0] = 1    # constant coefficient
    jac[:,1] = x    # x-coefficient
    jac[:,2] = y    # y-coefficient
    jac[:,3] = x*y  # xy-coefficient
    return m,jac
    
def poly2dfit(x,y,data,maxiter=2):
    """ Fit a 2D linear function to data robustly."""
    gd, = np.where(np.isfinite(data))
    xdata = [x[gd],y[gd]]
    initpars = np.zeros(4,float)
    med = np.median(data[gd])
    sig = dln.mad(data[gd])
    gd, = np.where( (np.abs(data-med)<3*sig) & np.isfinite(data))
    initpars[0] = med
    xdata = [x[gd],y[gd]]
    # Do the fit
    pars,perror,cov = lsq.lsq_solve(xdata,data[gd],jacpoly2d,initpars,maxiter=maxiter)
    return pars,perror

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
    # Not enough sources, lower thresholds
    if (ngdobjects<10):
        si = np.argsort(objects['magerr_auto'])
        halferr = objects['magerr_auto'][si[len(objects)//2]]
        gdobjects = ((objects['mag_auto']< 50) & (objects['magerr_auto']<halferr))
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
    si = np.argsort(objects['magerr_auto'])
    halferr = objects['magerr_auto'][si[len(objects)//2]]            
    gdobjects1 = ((objects['mag_auto']< 50) & (objects['magerr_auto']<halferr))
    ngdobjects1 = np.sum(gdobjects1)
    # Bright and faint limit, use 5th and 95th percentile
    minmag,maxmag = np.nanpercentile(objects['mag_auto'][gdobjects1],(5,95))
    # Select stars with
    # -good FWHM values
    # -good clas_star values (unless FWHM too large)
    # -good mag range, bright but not too bright
    # -no flags set
    gdobjects = ((objects['mag_auto']<50) & (objects['magerr_auto']<0.05) & 
                 (objects['fwhm']>0.5*fwhm) & (objects['fwhm']<1.5*fwhm) &
                 (objects['mag_auto']>(minmag+1.0)) & (objects['mag_auto']<(maxmag-0.5)) &
                 (objects['flags']==0) & (neidist>15.0) & (neimagdiff>1.0))
    ngdobjects = np.sum(gdobjects)
    # No candidate, loosen cuts
    if ngdobjects<50:
        if verbose:
            print("Too few PSF stars on first try. Loosening cuts")
        gdobjects = ((objects['mag_auto']<50) & (objects['magerr_auto']<0.10) & 
                     (objects['fwhm']>0.2*fwhm) & (objects['fwhm']<1.8*fwhm) &
                     (objects['mag_auto']>(minmag+0.5)) & (objects['mag_auto']<(maxmag-0.5)) &
                     (neidist>10) & (neimagdiff>1.0))
        ngdobjects = np.sum(gdobjects)
    # No candidate, loosen cuts again
    if ngdobjects<50:
        if verbose:
            print("Too few PSF stars on second try. Loosening cuts")
        gdobjects = ((objects['mag_auto']<50) & (objects['magerr_auto']<0.15) & 
                     (objects['fwhm']>0.2*fwhm) & (objects['fwhm']<1.8*fwhm) &
                     (objects['mag_auto']>(minmag+0.5)) & (objects['mag_auto']<(maxmag-0.5)))
        ngdobjects = np.sum(gdobjects)
    # No candidate, loosen cuts again
    if ngdobjects<50:
        if verbose:
            print("Too few PSF stars on second try. Loosening cuts")
        gdobjects = ((objects['mag_auto']<50) & (objects['magerr_auto']<halferr) & 
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

def refresh_mmap(hdulist):
    """
    Close and refresh an hdulist's memory map to free up virtual memory
    """

    if hasattr(hdulist._file,'_mmap')==False:
        return
    
    MEMMAP_MODES = {'readonly': mmap.ACCESS_COPY,
                'copyonwrite': mmap.ACCESS_COPY,
                'update': mmap.ACCESS_WRITE,
                'append': mmap.ACCESS_COPY,
                'denywrite': mmap.ACCESS_READ}
    access_mode = MEMMAP_MODES[hdulist._file.mode]
    hdulist._file._mmap = mmap.mmap(hdulist._file._file.fileno(), 0,
                                    access=access_mode,
                                    offset=0)
    # Loop over hdus and delete the data in memory
    for h in hdulist:
        if hasattr(h,'data'):
            del h.data

def saveoutput(filename,outfile,out,model,sky,psf):
    """
    Save Prometheus output to a file.

    Parameters
    ----------
    filename : str
       Filename of the original image file.
    outfile : str
       Filename for the output.
    cat : table
       The output table of best-fit PSF values for all of the sources.
    model : CCDData object
       The best-fitting model for the stars (without sky).
    sky : CCDData object
       The background sky image used for the image.
    psf : PSF object
       The best-fitting PSF model.
    
    Returns
    -------
    The output is saved to a FITS file.

    Example
    -------

    saveoutput(filename,outfile,out,model,sky,psf)

    """
    
    if os.path.exists(outfile): os.remove(outfile)
    hdulist = fits.HDUList()
    hdulist.append(fits.table_to_hdu(out))  # table    
    hdulist[0].header['COMMENT']='Prometheus version '+str(version)
    hdulist[0].header['COMMENT']='Date '+datetime.now().ctime()
    hdulist[0].header['COMMENT']='File '+filename
    hdulist[0].header['COMMENT']='HDU#0 : Header Only'
    hdulist[0].header['COMMENT']='HDU#1 : Source catalog'
    hdulist[0].header['COMMENT']='HDU#2 : Model image'
    hdulist[0].header['COMMENT']='HDU#3 : Sky model image'
    hdulist[0].header['COMMENT']='HDU#4 : PSF model'
    hdulist[1].header['COMMENT'] = 'Prometheus source catalog'
    hdulist.append(model.tohdu())  # model
    hdulist[2].header['COMMENT'] = 'Prometheus model image'
    hdulist.append(sky.tohdu())    # sky
    hdulist[3].header['COMMENT'] = 'Prometheus sky image'
    psfhdu = psf.tohdu()
    # psf, could be 2 HDUs
    if isinstance(psfhdu,fits.HDUList):
        for h in psfhdu:
            hdulist.append(h)
        hdulist[4].header['COMMENT'] = 'Prometheus PSF model'
        hdulist[5].header['COMMENT'] = 'Prometheus PSF model lookup table'        
    else:
        hdulist.append(psfhdu)     # psf
        hdulist[4].header['COMMENT'] = 'Prometheus PSF model'
    hdulist.writeto(outfile,overwrite=True)
    hdulist.close()
    

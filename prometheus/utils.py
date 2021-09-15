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
from . import detection, models, getpsf, allfit
from .ccddata import CCDData


def estimatefwhm(objects):
    """ Estimate FWHM using objects."""

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
    print('FWHM = %5.2f pixels (%d sources)' % (medfwhm, ngdobjects))

    return medfwhm

    
def pickpsfstars(objects,fwhm,image=None,nstars=100,logger=None):
    """ Pick PSF stars."""

    # -morph cuts
    # -magnitude limit (good S/N but not too bright due to saturation)
    # -no bad pixels in footprint
    # -no close neighbors

    # Use KD-tree to figure out closest neighbors

    # Check image for bad pixels in footprint
    

    # Select good sources
    gdobjects1 = ((objects['mag_auto']< 50) & (objects['magerr_auto']<0.05))
    ngdobjects1 = np.sum(gdobjects1)
    # Bright and faint limit, use 5th and 95th percentile
    minmag, maxmag = np.sort(objects[gdobjects1]['mag_auto'])[[int(np.round(0.05*ngdobjects1)),int(np.round(0.95*ngdobjects1))]]
    # Select stars with
    # -good FWHM values
    # -good clas_star values (unless FWHM too large)
    # -good mag range, bright but not too bright
    # -no flags set
    if fwhm<1.8:
        gdobjects = ((objects['mag_auto']< 50) & (objects['magerr_auto']<0.1) & 
                 (objects['fwhm']>0.5*fwhm) & (objects['fwhm']<1.5*fwhm) &
                 (objects['mag_auto']>(minmag+1.0)) & (objects['mag_auto']<(maxmag-0.5)) &
                 (objects['flags']==0))
        ngdobjects = np.sum(gdobjects)
    # Do not use CLASS_STAR if seeing bad, not as reliable
    else:
        gdobjects = ((objects['mag_auto']< 50) & (objects['magerr_auto']<0.1) & 
                 (objects['fwhm']>0.5*fwhm) & (objects['fwhm']<1.5*fwhm) &
                 (objects['mag_auto']>(minmag+1.0)) & (objects['mag_auto']<(maxmag-0.5)) &
                 (objects['flags']==0))
        ngdobjects = np.sum(gdobjects)
    # No candidate, loosen cuts
    if ngdobjects<10:
        print("Too few PSF stars on first try. Loosening cuts")
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
    print(str(len(psfobjects))+" PSF stars found")
    
    return psfobjects

#!/usr/bin/env python

"""APERTURE.PY - Aperture photometry

"""

__authors__ = 'David Nidever <dnidever@montana.edu?'
__version__ = '20210912'  # yyyymmdd


import os
import sys
import numpy as np
import warnings
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
from scipy.optimize import curve_fit, least_squares
from scipy.interpolate import interp1d
from dlnpyutils import utils as dln, bindata
import copy
import logging
import time
import matplotlib
from .ccddata import BoundingBox,CCDData
from matplotlib.patches import Ellipse
import sep


def aperphot(image,objects,aper=[3],gain=None,mag_zeropoint=25.0):
    """ Aperture photometry."""

    if isinstance(image,CCDData) is False:
        raise ValueError("Image must be a CCDData object")

    sky = None
    if image.data.flags['C_CONTIGUOUS']==False:
        data = image.data.copy(order='C')
        mask = image.mask.copy(order='C')
        err = image.uncertainty.array.copy(order='C')
        if hasattr(image,'sky'): sky = image.sky.copy(order='C')
    else:
        data = image.data
        mask = image.mask
        err = image.uncertainty.array
        if hasattr(image,'sky'): sky = image.sky
    # Get background if needed        
    if sky is None:
        bkg = sep.Background(data, mask=mask, bw=64, bh=64, fw=3, fh=3)
        sky = bkg.back()
        image.sky = sky
    data_sub = data-sky
    
    # Get gain from image if possible
    if gain is None:
        headgain = image.header.get('gain')
        if headgain is not None:
            gain = headgain
        else:
            gain = 1.0

    # Initialize the output catalog
    outcat = objects.copy()
    
    # Circular aperture photometry
    for i,ap in enumerate(aper):
        apflux, apfluxerr, apflag = sep.sum_circle(data_sub, outcat['x'], outcat['y'],
                                                   ap, err=err, mask=mask, gain=gain)
        # Add to the catalog
        outcat['flux_aper'+str(i+1)] = apflux
        outcat['fluxerr_aper'+str(i+1)] = apfluxerr
        outcat['mag_aper'+str(i+1)] = -2.5*np.log10(apflux)+mag_zeropoint
        outcat['magerr_aper'+str(i+1)] = (2.5/np.log(10))*(apfluxerr/apflux)  
        outcat['flag_aper'+str(i+1)] = apflag
    
    # FLUX_AUTO
    kronrad, krflag = sep.kron_radius(data_sub, outcat['x'], outcat['y'], outcat['a'],
                                      outcat['b'], outcat['theta'], 6.0, mask=mask)

    # Add more columns
    outcat['flux_auto'] = 0.0
    outcat['fluxerr_auto'] = 0.0
    outcat['mag_auto'] = 0.0
    outcat['magerr_auto'] = 0.0
    outcat['kronrad'] = kronrad
    outcat['flag_auto'] = np.int16(0)

    # BACKGROUND ANNULUS???
    
    # Only use elliptical aperture if Kron radius is large enough
    # Use circular aperture photometry if the Kron radius is too small
    r_min = 1.75  # minimum diameter = 3.5
    use_circle = kronrad * np.sqrt(outcat['a'] * outcat['b']) < r_min
    nuse_ellipse = np.sum(~use_circle)
    nuse_circle = np.sum(use_circle)
    # Elliptical aperture
    if nuse_ellipse>0:
        flux, fluxerr, flag = sep.sum_ellipse(data_sub, outcat['x'][~use_circle], outcat['y'][~use_circle],
                                              outcat['a'][~use_circle],outcat['b'][~use_circle],
                                              outcat['theta'][~use_circle], 2.5*kronrad[~use_circle],
                                              subpix=1, err=err, mask=mask)
        flag |= krflag[~use_circle]  # combine flags into 'flag'
        outcat['flux_auto'][~use_circle] = flux
        outcat['fluxerr_auto'][~use_circle] = fluxerr
        outcat['mag_auto'][~use_circle] = -2.5*np.log10(flux)+mag_zeropoint
        outcat['magerr_auto'][~use_circle] = (2.5/np.log(10))*(fluxerr/flux) 
        outcat['flag_auto'][~use_circle] = flag
        
    # Use circular aperture photometry if the Kron radius is too small
    if nuse_circle>0:
        cflux, cfluxerr, cflag = sep.sum_circle(data_sub, outcat['x'][use_circle],
                                                outcat['y'][use_circle], r_min, subpix=1,
                                                err=err, mask=mask)
        outcat['flux_auto'][use_circle] = cflux
        outcat['fluxerr_auto'][use_circle] = cfluxerr
        outcat['mag_auto'][use_circle] = -2.5*np.log10(cflux)+mag_zeropoint
        outcat['magerr_auto'][use_circle] = (2.5/np.log(10))*(cfluxerr/cflux) 
        outcat['flag_auto'][use_circle] = cflag
        outcat['kronrad'][use_circle] = r_min
    
    return outcat

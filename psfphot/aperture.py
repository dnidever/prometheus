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


def aperphot(image,bkg,objects,aper=[3],gain=None):
    """ Aperture photometry."""

    if isinstance(image,CCDData) is False:
        raise ValueError("Image must be a CCDData object")
    
    # Get gain from image if possible
    if gain is None:
        headgain = image.header.get('gain')
        if headgain is not None:
            gain = headgain
        else:
            gain = 1.0
        
    # Background subtraction with SEP
    sky = bkg.back()
    data_sub = image.data-sky
    flux, fluxerr, flag = sep.sum_circle(data_sub, objects['x'], objects['y'],
                                         aper, err=image.uncertainty.array, gain=gain)

    # FLUX_AUTO
    kronrad, krflag = sep.kron_radius(data, x, y, a, b, theta, 6.0)
    flux, fluxerr, flag = sep.sum_ellipse(data, x, y, a, b, theta, 2.5*kronrad,
                                          subpix=1)
    flag |= krflag  # combine flags into 'flag'

    # Use circular aperture photometry if the Kron radius is too small
    r_min = 1.75  # minimum diameter = 3.5
    use_circle = kronrad * np.sqrt(a * b) < r_min
    cflux, cfluxerr, cflag = sep.sum_circle(data, x[use_circle], y[use_circle],
                                            r_min, subpix=1)
    flux[use_circle] = cflux
    fluxerr[use_circle] = cfluxerr
    flag[use_circle] = cflag
    
    
    import pdb; pdb.set_trace()
    


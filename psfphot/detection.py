#!/usr/bin/env python

"""DETECTION.PY - Detection algorithms

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
    
# A bunch of the Gaussian2D and Moffat2D code comes from astropy's modeling module
# https://docs.astropy.org/en/stable/_modules/astropy/modeling/functional_models.html

# Maybe x0/y0 should NOT be part of the parameters, and
# x/y should actually just be dx/dy (relative to x0/y0)

def plotobj(image,objects):
    """ Plot objects on top of image."""



    # plot background-subtracted image
    fig, ax = plt.subplots()
    m, s = np.mean(image.data), np.std(image.data)
    im = ax.imshow(image.data, interpolation='nearest', cmap='gray',
                   vmin=m-s, vmax=m+s, origin='lower')

    # plot an ellipse for each object
    for i in range(len(objects)):
        e = Ellipse(xy=(objects['x'][i], objects['y'][i]),
                    width=6*objects['a'][i],
                    height=6*objects['b'][i],
                    angle=objects['theta'][i] * 180. / np.pi)
        e.set_facecolor('none')
        e.set_edgecolor('red')
        ax.add_artist(e)
    

def detection(image,nsigma=1.5):
    """ Detection algorithm """

    if isinstance(image,CCDData) is False:
        raise ValueError("Image must be a CCDData object")
    
    # Background subtraction with SEP

    # measure a spatially varying background on the image
    bkg = sep.Background(image.data, mask=image.mask, bw=64, bh=64, fw=3, fh=3)
    sky = bkg.back()
    data_sub = image.data-sky


    # matched filter
    #  by default a 3x3 kernel is used
    #  to turn off use filter_kernel=None
    #  for optimal detection the kernel size should be approximately the PSF
    kernel = np.array([[1., 2., 3., 2., 1.],
                       [2., 3., 5., 3., 2.],
                       [3., 5., 8., 5., 3.],
                       [2., 3., 5., 3., 2.],
                       [1., 2., 3., 2., 1.]])
    #objects = sep.extract(data, thresh, filter_kernel=kernel)
    
    # Detection with SEP
    objects = sep.extract(data_sub, nsigma, err=image.uncertainty.array)
    nobj = len(objects)

    
    import pdb; pdb.set_trace()
    


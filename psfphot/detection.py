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
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import sep
from photutils.detection import DAOStarFinder,IRAFStarFinder
    
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
        if 'a' in objects.colnames:        
            e = Ellipse(xy=(objects['x'][i], objects['y'][i]),
                        width=6*objects['a'][i],
                        height=6*objects['b'][i],
                        angle=objects['theta'][i] * 180. / np.pi)
        elif 'fwhm' in objects.colnames:
            e = Ellipse(xy=(objects['x'][i], objects['y'][i]),
                        width=3*objects['fwhm'][i],
                        height=3*objects['fwhm'][i],
                        angle=0.0)
        else:
            e = Ellipse(xy=(objects['x'][i], objects['y'][i]),
                        width=10.0, height=10.0, angle=0.0)
        e.set_facecolor('none')
        e.set_edgecolor('red')
        ax.add_artist(e)
        

def detection(image,method='sep',nsigma=1.5,fwhm=3,minarea=3,deblend_nthresh=32,deblend_cont=0.000015):
    """ Detection algorithm """

    if isinstance(image,CCDData) is False:
        raise ValueError("Image must be a CCDData object")
    
    # Background subtraction with SEP

    # measure a spatially varying background on the image
    if image.data.flags['C_CONTIGUOUS']==False:
        data = image.data.copy(order='C')
        mask = image.mask.copy(order='C')
        err = image.uncertainty.array.copy(order='C')
    else:
        data = image.data
        mask = image.mask
        err = image.uncertainty.array
    bkg = sep.Background(data, mask=mask, bw=64, bh=64, fw=3, fh=3)
    sky = bkg.back()
    image.sky = sky
    data_sub = data-sky

    # Detection method
    method = str(method).lower()
    
    # SEP
    if method=='sep':
    
        # matched filter
        #  by default a 3x3 kernel is used
        #  to turn off use filter_kernel=None
        #  for optimal detection the kernel size should be approximately the PSF
        if fwhm is not None:
            npix = np.round(1.6*fwhm)
            if npix % 2 == 0: npix += 1
            npix = int(npix)
            x = np.arange(npix).astype(float)-npix//2
            kernel = np.exp(-0.5*( x.reshape(-1,1)**2 + x.reshape(1,-2)**2 )/(fwhm/2.35)**2)
            #kernel = np.array([[1., 2., 3., 2., 1.],
            #                   [2., 3., 5., 3., 2.],
            #                   [3., 5., 8., 5., 3.],
            #                   [2., 3., 5., 3., 2.],
            #                   [1., 2., 3., 2., 1.]])
            #objects = sep.extract(data, thresh, filter_kernel=kernel)
            #kernel = np.array([[1,2,1], [2,4,2], [1,2,1]])  # default 3x3 kernel
        else:
            kernel = None
        
        # Detection with SEP
        objects,segmap = sep.extract(data_sub, nsigma, filter_kernel=kernel,minarea=minarea,clean=False,
                                     mask=mask, err=err, deblend_nthresh=deblend_nthresh,
                                     deblend_cont=deblend_cont, segmentation_map=True)
        nobj = len(objects)
        objects = Table(objects)
        objects['id'] = np.arange(nobj)+1
        objects['fwhm'] = np.sqrt(objects['a']**2+objects['b']**2)*2.35
        return objects,segmap


    # DAOFinder
    elif method=='dao':
        threshold = np.median(err)*nsigma
        daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold, sky=0.0)  
        objects = daofind(data_sub, mask=mask)
        # homogenize the columns
        objects['xcentroid'].name = 'x'
        objects['ycentroid'].name = 'y'        
        return objects
        
    # IRAFFinder
    elif method=='iraf':
        threshold = np.median(err)*nsigma        
        iraffind = IRAFStarFinder(fwhm=fwhm, threshold=threshold, sky=0.0)
        objects = iraffind(data_sub, mask=mask)
        # homogenize the columns
        objects['xcentroid'].name = 'x'
        objects['ycentroid'].name = 'y'        
        return objects
        
    else:
        raise ValueError('Only sep, dao or iraf methods supported')
        
    
    import pdb; pdb.set_trace()
    


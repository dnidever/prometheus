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
from skimage.feature import peak_local_max
    
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

def sepdetect(image,nsigma=1.5,fwhm=3.0,minarea=3,deblend_nthresh=32,
              deblend_cont=0.000015,kernel=None,maskthresh=0.0,
              segmentation_map=False):
    """ 
    Detection with sep.

    Parameters
    ----------
    image : CCDData object
       Image on which to run sep for detection.
    nsigma : float, optional
       Detection level in units of sigma.  Default is 1.5.
    fwhm : float, optional
       Initial guess fwhm to use.  Default is 3.0.
    minarea : int, optional
       Minimum number of pixels for a detection.  Default is 3.
    deblend_ntresh : int, optional
       Deblending number of thresholds.  Default is 32.
    deblend_cont : float, optional
       Deblending contrast.  Default is 0.000015.
    kernel : numpy array, optional
       Matched filter smoothing kernel.  Default is to construct
         it using the fwhm value.
    maskthresh : float, optional
       Mask threshold.  Default is 0.0.
    segmentation_map : bool, optional
       Return segmentation map.  Default is False.

    Returns
    -------
    objects : table
       Table of the detected objects and properties.
    segmap : numpy array
       Segmentation map if segmentation_map=True was input.

    Example
    -------

    objects = sepdetect(image)

    """
    
    # matched filter
    #  by default a 3x3 kernel is used
    #  to turn off use filter_kernel=None
    #  for optimal detection the kernel size should be approximately the PSF
    if fwhm is not None and kernel is None:
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

            
    # Detection with SEP
    #  NOTE! filter_type='matched' for some reason causes ploblems when 2D error array is input

    # Get C-continuous data
    data,error,mask,sky = image.ccont
    out = sep.extract(data-sky, nsigma, filter_kernel=kernel,minarea=minarea,
                      clean=False,mask=mask, err=error,
                      maskthresh=maskthresh,deblend_nthresh=deblend_nthresh,
                      deblend_cont=deblend_cont,filter_type='conv',
                      segmentation_map=segmentation_map)
    if segmentation_map:
        objects,segmap = out
    else:
        objects = out
    nobj = len(objects)
    objects = Table(objects)
    objects['id'] = np.arange(nobj)+1
    objects['fwhm'] = np.sqrt(objects['a']*objects['b'])*2.35
    objects['flag'].name = 'flags'

    # Make sure theta's are between -90 and +90 deg (in radians)
    hi = objects['theta']>0.5*np.pi
    if np.sum(hi)>0:
        objects['theta'][hi] -= np.pi
    lo = objects['theta']<-0.5*np.pi    
    if np.sum(lo)>0:
        objects['theta'][lo] += np.pi
        
    if segmentation_map:
        return objects,segmap
    else:
        return objects
    
def daodetect(image,nsigma=1.5,fwhm=3.0):
    """ 
    Detection with photutils's DAOFinder.

    Parameters
    ----------
    image : CCDData object
       Image on which to run sep for detection.
    nsigma : float, optional
       Detection level in units of sigma.  Default is 1.5.
    fwhm : float, optional
       Initial guess fwhm to use.  Default is 3.0.
 
    Returns
    -------
    objects : table
       Table of the detected objects and properties.

    Example
    -------

    objects = daodetect(image)

    """

    threshold = np.median(image.error)*nsigma
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold, sky=0.0)  
    objects = daofind(image.data-image.sky, mask=image.mask)
    # homogenize the columns
    objects['xcentroid'].name = 'x'
    objects['ycentroid'].name = 'y'
    
    return objects
    
def irafdetect(image,nsigma=1.5,fwhm=3.0):
    """
    Detection with photutil's IRAFFinder.

    Parameters
    ----------
    image : CCDData object
       Image on which to run sep for detection.
    nsigma : float, optional
       Detection level in units of sigma.  Default is 1.5.
    fwhm : float, optional
       Initial guess fwhm to use.  Default is 3.0.
 
    Returns
    -------
    objects : table
       Table of the detected objects and properties.

    Example
    -------

    objects = irafdetect(image)

    """
    threshold = np.median(image.error)*nsigma        
    iraffind = IRAFStarFinder(fwhm=fwhm, threshold=threshold, sky=0.0)
    objects = iraffind(image.data-image.sky, mask=image.mask)
    # homogenize the columns
    objects['xcentroid'].name = 'x'
    objects['ycentroid'].name = 'y'

    return objects

def peaks(image,nsigma=1.5,thresh=None):
    """
    Detect peaks in an image.

    Parameters
    ----------
    image : CCDData object
       Image on which to run sep for detection.
    nsigma : float, optional
       Detection level in units of sigma.  Default is 1.5.
    thresh : float, optional
       Absolute threshold for detection.
 
    Returns
    -------
    objects : table
       Table of the detected peaks.

    Example
    -------

    objects = peaks(image)

    """

    # Comparison between image_max and im to find the coordinates of local maxima
    data = image.data-image.sky
    err = image.error
    coordinates = peak_local_max(data, threshold_abs=thresh, min_distance=3)
    xind = coordinates[:,1]
    yind = coordinates[:,0]    
    
    # Check that they are above the error limit
    nthresh = data[yind,xind]/(nsigma*err[yind,xind])
    if image.mask is None:
        gd, = np.where(nthresh >= 1.0)
    else:
        gd, = np.where((nthresh >= 1.0) & (image.mask[yind,xind]==False))
    nobj = len(gd)
    dtype = np.dtype([('id',int),('x',float),('y',float),('nthresh',float)])
    objects = np.zeros(nobj,dtype=dtype)
    objects['id'] = np.arange(nobj)+1
    objects['x'] = xind[gd]
    objects['y'] = yind[gd]
    objects['nthresh'] = nthresh[gd]
    return objects
    
def detect(image,method='sep',nsigma=1.5,fwhm=3.0,minarea=3,deblend_nthresh=32,
           deblend_cont=0.000015,kernel=None,maskthresh=0.0,verbose=False):
    """
    Detection algorithm

    Parameters
    ----------
    image : CCDData object
       The image to detect sources in.
    method : str, optional
       Method to use.  Options are sep, dao, and iraf.  Default is sep.
    nsigma : float, optional
       Detection threshold in number of sigma.  Default is 1.5.
    fwhm : float, optional
       Estimate for PSF full width at half maximum.  Default is 3.0.
    minarea : int, optional
       Minimum area requirement for an object (sep only).  Default is 3.
    deblend_nthresh : int, optional
       Number of deblending thresholds (sep only).  Default is 32.
    deblend_cont : float, optional
       Minimum contrast ratio used for object deblending (sep only).
         Default is 0.000015.  To entirely disable deblending, set to 1.0.
    kernel : numpy array, optional
        Filter kernel used for on-the-fly filtering (used to
        enhance detection). Default is a 3x3 array.
    maskthresh : float, optional
       Threshold for a pixel to be masked (sep only). Default is 0.0.
    verbose : boolean, optional
       Verbose output to the screen.

    Returns
    -------
    objects : astropy Table
       Table of objects with centroids.
    segmap : numpy array
       Segmentation (sep only).

    Example
    -------

    obj = detect(image,nsigma=1.5,fwhm=4.0)

    """

    if isinstance(image,CCDData) is False:
        raise ValueError("Image must be a CCDData object")

    # Detection method
    method = str(method).lower()

    if verbose:
        print('Detection method = '+method)
        print('Nsigma = %5.2f' % nsigma)
        
    # SEP
    if method=='sep':
        return sepdetect(image,nsigma=nsigma,minarea=minarea,fwhm=fwhm,
                         maskthresh=maskthresh,deblend_nthresh=deblend_nthresh,
                         deblend_cont=deblend_cont,kernel=kernel)


    # DAOFinder
    elif method=='dao':
        return daodetect(image,nsigma=nsigma,fwhm=fwhm)
        return objects
        
    # IRAFFinder
    elif method=='iraf':
        return irafdetect(image,nsigma=nsigma,fwhm=fwhm)
        return objects
        
    else:
        raise ValueError('Only sep, dao or iraf methods supported')

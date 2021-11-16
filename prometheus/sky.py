#!/usr/bin/env python

"""SKY.PY - Sky estimation algorithms

"""

__authors__ = 'David Nidever <dnidever@montana.edu?'
__version__ = '20210913'  # yyyymmdd

import numpy as np
import sep
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground, MADStdBackgroundRMS

def sepsky(image,box_size=(64,64),filter_size=(3,3)):
    """ Estimate sky background using sep."""

    # Check if the data is "sep ready"

    data = image.sepready(image.data)
    if image.mask is not None:
        mask = image.sepready(image.mask)
    else:
        mask = None
        
    #image.native()  # make sure the arrays use native byte-order
    
    ## Background subtraction with SEP
    ##  measure a spatially varying background on the image
    #if image.data.flags['C_CONTIGUOUS']==False:
    #    data = image.data.copy(order='C')
    #    mask = image.mask.copy(order='C')
    #else:
    #    data = image.data
    #    mask = image.mask
    bkg = sep.Background(data, mask=mask, bw=box_size[0], bh=box_size[1], fw=filter_size[0], fh=filter_size[1])
    return bkg.back()

def photutilsky(image, box_size=(50,50), filter_size=(3,3)):
    """ Estimate sky background using photutils."""

    sigma_clip = SigmaClip(sigma=3.)
    bkg_estimator = MedianBackground()
    bkg = Background2D(image.data, box_size, mask=image.mask, filter_size=filter_size,
                       sigma_clip=sigma_clip)
    return bkg.background

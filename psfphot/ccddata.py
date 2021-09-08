#!/usr/bin/env python

"""CCDDATA.PY - Thin wrapper around CCDData class

"""

__authors__ = 'David Nidever <dnidever@montana.edu?'
__version__ = '20210908'  # yyyymmdd


import numpy as np
from astropy.nddata import CCDData as CCD,StdDevUncertainty

class CCData(CCD):

    def __init__(self,image):
        nx,ny = image.shape
        self._bbox = [[0,nx],[0,ny]]
        self._x = np.arange(x)
        self._y = np.arange(y)
        
    @property
    def bbox(self):
        """ Boundary box."""
        return self._bbox

    @property
    def x(self):
        """ X-array."""
        return self._x

    @property
    def y(self):
        """ Y-array."""
        return self._y 

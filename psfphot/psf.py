#!/usr/bin/env python

"""PSF.PY - PSF photometry models

"""

__authors__ = 'David Nidever <dnidever@montana.edu?'
__version__ = '20210430'  # yyyymmdd


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


# PSF classes

# PSF base class
class PSFBase:

    def __init__(self,pars,binned=False):
        self.pars = pars
        self.binned = binned
        
    def evaluate(self):
        """ Evaluate the function.  Must be defined by subclass."""
        pass

    def derivative(self):
        """ Return the derivate of the function.  Must be defined by subclass."""
        pass
        
    def copy(self):
        """ Create a new copy of this LSF object."""
        return copy.deepcopy(self)        

# PSF Gaussian class
class PSFGaussian(PSFBase):

    # Initalize the object
    def __init__(self,pars,binned=False):
        super().__init__(pars,binned=binned)

    def evaluate():

        
        
# PSF Moffat class
class PSFMoffat(PSFBase):

        # Initalize the object
    def __init__(self,pars,binned=False):
        super().__init__(pars,binned=binned)


    def evaluate(x, y, pars):
        """Two dimensional Moffat model function"""
        # pars = [amplitude, x0, y0, gamma, alpha]
        rr_gg = ((x - pars[1]) ** 2 + (y - pars[2]) ** 2) / pars[3] ** 2
        return pars[0] * (1 + rr_gg) ** (-pars[4])

    def fit_deriv(x, y, pars):
        """Two dimensional Moffat model derivative with respect to parameters"""

        rr_gg = ((x - pars[1]) ** 2 + (y - pars[2]) ** 2) / pars[3] ** 2
        d_A = (1 + rr_gg) ** (-pars[4])
        d_x_0 = (2 * pars[0] * pars[4] * d_A * (x - pars[1]) /
                 (pars[3] ** 2 * (1 + rr_gg)))
        d_y_0 = (2 * pars[0] * pars[4] * d_A * (y - pars[2]) /
                 (pars[3] ** 2 * (1 + rr_gg)))
        d_alpha = -pars[0] * d_A * np.log(1 + rr_gg)
        d_gamma = (2 * pars[0] * pars[4] * d_A * rr_gg /
                   (pars[3] * (1 + rr_gg)))
        return [d_A, d_x_0, d_y_0, d_gamma, d_alpha]
    
# PSF Penny class
class PSFPenny(PSFBase):
    pass


class PSFEmpirical(PSFBase):
    """ Empirical look-up table PSF, can vary spatially."""
    pass

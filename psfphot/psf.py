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


def gaussian2d(x,y,pars,deriv=False,nderiv=None):
    """Two dimensional Gaussian model function"""
    # pars = [amplitude, x0, y0, xsigma, ysigma, theta]
    theta = np.deg2rad(pars[5])
    cost2 = np.cos(theta) ** 2
    sint2 = np.sin(theta) ** 2
    sin2t = np.sin(2. * theta)
    xstd2 = pars[3] ** 2
    ystd2 = pars[4] ** 2
    xdiff = x - pars[1]
    ydiff = y - pars[2]
    a = 0.5 * ((cost2 / xstd2) + (sint2 / ystd2))
    b = 0.5 * ((sin2t / xstd2) - (sin2t / ystd2))
    c = 0.5 * ((sint2 / xstd2) + (cost2 / ystd2))
    g = pars[0] * np.exp(-((a * xdiff ** 2) + (b * xdiff * ydiff) +
                           (c * ydiff ** 2)))

    # Compute derivative as well
    if deriv is True:

        # How many derivative terms to return
        if nderiv is not None:
            if nderiv <=0:
                nderiv = 6
        else:
            nderiv = 6
        
        derivative = []
        if nderiv>=1:
            dg_dA = g / pars[0]
            derivative.append(dg_dA)
        if nderiv>=2:        
            dg_dx_mean = g * ((2. * a * xdiff) + (b * ydiff))
            derivative.append(dg_dx_mean)
        if nderiv>=3:
            dg_dy_mean = g * ((b * xdiff) + (2. * c * ydiff))
            derivative.append(dg_dy_mean)
        if nderiv>=4:
            cost = np.cos(theta)
            sint = np.sin(theta)
            xstd3 = pars[1] ** 3
            da_dx_stddev = -cost2 / xstd3
            db_dx_stddev = -sin2t / xstd3
            dc_dx_stddev = -sint2 / xstd3        
            dg_dx_stddev = g * (-(da_dx_stddev * xdiff2 +
                                  db_dx_stddev * xdiff * ydiff +
                                  dc_dx_stddev * ydiff2))
            derivative.append(dg_dx_stddev)
        if nderiv>=5:
            ystd3 = pars[2] ** 3            
            da_dy_stddev = -sint2 / ystd3
            db_dy_stddev = sin2t / ystd3
            dc_dy_stddev = -cost2 / ystd3        
            dg_dy_stddev = g * (-(da_dy_stddev * xdiff2 +
                                  db_dy_stddev * xdiff * ydiff +
                                  dc_dy_stddev * ydiff2))
            derivative.append(dg_dy_stddev)
        if nderiv>=6:
            cos2t = np.cos(2. * theta)            
            da_dtheta = (sint * cost * ((1. / ystd2) - (1. / xstd2)))
            db_dtheta = (cos2t / xstd2) - (cos2t / ystd2)
            dc_dtheta = -da_dtheta        
            dg_dtheta = g * (-(da_dtheta * xdiff2 +
                               db_dtheta * xdiff * ydiff +
                               dc_dtheta * ydiff2))
            derivative.append(dg_dtheta)

        return g,derivative
            
    # No derivative
    else:        
        return g

    
def gaussian2d_deriv(x, y, pars, nderiv=None):
    """Two dimensional Gaussian model derivative with respect to parameters"""

    # How many derivative terms to return    
    if nderiv is not None:
        if nderiv <=0:
            nderiv = 6
    else:
        nderiv = 6
            
    theta = np.deg2rad(pars[5])
    cost2 = cost ** 2
    sint2 = sint ** 2
    sin2t = np.sin(2. * theta)
    xstd2 = pars[1] ** 2
    ystd2 = pars[2] ** 2
    xdiff = x - pars[1]
    ydiff = y - pars[2]
    xdiff2 = xdiff ** 2        
    ydiff2 = ydiff ** 2
    a = 0.5 * ((cost2 / xstd2) + (sint2 / ystd2))
    b = 0.5 * ((sin2t / xstd2) - (sin2t / ystd2))
    c = 0.5 * ((sint2 / xstd2) + (cost2 / ystd2))
    g = pars[0] * np.exp(-((a * xdiff2) + (b * xdiff * ydiff) +
                           (c * ydiff2)))
    derivative = []
    if nderiv>=1:
        dg_dA = g / pars[0]
        derivative.append(dg_dA)
    if nderiv>=2:        
        dg_dx_mean = g * ((2. * a * xdiff) + (b * ydiff))
        derivative.append(dg_dx_mean)
    if nderiv>=3:
        dg_dy_mean = g * ((b * xdiff) + (2. * c * ydiff))
        derivative.append(dg_dy_mean)
    if nderiv>=4:
        cost = np.cos(theta)
        sint = np.sin(theta)
        xstd3 = pars[1] ** 3        
        da_dx_stddev = -cost2 / xstd3
        db_dx_stddev = -sin2t / xstd3
        dc_dx_stddev = -sint2 / xstd3        
        dg_dx_stddev = g * (-(da_dx_stddev * xdiff2 +
                              db_dx_stddev * xdiff * ydiff +
                              dc_dx_stddev * ydiff2))
        derivative.append(dg_dx_stddev)
    if nderiv>=5:
        ystd3 = pars[2] ** 3        
        da_dy_stddev = -sint2 / ystd3
        db_dy_stddev = sin2t / ystd3
        dc_dy_stddev = -cost2 / ystd3        
        dg_dy_stddev = g * (-(da_dy_stddev * xdiff2 +
                              db_dy_stddev * xdiff * ydiff +
                              dc_dy_stddev * ydiff2))
        derivative.append(dg_dy_stddev)
    if nderiv>=6:
        cos2t = np.cos(2. * theta)        
        da_dtheta = (sint * cost * ((1. / ystd2) - (1. / xstd2)))
        db_dtheta = (cos2t / xstd2) - (cos2t / ystd2)
        dc_dtheta = -da_dtheta        
        dg_dtheta = g * (-(da_dtheta * xdiff2 +
                           db_dtheta * xdiff * ydiff +
                           dc_dtheta * ydiff2))
        derivative.append(dg_dtheta)

    return derivative

def gaussian2d_integrate(x, y, pars, deriv=False, nderiv=None):
    """ Two dimensional Gaussian model function integrated over the pixels."""

    # Use Error function
    
    shape = x.shape
    
    # pars = [amplitude, x0, y0, xsigma, ysigma, theta]
    theta = np.deg2rad(pars[5])
    cost2 = np.cos(theta) ** 2
    sint2 = np.sin(theta) ** 2
    sin2t = np.sin(2. * theta)
    xstd2 = pars[3] ** 2
    ystd2 = pars[4] ** 2
    xdiff = x - pars[1]
    ydiff = y - pars[2]
    a = 0.5 * ((cost2 / xstd2) + (sint2 / ystd2))
    b = 0.5 * ((sin2t / xstd2) - (sin2t / ystd2))
    c = 0.5 * ((sint2 / xstd2) + (cost2 / ystd2))
    g = pars[0] * np.exp(-((a * xdiff ** 2) + (b * xdiff * ydiff) +
                           (c * ydiff ** 2)))

    # Compute derivative as well
    if deriv is True:

        # How many derivative terms to return
        if nderiv is not None:
            if nderiv <=0:
                nderiv = 6
        else:
            nderiv = 6
        
        derivative = []
        if nderiv>=1:
            dg_dA = g / pars[0]
            derivative.append(dg_dA)
        if nderiv>=2:        
            dg_dx_mean = g * ((2. * a * xdiff) + (b * ydiff))
            derivative.append(dg_dx_mean)
        if nderiv>=3:
            dg_dy_mean = g * ((b * xdiff) + (2. * c * ydiff))
            derivative.append(dg_dy_mean)
        if nderiv>=4:
            cost = np.cos(theta)
            sint = np.sin(theta)
            xstd3 = pars[1] ** 3
            da_dx_stddev = -cost2 / xstd3
            db_dx_stddev = -sin2t / xstd3
            dc_dx_stddev = -sint2 / xstd3        
            dg_dx_stddev = g * (-(da_dx_stddev * xdiff2 +
                                  db_dx_stddev * xdiff * ydiff +
                                  dc_dx_stddev * ydiff2))
            derivative.append(dg_dx_stddev)
        if nderiv>=5:
            ystd3 = pars[2] ** 3            
            da_dy_stddev = -sint2 / ystd3
            db_dy_stddev = sin2t / ystd3
            dc_dy_stddev = -cost2 / ystd3        
            dg_dy_stddev = g * (-(da_dy_stddev * xdiff2 +
                                  db_dy_stddev * xdiff * ydiff +
                                  dc_dy_stddev * ydiff2))
            derivative.append(dg_dy_stddev)
        if nderiv>=6:
            cos2t = np.cos(2. * theta)            
            da_dtheta = (sint * cost * ((1. / ystd2) - (1. / xstd2)))
            db_dtheta = (cos2t / xstd2) - (cos2t / ystd2)
            dc_dtheta = -da_dtheta        
            dg_dtheta = g * (-(da_dtheta * xdiff2 +
                               db_dtheta * xdiff * ydiff +
                               dc_dtheta * ydiff2))
            derivative.append(dg_dtheta)

        return g,derivative
            
    # No derivative
    else:        
        return g

    

def moffat2d(x, y, pars, deriv=False, nderiv=None):
    """Two dimensional Moffat model function"""
    # pars = [amplitude, x0, y0, sigma, beta]
    rr_gg = ((x - pars[1]) ** 2 + (y - pars[2]) ** 2) / pars[3] ** 2
    g = pars[0] * (1 + rr_gg) ** (-pars[4])

    # Compute derivative as well
    if deriv is True:

        # How many derivative terms to return
        if nderiv is not None:
            if nderiv <=0:
                nderiv = 5
        else:
            nderiv = 5
        
        derivative = []
        if nderiv>=1:
            d_A = g/pars[0]
            derivative.append(d_A)
        if nderiv>=2:
            d_x_0 = (2 * pars[0] * pars[4] * d_A * (x - pars[1]) /
                     (pars[3] ** 2 * (1 + rr_gg)))
            derivative.append(d_x_0)            
        if nderiv>=3:
            d_y_0 = (2 * pars[0] * pars[4] * d_A * (y - pars[2]) /
                     (pars[3] ** 2 * (1 + rr_gg)))
            derivative.append(d_y_0)            
        if nderiv>=4:
            d_sigma = (2 * pars[0] * pars[4] * d_A * rr_gg /
                       (pars[3] * (1 + rr_gg)))
            derivative.append(d_sigma)
        if nderiv>=5:            
            d_beta = -pars[0] * d_A * np.log(1 + rr_gg)
            derivative.append(d_beta)            

        return g,derivative

    # No derivative
    else:
        return g
        

def moffat2d_deriv(x, y, pars, nderiv=None):
    """Two dimensional Moffat model derivative with respect to parameters"""
    # pars = [amplitude, x0, y0, sigma, beta]
            
    # How many derivative terms to return
    if nderiv is not None:
        if nderiv <=0:
            nderiv = 5
    else:
        nderiv = 5
        
    rr_gg = ((x - pars[1]) ** 2 + (y - pars[2]) ** 2) / pars[3] ** 2

    derivative = []
    if nderiv>=1:
        d_A = (1 + rr_gg) ** (-pars[4])
        derivative.append(d_A)
    if nderiv>=2:
        d_x_0 = (2 * pars[0] * pars[4] * d_A * (x - pars[1]) /
                 (pars[3] ** 2 * (1 + rr_gg)))
        derivative.append(d_x_0)            
    if nderiv>=3:
        d_y_0 = (2 * pars[0] * pars[4] * d_A * (y - pars[2]) /
                 (pars[3] ** 2 * (1 + rr_gg)))
        derivative.append(d_y_0)            
    if nderiv>=4:
        d_sigma = (2 * pars[0] * pars[4] * d_A * rr_gg /
                   (pars[3] * (1 + rr_gg)))
        derivative.append(d_sigma)
    if nderiv>=5:            
        d_beta = -pars[0] * d_A * np.log(1 + rr_gg)
        derivative.append(d_beta)            

    return derivative


def moffat2d_integrate(x, y, pars, deriv=False, nderiv=None, osamp=4):
    """Two dimensional Moffat model function"""
    # pars = [amplitude, x0, y0, sigma, beta]

    osamp2 = float(osamp)**2
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    nx = x.size
    xshape = x.shape
    # They must be 1D
    if x.ndim>1:
        x = x.flatten()
        y = y.flatten()
    dx = (np.arange(osamp).astype(float)+1)/osamp-(1/(2*osamp))-0.5
    dx2 = np.tile(dx,(osamp,1))
    x2 = np.tile(x,(osamp,osamp,1)) + np.tile(dx2.T,(nx,1,1)).T
    y2 = np.tile(y,(osamp,osamp,1)) + np.tile(dx2,(nx,1,1)).T
        
    rr_gg = ((x2 - pars[1]) ** 2 + (y2 - pars[2]) ** 2) / pars[3] ** 2
    g = pars[0] * (1 + rr_gg) ** (-pars[4])


    # Compute derivative as well
    if deriv is True:

        # How many derivative terms to return
        if nderiv is not None:
            if nderiv <=0:
                nderiv = 5
        else:
            nderiv = 5
        
        derivative = []
        if nderiv>=1:
            d_A = g/pars[0]
            derivative.append(np.sum(np.sum(d_A,axis=0),axis=0)/osamp2)
        if nderiv>=2:
            d_x_0 = (2 * pars[0] * pars[4] * d_A * (x2 - pars[1]) /
                     (pars[3] ** 2 * (1 + rr_gg)))
            derivative.append(np.sum(np.sum(d_x_0,axis=0),axis=0)/osamp2)        
        if nderiv>=3:
            d_y_0 = (2 * pars[0] * pars[4] * d_A * (y2 - pars[2]) /
                     (pars[3] ** 2 * (1 + rr_gg)))
            derivative.append(np.sum(np.sum(d_y_0,axis=0),axis=0)/osamp2)  
        if nderiv>=4:
            d_sigma = (2 * pars[0] * pars[4] * d_A * rr_gg /
                       (pars[3] * (1 + rr_gg)))
            derivative.append(np.sum(np.sum(d_sigma,axis=0),axis=0)/osamp2)
        if nderiv>=5:            
            d_beta = -pars[0] * d_A * np.log(1 + rr_gg)
            derivative.append(np.sum(np.sum(d_beta,axis=0),axis=0)/osamp2)  

        g = np.sum(np.sum(g,axis=0),axis=0)/osamp2           
        return g,derivative

    # No derivative
    else:
        g = np.sum(np.sum(g,axis=0),axis=0)/osamp2
        return g

    
# PSF classes


# PSF base class
class PSFBase:

    def __init__(self,pars,binned=False):
        self._params = np.atleast_1d(pars)
        self.binned = binned

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self,value):
        self._params = value
        
    def __call__(self,x,y,*args,**kwargs):
        return self.evaluate(x,y,*args,**kwargs)

    def __str__(self):
        return self.__class__.__name__+'('+str(list(self.params))+',binned='+str(self.binned)+')'

    def __repr__(self):
        return self.__class__.__name__+'('+str(list(self.params))+',binned='+str(self.binned)+')'        
    
    def evaluate(self):
        """ Evaluate the function.  Must be defined by subclass."""
        pass

    def deriv(self):
        """ Return the derivate of the function.  Must be defined by subclass."""
        pass
        
    def copy(self):
        """ Create a new copy of this LSF object."""
        return copy.deepcopy(self)        

    
# PSF Gaussian class
class PSFGaussian(PSFBase):

    # Initalize the object
    def __init__(self,pars=None,binned=False):
        if pars is None:
            pars = np.array([1.0,0.0,0.0,1.0,1.0,0.0])
        if len(pars)<6:
            raise ValueError('5 parameters required')
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta]
        if pars[3]<=0 or pars[4]<=0:
            raise ValueError('sigma parameters must be >0')
        super().__init__(pars,binned=binned)
        
    def evaluate(self,x, y, pars=None, deriv=False, nderiv=None):
        """Two dimensional Gaussian model function"""
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta]
        if pars is None:
            pars = self.params
        return gaussian2d(x, y, pars, deriv=deriv, nderiv=nderiv)
    
    def deriv(self,x, y, pars=None, nderiv=None):
        """Two dimensional Gaussian model derivative with respect to parameters"""
        if pars is None:
            pars = self.params
        return gaussian2d_deriv(x, y, pars, nderiv=nderiv)
        
        
# PSF Moffat class
class PSFMoffat(PSFBase):

    # add separate X/Y sigma values and cross term like in DAOPHOT
    
    
    # Initalize the object
    def __init__(self,pars=None,binned=False):
        if pars is None:
            pars = np.zeros(5,float)
            pars = np.array([1.0,0.0,0.0,1.0,2.5])
        if len(pars)<5:
            raise ValueError('5 parameters required')
        # pars = [amplitude, x0, y0, sigma, beta]
        if pars[3]<=0:
            raise ValueError('sigma must be >0')
        if pars[4]<1 or pars[4]>6:
            raise ValueError('alpha must be >1 and <6')
        super().__init__(pars,binned=binned)
        
    def evaluate(self,x, y, pars=None, deriv=False, nderiv=None):
        """Two dimensional Moffat model function"""
        # pars = [amplitude, x0, y0, sigma, beta]
        if pars is None:
            pars = self.params
        return moffat2d(x, y, pars, deriv=deriv, nderiv=nderiv)

    def deriv(self,x, y, pars=None, nderiv=None):
        """Two dimensional Moffat model derivative with respect to parameters"""
        if pars is None:
            pars = self.params
        return moffat2d_deriv(x, y, pars, nderiv=nderiv)

# PSF Lorentz class
class PSFLorentz(PSFBase):
    pass
    
    
# PSF Penny class
class PSFPenny(PSFBase):
    """ Gaussian core and Lorentzian wings, only Gaussian is tilted."""
    pass


class PSFEmpirical(PSFBase):
    """ Empirical look-up table PSF, can vary spatially."""
    pass

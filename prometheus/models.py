#!/usr/bin/env python

"""MODELS.PY - PSF photometry models

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
from . import getpsf
from .ccddata import BoundingBox,CCDData
from . import leastsquares as lsq

# A bunch of the Gaussian2D and Moffat2D code comes from astropy's modeling module
# https://docs.astropy.org/en/stable/_modules/astropy/modeling/functional_models.html

# Maybe x0/y0 should NOT be part of the parameters, and
# x/y should actually just be dx/dy (relative to x0/y0)


def gaussian2d(x,y,pars,deriv=False,nderiv=None):
    """Two dimensional Gaussian model function"""
    # pars = [amplitude, x0, y0, xsigma, ysigma, theta]

    xdiff = x - pars[1]
    ydiff = y - pars[2]
    amp = pars[0]
    xsig = pars[3]
    ysig = pars[4]
    theta = pars[5]
    cost2 = np.cos(theta) ** 2
    sint2 = np.sin(theta) ** 2
    sin2t = np.sin(2. * theta)
    xsig2 = xsig ** 2
    ysig2 = ysig ** 2
    a = ((cost2 / xsig2) + (sint2 / ysig2))
    b = ((sin2t / xsig2) - (sin2t / ysig2))    
    c = ((sint2 / xsig2) + (cost2 / ysig2))

    g = amp * np.exp(-0.5*((a * xdiff**2) + (b * xdiff * ydiff) +
                           (c * ydiff**2)))
    
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
            dg_dA = g / amp
            derivative.append(dg_dA)
        if nderiv>=2:        
            dg_dx_mean = g * 0.5*((2 * a * xdiff) + (b * ydiff))
            derivative.append(dg_dx_mean)
        if nderiv>=3:
            dg_dy_mean = g * 0.5*((2 * c * ydiff) + (b * xdiff))
            derivative.append(dg_dy_mean)
        if nderiv>=4:
            xdiff2 = xdiff ** 2
            ydiff2 = ydiff ** 2
            xsig3 = xsig ** 3
            da_dxsig = -cost2 / xsig3
            db_dxsig = -sin2t / xsig3            
            dc_dxsig = -sint2 / xsig3            
            dg_dxsig = g * (-(da_dxsig * xdiff2 +
                              db_dxsig * xdiff * ydiff +
                              dc_dxsig * ydiff2))
            derivative.append(dg_dxsig)
        if nderiv>=5:
            ysig3 = ysig ** 3
            da_dysig = -sint2 / ysig3
            db_dysig = sin2t / ysig3            
            dc_dysig = -cost2 / ysig3            
            dg_dysig = g * (-(da_dysig * xdiff2 +
                              db_dysig * xdiff * ydiff +
                              dc_dysig * ydiff2))
            derivative.append(dg_dysig)
        if nderiv>=6:
            sint = np.sin(theta)
            cost = np.cos(theta)
            cos2t = np.cos(2.0*theta)
            da_dtheta = (sint * cost * ((1. / ysig2) - (1. / xsig2)))
            db_dtheta = (cos2t / xsig2) - (cos2t / ysig2)            
            dc_dtheta = -da_dtheta            
            dg_dtheta = g * (-(da_dtheta * xdiff2 +
                               db_dtheta * xdiff * ydiff +
                               dc_dtheta * ydiff2))
            derivative.append(dg_dtheta)

        return g,derivative
            
    # No derivative
    else:        
        return g


def gaussian2d_fwhm(pars):
    """ Return the FWHM of a 2D Gaussian."""
    # pars = [amplitude, x0, y0, xsig, ysig, theta]

    # xdiff = x-x0
    # ydiff = y-y0
    # f(x,y) = A*exp(-0.5 * (a*xdiff**2 + b*xdiff*ydiff + c*ydiff**2))

    xsig = pars[3]
    ysig = pars[4]

    # The mean radius of an ellipse is: (2a+b)/3
    sig_major = np.max([xsig,ysig])
    sig_minor = np.min([xsig,ysig])
    mnsig = (2.0*sig_major+sig_minor)/3.0
    # Convert sigma to FWHM
    # FWHM = 2*sqrt(2*ln(2))*sig ~ 2.35482*sig
    fwhm = mnsig*2.35482

    return fwhm


def gaussian2d_flux(pars):
    """ Return the total Flux of a 2D Gaussian."""
    # Volume is 2*pi*A*sigx*sigy
    amp = pars[0]
    xsig = pars[3]
    ysig = pars[4]    
    volume = 2*np.pi*amp*xsig*ysig
    
    return volume
    

def gaussian2d_sigtheta2abc(xstd,ystd,theta):
    """ Convert 2D Gaussian sigma_x, sigma_y and theta to a, b, c coefficients."""
    
    # xdiff = x-x0
    # ydiff = y-y0
    # f(x,y) = A*exp(-0.5 * (a*xdiff**2 + b*xdiff*ydiff + c*ydiff2))
    
    # a is x**2 term
    # b is y**2 term
    # c is x*y term

    #cost2 = np.cos(theta) ** 2
    #sint2 = np.sin(theta) ** 2
    #sin2t = np.sin(2. * theta)
    #xstd2 = x_stddev ** 2
    #ystd2 = y_stddev ** 2
    #a = ((cost2 / xstd2) + (sint2 / ystd2))
    #b = ((sin2t / xstd2) - (sin2t / ystd2))
    #c = ((sint2 / xstd2) + (cost2 / ystd2))    


    cost2 = np.cos(theta) ** 2
    sint2 = np.sin(theta) ** 2
    sin2t = np.sin(2. * theta)
    xstd2 = xstd ** 2
    ystd2 = ystd ** 2
    a = ((cost2 / xstd2) + (sint2 / ystd2))
    b = ((sin2t / xstd2) - (sin2t / ystd2))
    c = ((sint2 / xstd2) + (cost2 / ystd2))    

    return a,b,c

    
def gaussian2d_abc2sigtheta(a,b,c):
    """ Convert 2D Gaussian a, b, c coefficients to sigma_x, sigma_y and theta."""
    
    # xdiff = x-x0
    # ydiff = y-y0
    # f(x,y) = A*exp(-0.5 * (a*xdiff**2 + b*xdiff*ydiff + c*ydiff**2))
    
    # a is x**2 term
    # b is x*y term
    # c is y**2 term
    
    #cost2 = np.cos(theta) ** 2
    #sint2 = np.sin(theta) ** 2
    #sin2t = np.sin(2. * theta)
    #xstd2 = x_stddev ** 2
    #ystd2 = y_stddev ** 2
    #a = ((cost2 / xstd2) + (sint2 / ystd2))
    #b = ((sin2t / xstd2) - (sin2t / ystd2))
    #c = ((sint2 / xstd2) + (cost2 / ystd2))    

    # a+c = 1/xstd2 + 1/ystd2
    # b = sin2t * (1/xstd2 + 1/ystd2)
    # tan 2*theta = b/(a-c)
    if a==c or b==0:
        theta = 0.0
    else:
        theta = np.arctan2(b,a-c)/2.0

    if theta==0:
        # a = 1 / xstd2
        # b = 0        
        # c = 1 / ystd2
        xstd = 1/np.sqrt(a)
        ystd = 1/np.sqrt(c)
        return xstd,ystd,theta        
        
    sin2t = np.sin(2.0*theta)
    # b/sin2t + (a+c) = 2/xstd2
    # xstd2 = 2.0/(b/sin2t + (a+c))
    xstd = np.sqrt( 2.0/(b/sin2t + (a+c)) )

    # a+c = 1/xstd2 + 1/ystd2
    ystd = np.sqrt( 1/(a+c-1/xstd**2) )

    return xstd,ystd,theta

    
def gaussian2d_integrate(x, y, pars, deriv=False, nderiv=None, osamp=4):
    """ Two dimensional Gaussian model function integrated over the pixels."""

    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    
    # Deal with the shape, must be 1D to function properly
    shape = x.shape
    ndim = x.ndim
    if ndim>1:
        x = x.flatten()
        y = y.flatten()

    osamp2 = float(osamp)**2
    nx = x.size
    dx = (np.arange(osamp).astype(float)+1)/osamp-(1/(2*osamp))-0.5
    dx2 = np.tile(dx,(osamp,1))
    x2 = np.tile(x,(osamp,osamp,1)) + np.tile(dx2.T,(nx,1,1)).T
    y2 = np.tile(y,(osamp,osamp,1)) + np.tile(dx2,(nx,1,1)).T    
    
    # pars = [amplitude, x0, y0, xsigma, ysigma, theta]
    theta = np.deg2rad(pars[5])
    cost2 = np.cos(theta) ** 2
    sint2 = np.sin(theta) ** 2
    sin2t = np.sin(2. * theta)
    xstd2 = pars[3] ** 2
    ystd2 = pars[4] ** 2
    xdiff = x2 - pars[1]
    ydiff = y2 - pars[2]
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
            derivative.append(np.sum(np.sum(dg_dA,axis=0),axis=0)/osamp2)
        if nderiv>=2:        
            dg_dx_mean = g * ((2. * a * xdiff) + (b * ydiff))
            derivative.append(np.sum(np.sum(dg_dx_mean,axis=0),axis=0)/osamp2)
        if nderiv>=3:
            dg_dy_mean = g * ((b * xdiff) + (2. * c * ydiff))
            derivative.append(np.sum(np.sum(dg_dy_mean,axis=0),axis=0)/osamp2)
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
            derivative.append(np.sum(np.sum(dg_dx_stddev,axis=0),axis=0)/osamp2)
        if nderiv>=5:
            ystd3 = pars[2] ** 3            
            da_dy_stddev = -sint2 / ystd3
            db_dy_stddev = sin2t / ystd3
            dc_dy_stddev = -cost2 / ystd3        
            dg_dy_stddev = g * (-(da_dy_stddev * xdiff2 +
                                  db_dy_stddev * xdiff * ydiff +
                                  dc_dy_stddev * ydiff2))
            derivative.append(np.sum(np.sum(dg_dy_stddev,axis=0),axis=0)/osamp2)
        if nderiv>=6:
            cos2t = np.cos(2. * theta)            
            da_dtheta = (sint * cost * ((1. / ystd2) - (1. / xstd2)))
            db_dtheta = (cos2t / xstd2) - (cos2t / ystd2)
            dc_dtheta = -da_dtheta        
            dg_dtheta = g * (-(da_dtheta * xdiff2 +
                               db_dtheta * xdiff * ydiff +
                               dc_dtheta * ydiff2))
            derivative.append(np.sum(np.sum(dg_dtheta,axis=0),axis=0)/osamp2)

        g = np.sum(np.sum(g,axis=0),axis=0)/osamp2

        # Reshape
        if ndim>1:
            g = g.reshape(shape)
            derivative = [d.reshape(shape) for d in derivative]
        
        return g,derivative

    # No derivative
    else:
        g = np.sum(np.sum(g,axis=0),axis=0)/osamp2
        # Reshape
        if ndim>1:
            g = g.reshape(shape)
        return g
    

def moffat2d(x, y, pars, deriv=False, nderiv=None):
    """Two dimensional Moffat model function"""
    # pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta]

    xdiff = x - pars[1]
    ydiff = y - pars[2]
    amp = pars[0]
    xsig = pars[3]
    ysig = pars[4]
    theta = pars[5]
    beta = pars[6]
    cost2 = np.cos(theta) ** 2
    sint2 = np.sin(theta) ** 2
    sin2t = np.sin(2. * theta)
    xsig2 = xsig ** 2
    ysig2 = ysig ** 2
    a = ((cost2 / xsig2) + (sint2 / ysig2))
    b = ((sin2t / xsig2) - (sin2t / ysig2))    
    c = ((sint2 / xsig2) + (cost2 / ysig2))

    rr_gg = (a * xdiff ** 2) + (b * xdiff * ydiff) + (c * ydiff ** 2)
    g = amp * (1 + rr_gg) ** (-beta)
    
    
    # Compute derivative as well
    if deriv is True:

        # How many derivative terms to return
        if nderiv is not None:
            if nderiv <=0:
                nderiv = 7
        else:
            nderiv = 7
        
        derivative = []
        if nderiv>=1:
            dg_dA = g / amp
            derivative.append(dg_dA)
        if nderiv>=2:
            dg_dx_0 = beta*g/(1+rr_gg) * (2*a*xdiff + b*ydiff)
            derivative.append(dg_dx_0)            
        if nderiv>=3:
            dg_dy_0 = beta*g/(1+rr_gg) * (2*c*ydiff + b*xdiff)
            derivative.append(dg_dy_0)
        if nderiv>=4:
            xdiff2 = xdiff ** 2
            ydiff2 = ydiff ** 2
            xsig3 = xsig ** 3
            da_dxsig = -cost2 / xsig3
            db_dxsig = -sin2t / xsig3            
            dc_dxsig = -sint2 / xsig3            
            dg_dxsig = (-beta)*g/(1+rr_gg) * 2*(da_dxsig * xdiff2 +
                                                db_dxsig * xdiff * ydiff +
                                                dc_dxsig * ydiff2)
            derivative.append(dg_dxsig)
        if nderiv>=5:
            ysig3 = ysig ** 3
            da_dysig = -sint2 / ysig3
            db_dysig = sin2t / ysig3            
            dc_dysig = -cost2 / ysig3            
            dg_dysig = (-beta)*g/(1+rr_gg) * 2*(da_dysig * xdiff2 +
                                                db_dysig * xdiff * ydiff +
                                                dc_dysig * ydiff2)
            derivative.append(dg_dysig)            
        if nderiv>=6:
            sint = np.sin(theta)
            cost = np.cos(theta)
            cos2t = np.cos(2.0*theta)
            da_dtheta = (sint * cost * ((1. / ysig2) - (1. / xsig2)))
            db_dtheta = (cos2t / xsig2) - (cos2t / ysig2)
            dc_dtheta = -da_dtheta            
            dg_dtheta = (-beta)*g/(1+rr_gg) * 2*(da_dtheta * xdiff2 +
                                                 db_dtheta * xdiff * ydiff +
                                                 dc_dtheta * ydiff2)
            derivative.append(dg_dtheta)
        if nderiv>=7:            
            dg_dbeta = -g * np.log(1 + rr_gg)
            derivative.append(dg_dbeta) 
            
        return g,derivative

    # No derivative
    else:
        return g

def moffat2d_fwhm(pars):
    """ Return the FWHM of a 2D Moffat function."""
    # [amplitude, x0, y0, xsig, ysig, theta, beta]
    # https://nbviewer.jupyter.org/github/ysbach/AO_2017/blob/master/04_Ground_Based_Concept.ipynb#1.2.-Moffat

    xsig = pars[3]
    ysig = pars[4]
    beta = pars[6]
    
    # The mean radius of an ellipse is: (2a+b)/3
    sig_major = np.max([xsig,ysig])
    sig_minor = np.min([xsig,ysig])
    mnsig = (2.0*sig_major+sig_minor)/3.0
    
    return 2.0 * np.abs(mnsig) * np.sqrt(2.0 ** (1.0/beta) - 1.0)


def moffat2d_flux(pars):
    """ Return the total Flux of a 2D Moffat."""
    # [amplitude, x0, y0, xsig, ysig, theta, beta]
    # Volume is 2*pi*A*sigx*sigy
    # area of 1D moffat function is pi*alpha**2 / (beta-1)
    # maybe the 2D moffat volume is (xsig*ysig*pi**2/(beta-1))**2

    amp = pars[0]
    xsig = pars[3]
    ysig = pars[4]
    beta = pars[6]

    # This worked for beta=2.5, but was too high by ~1.05-1.09 for beta=1.5
    #volume = amp * xstd*ystd*np.pi/(beta-1)
    volume = amp * xsig*ysig*np.pi/(beta-1)
    # what is the beta dependence?? linear is very close!

    # I think undersampling is becoming an issue at beta=3.5 with fwhm=2.78
    
    return volume


def moffat2d_integrate(x, y, pars, deriv=False, nderiv=None, osamp=4):
    """Two dimensional Moffat model function integrated over the pixels"""
    # pars = [amplitude, x0, y0, sigma, beta]

    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    
    # Deal with the shape, must be 1D to function properly
    shape = x.shape
    ndim = x.ndim
    if ndim>1:
        x = x.flatten()
        y = y.flatten()
    
    osamp2 = float(osamp)**2
    nx = x.size
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

        # Reshape
        if ndim>1:
            g = g.reshape(shape)
            derivative = [d.reshape(shape) for d in derivative]
        
        return g,derivative

    # No derivative
    else:
        g = np.sum(np.sum(g,axis=0),axis=0)/osamp2
        # Reshape
        if ndim>1:
            g = g.reshape(shape)
        return g

    
def penny2d(x, y, pars, deriv=False, nderiv=None):
    """ Gaussian core and Lorentzian-like wings, only Gaussian is tilted."""
    # Lorentzian are azimuthally symmetric.
    # Lorentzian cannot be normalized, use Moffat beta=1.2 instead
    # pars = [amp,x0,y0,xsig,ysig,theta, relamp,sigma]

    xdiff = x - pars[1]
    ydiff = y - pars[2]
    amp = pars[0]
    xsig = pars[3]
    ysig = pars[4]
    theta = pars[5]
    cost2 = np.cos(theta) ** 2
    sint2 = np.sin(theta) ** 2
    sin2t = np.sin(2. * theta)
    xsig2 = xsig ** 2
    ysig2 = ysig ** 2
    a = ((cost2 / xsig2) + (sint2 / ysig2))
    b = ((sin2t / xsig2) - (sin2t / ysig2))    
    c = ((sint2 / xsig2) + (cost2 / ysig2))
    relamp = pars[6]    
    # Gaussian component
    g = amp * (1-relamp) * np.exp(-0.5*((a * xdiff ** 2) + (b * xdiff*ydiff) +
                                        (c * ydiff ** 2)))
    # Add Lorentzian/Moffat beta=1.2 wings
    sigma = pars[7]
    rr_gg = (xdiff ** 2 + ydiff ** 2) / sigma ** 2
    beta = 1.2
    l = amp * relamp / (1 + rr_gg)**(beta)
    # Sum of Gaussian + Lorentzian
    f = g + l

   
    # Compute derivative as well
    if deriv is True:

        # How many derivative terms to return
        if nderiv is not None:
            if nderiv <=0:
                nderiv = 8
        else:
            nderiv = 8
            
        derivative = []
        if nderiv>=1:
            df_dA = f / amp
            derivative.append(df_dA)
        if nderiv>=2:
            #df_dx_mean = ( g * 0.5*((2 * a * xdiff) + (b * ydiff)) +
            #               2*l*xdiff/(sigma**2 * (1+rr_gg)) )
            df_dx_mean = ( g * 0.5*((2 * a * xdiff) + (b * ydiff)) +                           
                           2*beta*l*xdiff/(sigma**2 * (1+rr_gg)) )            
            derivative.append(df_dx_mean)
        if nderiv>=3:
            #df_dy_mean = ( g * 0.5*((2 * c * ydiff) + (b * xdiff)) +
            #               2*l*ydiff/(sigma**2 * (1+rr_gg)) )
            df_dy_mean = ( g * 0.5*((2 * c * ydiff) + (b * xdiff)) +
                           2*beta*l*ydiff/(sigma**2 * (1+rr_gg)) )            
            derivative.append(df_dy_mean)
        if nderiv>=4:
            xdiff2 = xdiff ** 2
            ydiff2 = ydiff ** 2
            xsig3 = xsig ** 3
            da_dxsig = -cost2 / xsig3
            db_dxsig = -sin2t / xsig3            
            dc_dxsig = -sint2 / xsig3            
            df_dxsig = g * (-(da_dxsig * xdiff2 +
                              db_dxsig * xdiff * ydiff +
                              dc_dxsig * ydiff2))
            derivative.append(df_dxsig)
        if nderiv>=5:
            ysig3 = ysig ** 3
            da_dysig = -sint2 / ysig3
            db_dysig = sin2t / ysig3            
            dc_dysig = -cost2 / ysig3            
            df_dysig = g * (-(da_dysig * xdiff2 +
                              db_dysig * xdiff * ydiff +
                              dc_dysig * ydiff2))
            derivative.append(df_dysig)
        if nderiv>=6:
            sint = np.sin(theta)
            cost = np.cos(theta)
            cos2t = np.cos(2.0*theta)
            da_dtheta = (sint * cost * ((1. / ysig2) - (1. / xsig2)))
            db_dtheta = (cos2t / xsig2) - (cos2t / ysig2)            
            dc_dtheta = -da_dtheta            
            df_dtheta = g * (-(da_dtheta * xdiff2 +
                               db_dtheta * xdiff * ydiff +
                               dc_dtheta * ydiff2))
            derivative.append(df_dtheta)
        if nderiv>=7:
            df_drelamp = -g/(1-relamp) + l/relamp
            derivative.append(df_drelamp)
        if nderiv>=8:
            #df_dsigma = l/(1+rr_gg) * 2*rr_gg/sigma
            df_dsigma = beta*l/(1+rr_gg) * 2*(xdiff2+ydiff2)/sigma**3 
            derivative.append(df_dsigma)
            
        return f,derivative
            
    # No derivative
    else:        
        return f


def penny2d_fwhm(pars):
    """ Return the FWHM of a 2D Penny function."""
    # [amplitude, x0, y0, xsig, ysig, theta, relative amplitude, sigma]

    amp = pars[0]
    xsig = pars[3]
    ysig = pars[4]
    relamp = pars[6]
    sigma = pars[7]
    beta = 1.2   # Moffat
    
    # The mean radius of an ellipse is: (2a+b)/3
    sig_major = np.max([xsig,ysig])
    sig_minor = np.min([xsig,ysig])
    mnsig = (2.0*sig_major+sig_minor)/3.0
    # Convert sigma to FWHM
    # FWHM = 2*sqrt(2*ln(2))*sig ~ 2.35482*sig
    gfwhm = mnsig*2.35482
    if relamp==0:
        return gfwhm
    
    # Moffat beta=1.2 FWHM
    mfwhm = 2.0 * np.abs(sigma) * np.sqrt(2.0 ** (1.0/beta) - 1.0)

    # Generate a small profile
    x = np.arange( np.min([gfwhm,mfwhm])/2.35/2, np.max([gfwhm,mfwhm]), 0.5)
    f = (1-relamp)*np.exp(-0.5*(x/mnsig)**2) + relamp/(1+(x/sigma)**2)**beta
    hwhm = np.interp(0.5,f[::-1],x[::-1])
    fwhm = 2*hwhm
    
    return fwhm


def penny2d_flux(pars):
    """ Return the total Flux of a 2D Penny function."""
    # [amplitude, x0, y0, xsig, ysig, theta, relative amplitude, sigma]    

    # Volume is 2*pi*A*sigx*sigy
    # area of 1D moffat function is pi*alpha**2 / (beta-1)
    # maybe the 2D moffat volume is (xsig*ysig*pi**2/(beta-1))**2

    amp = pars[0]
    xsig = pars[3]
    ysig = pars[4]
    relamp = pars[6]
    sigma = pars[7]
    beta = 1.2   # Moffat

    # Gaussian portion
    # Volume is 2*pi*A*sigx*sigy
    gvolume = 2*np.pi*amp*(1-relamp)*xsig*ysig

    # Moffat beta=1.2 wings portion
    lvolume = amp*relamp * sigma**2 * np.pi/(beta-1)
    
    # Sum
    volume = gvolume + lvolume
    
    return volume

def penny2d_integrate(x, y, pars, deriv=False, nderiv=None, osamp=4):
    """ Gaussian core and Lorentzian-like wings, only Gaussian is tilted integrated over the pixels."""
    # Lorentzian are azimuthally symmetric.
    # Lorentzian cannot be normalized, use Moffat beta=1.2 instead
    # pars = [amp,x0,y0,xsig,ysig,theta, relamp,sigma]

    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    
    # Deal with the shape, must be 1D to function properly
    shape = x.shape
    ndim = x.ndim
    if ndim>1:
        x = x.flatten()
        y = y.flatten()
    
    osamp2 = float(osamp)**2
    nx = x.size
    dx = (np.arange(osamp).astype(float)+1)/osamp-(1/(2*osamp))-0.5
    dx2 = np.tile(dx,(osamp,1))
    x2 = np.tile(x,(osamp,osamp,1)) + np.tile(dx2.T,(nx,1,1)).T
    y2 = np.tile(y,(osamp,osamp,1)) + np.tile(dx2,(nx,1,1)).T
    
    xdiff = x2 - pars[1]
    ydiff = y2 - pars[2]
    amp = pars[0]
    xsig = pars[3]
    ysig = pars[4]
    theta = pars[5]
    cost2 = np.cos(theta) ** 2
    sint2 = np.sin(theta) ** 2
    sin2t = np.sin(2. * theta)
    xsig2 = xsig ** 2
    ysig2 = ysig ** 2
    a = ((cost2 / xsig2) + (sint2 / ysig2))
    b = ((sin2t / xsig2) - (sin2t / ysig2))    
    c = ((sint2 / xsig2) + (cost2 / ysig2))
    relamp = pars[6]    
    # Gaussian component
    g = amp * (1-relamp) * np.exp(-0.5*((a * xdiff ** 2) + (b * xdiff*ydiff) +
                                        (c * ydiff ** 2)))
    # Add Lorentzian/Moffat beta=1.2 wings
    sigma = pars[7]
    rr_gg = (xdiff ** 2 + ydiff ** 2) / sigma ** 2
    beta = 1.2
    l = amp * relamp / (1 + rr_gg)**(beta)
    # Sum of Gaussian + Lorentzian
    f = g + l

   
    # Compute derivative as well
    if deriv is True:

        # How many derivative terms to return
        if nderiv is not None:
            if nderiv <=0:
                nderiv = 8
        else:
            nderiv = 8
            
        derivative = []
        if nderiv>=1:
            df_dA = f / amp
            derivative.append(np.sum(np.sum(df_dA,axis=0),axis=0)/osamp2)
        if nderiv>=2:
            #df_dx_mean = ( g * 0.5*((2 * a * xdiff) + (b * ydiff)) +
            #               2*l*xdiff/(sigma**2 * (1+rr_gg)) )
            df_dx_mean = ( g * 0.5*((2 * a * xdiff) + (b * ydiff)) +                           
                           2*beta*l*xdiff/(sigma**2 * (1+rr_gg)) )            
            derivative.append(np.sum(np.sum(df_dx_mean,axis=0),axis=0)/osamp2)
        if nderiv>=3:
            #df_dy_mean = ( g * 0.5*((2 * c * ydiff) + (b * xdiff)) +
            #               2*l*ydiff/(sigma**2 * (1+rr_gg)) )
            df_dy_mean = ( g * 0.5*((2 * c * ydiff) + (b * xdiff)) +
                           2*beta*l*ydiff/(sigma**2 * (1+rr_gg)) )            
            derivative.append(np.sum(np.sum(df_dy_mean,axis=0),axis=0)/osamp2)
        if nderiv>=4:
            xdiff2 = xdiff ** 2
            ydiff2 = ydiff ** 2
            xsig3 = xsig ** 3
            da_dxsig = -cost2 / xsig3
            db_dxsig = -sin2t / xsig3            
            dc_dxsig = -sint2 / xsig3            
            df_dxsig = g * (-(da_dxsig * xdiff2 +
                              db_dxsig * xdiff * ydiff +
                              dc_dxsig * ydiff2))
            derivative.append(np.sum(np.sum(df_dxsig,axis=0),axis=0)/osamp2)
        if nderiv>=5:
            ysig3 = ysig ** 3
            da_dysig = -sint2 / ysig3
            db_dysig = sin2t / ysig3            
            dc_dysig = -cost2 / ysig3            
            df_dysig = g * (-(da_dysig * xdiff2 +
                              db_dysig * xdiff * ydiff +
                              dc_dysig * ydiff2))
            derivative.append(np.sum(np.sum(df_dysig,axis=0),axis=0)/osamp2)
        if nderiv>=6:
            sint = np.sin(theta)
            cost = np.cos(theta)
            cos2t = np.cos(2.0*theta)
            da_dtheta = (sint * cost * ((1. / ysig2) - (1. / xsig2)))
            db_dtheta = (cos2t / xsig2) - (cos2t / ysig2)            
            dc_dtheta = -da_dtheta            
            df_dtheta = g * (-(da_dtheta * xdiff2 +
                               db_dtheta * xdiff * ydiff +
                               dc_dtheta * ydiff2))
            derivative.append(np.sum(np.sum(df_dtheta,axis=0),axis=0)/osamp2)
        if nderiv>=7:
            df_drelamp = -g/(1-relamp) + l/relamp
            derivative.append(df_drelamp)
        if nderiv>=8:
            #df_dsigma = l/(1+rr_gg) * 2*rr_gg/sigma
            df_dsigma = beta*l/(1+rr_gg) * 2*(xdiff2+ydiff2)/sigma**3 
            derivative.append(np.sum(np.sum(df_dsigma,axis=0),axis=0)/osamp2)

        f = np.sum(np.sum(g,axis=0),axis=0)/osamp2

        # Reshape
        if ndim>1:
            f = f.reshape(shape)
            derivative = [d.reshape(shape) for d in derivative]
        
        return f,derivative

    # No derivative
    else:
        f = np.sum(np.sum(f,axis=0),axis=0)/osamp2
        # Reshape
        if ndim>1:
            f = f.reshape(shape)
        return f



def gausspow2d(x, y, pars, deriv=False, nderiv=None):
    """ DoPHOT PSF, sum of elliptical Gaussians."""
    # Schechter, Mateo & Saha (1993), eq. 1 on pg.4
    # I(x,y) = Io * (1+z2+0.5*beta4*z2**2+(1/6)*beta6*z2**3)**(-1)
    # z2 = [0.5*(x**2/sigx**2 + 2*sigxy*x*y + y**2/sigy**2]
    # x = (x'-x0)
    # y = (y'-y0)
    # nominal center of image at (x0,y0)
    # if beta4=beta6=1, then it's just a truncated power series for a Gaussian
    # 8 free parameters
    # pars = [amplitude, x0, y0, sigx, sigy, theta, beta4, beta6]
    
    xdiff = x - pars[1]
    ydiff = y - pars[2]
    amp = pars[0]
    xsig = pars[3]
    ysig = pars[4]
    theta = pars[5]
    beta4 = pars[6]
    beta6 = pars[7]
    xdiff2 = xdiff**2
    ydiff2 = ydiff**2

    # convert sigx, sigy and theta to a, b, c terms
    cost2 = np.cos(theta) ** 2
    sint2 = np.sin(theta) ** 2
    sin2t = np.sin(2. * theta)
    xsig2 = xsig ** 2
    ysig2 = ysig ** 2
    a = ((cost2 / xsig2) + (sint2 / ysig2))
    b = ((sin2t / xsig2) - (sin2t / ysig2))    
    c = ((sint2 / xsig2) + (cost2 / ysig2))
    
    z2 = 0.5*(a*xdiff2 + b*xdiff*ydiff + c*ydiff2)
    gxy = (1+z2+0.5*beta4*z2**2+(1.0/6.0)*beta6*z2**3)
    g = amp / gxy

    
    # Compute derivative as well
    if deriv is True:

        # How many derivative terms to return
        if nderiv is not None:
            if nderiv <=0:
                nderiv = 8
        else:
            nderiv = 8
        
        derivative = []
        if nderiv>=1:
            dg_dA = g / amp
            derivative.append(dg_dA)
        if nderiv>=2:
            dg_dx_0 = g * (1+beta4*z2+0.5*beta6*z2**2)*(2*a*xdiff+b*ydiff) / gxy
            derivative.append(dg_dx_0)            
        if nderiv>=3:
            dg_dy_0 = g * (1+beta4*z2+0.5*beta6*z2**2)*(2*c*ydiff+b*xdiff) / gxy            
            derivative.append(dg_dy_0)
        if nderiv>=4:
            xsig3 = xsig ** 3
            da_dxsig = -cost2 / xsig3
            db_dxsig = -sin2t / xsig3            
            dc_dxsig = -sint2 / xsig3         
            dg_dxsig = -g * (1+beta4*z2+0.5*beta6*z2**2)*(da_dxsig*xdiff2+db_dxsig*xdiff*ydiff+dc_dxsig*ydiff2) / gxy     
            derivative.append(dg_dxsig)
        if nderiv>=5:
            ysig3 = ysig ** 3
            da_dysig = -sint2 / ysig3
            db_dysig = sin2t / ysig3            
            dc_dysig = -cost2 / ysig3        
            dg_dysig = -g * (1+beta4*z2+0.5*beta6*z2**2)*(da_dysig*xdiff2+db_dysig*xdiff*ydiff+dc_dysig*ydiff2) / gxy     
            derivative.append(dg_dysig)
        if nderiv>=6:
            sint = np.sin(theta)
            cost = np.cos(theta)
            cos2t = np.cos(2.0*theta)
            da_dtheta = (sint * cost * ((1. / ysig2) - (1. / xsig2)))
            db_dtheta = (cos2t / xsig2) - (cos2t / ysig2)            
            dc_dtheta = -da_dtheta 
            dg_dtheta = -g * (1+beta4*z2+0.5*beta6*z2**2)*(da_dtheta*xdiff2+db_dtheta*xdiff*ydiff+dc_dtheta*ydiff2) / gxy     
            derivative.append(dg_dtheta)
        if nderiv>=7:            
            dg_dbeta4 = -g * (0.5*z2**2) / gxy
            derivative.append(dg_dbeta4)
        if nderiv>=8:            
            dg_dbeta6 = -g * ((1.0/6.0)*z2**3) / gxy
            derivative.append(dg_dbeta6) 
            
        return g,derivative

    # No derivative
    else:
        return g


def gausspow2d_fwhm(pars):
    """ Return the FWHM of a 2D DoPHOT Gausspow function."""
    # pars = [amplitude, x0, y0, xsig, ysig, theta, beta4, beta6]    

    amp = pars[0]
    xsig = pars[3]
    ysig = pars[4]
    theta = pars[5]
    beta4 = pars[6]
    beta6 = pars[7]

    # convert sigx, sigy and theta to a, b, c terms
    cost2 = np.cos(theta) ** 2
    sint2 = np.sin(theta) ** 2
    sin2t = np.sin(2. * theta)
    xsig2 = xsig ** 2
    ysig2 = ysig ** 2
    a = ((cost2 / xsig2) + (sint2 / ysig2))
    b = ((sin2t / xsig2) - (sin2t / ysig2))    
    c = ((sint2 / xsig2) + (cost2 / ysig2))
    
    # The mean radius of an ellipse is: (2a+b)/3
    sig_major = np.max([xsig,ysig])
    sig_minor = np.min([xsig,ysig])
    mnsig = (2.0*sig_major+sig_minor)/3.0
    # Convert sigma to FWHM
    # FWHM = 2*sqrt(2*ln(2))*sig ~ 2.35482*sig
    gfwhm = mnsig*2.35482

    # Generate a small profile along one axis with xsig=mnsig
    x = np.arange(gfwhm/2.35/2, gfwhm, 0.5)
    z2 = 0.5*(x/mnsig)**2
    gxy = (1+z2+0.5*beta4*z2**2+(1.0/6.0)*beta6*z2**3)
    f = amp / gxy

    hwhm = np.interp(0.5,f[::-1],x[::-1])
    fwhm = 2*hwhm
    
    return fwhm

def gausspow2d_flux(pars):
    """ Return the flux of a 2D DoPHOT Gausspow function."""
    # pars = [amplitude, x0, y0, xsig, ysig, theta, beta4, beta6]

    amp = pars[0]
    xsig = pars[3]
    ysig = pars[4]
    beta4 = pars[6]
    beta6 = pars[7]

    # Theta angle doesn't matter

    # Integral from 0 to +infinity of
    # dx/(1+0.5*x+beta4/8*x^2+beta6/48*x^3)
    # I calculated the integral for various values of beta4 and beta6 using
    # WolframAlpha, https://www.wolframalpha.com/
    # Integrate 1/(1+0.5*x+beta4*Power[x,2]/8+beta6*Power[x,3]/48)dx, x=0 to inf
    p = [0.20326739, 0.019689948, 0.023564239, 0.63367201, 0.044905046, 0.28862448]
    integral = p[0]/(p[1]+p[2]*beta4**p[3]+p[4]*beta6**p[5])
    # The integral is then multiplied by amp*pi*xsig*ysig

    # This seems to be accurate to ~0.5%
    
    volume = np.pi*amp*xsig*ysig*integral
    
    return volume

    
def gausspow2d_integrate(x, y, pars, deriv=False, nderiv=None, osamp=4):
    """ DoPHOT PSF, integrated over the pixels."""
    # Schechter, Mateo & Saha (1993), eq. 1 on pg.4
    # I(x,y) = Io * (1+z2+0.5*beta4*z2**2+(1/6)*beta6*z2**3)**(-1)
    # z2 = [0.5*(x**2/sigx**2 + 2*sigxy*x*y + y**2/sigy**2]
    # x = (x'-x0)
    # y = (y'-y0)
    # nominal center of image at (x0,y0)
    # if beta4=beta6=1, then it's just a truncated power series for a Gaussian
    # 8 free parameters
    # pars = [amplitude, x0, y0, sigx, sigy, theta, beta4, beta6]

    # Deal with the shape, must be 1D to function properly
    shape = x.shape
    ndim = x.ndim
    if ndim>1:
        x = x.flatten()
        y = y.flatten()

    osamp2 = float(osamp)**2
    nx = x.size
    dx = (np.arange(osamp).astype(float)+1)/osamp-(1/(2*osamp))-0.5
    dx2 = np.tile(dx,(osamp,1))
    x2 = np.tile(x,(osamp,osamp,1)) + np.tile(dx2.T,(nx,1,1)).T
    y2 = np.tile(y,(osamp,osamp,1)) + np.tile(dx2,(nx,1,1)).T    

    # pars = [amplitude, x0, y0, sigx, sigy, theta, beta4, beta6]    
    xdiff = x2 - pars[1]
    ydiff = y2 - pars[2]
    amp = pars[0]
    xsig = pars[3]
    ysig = pars[4]
    theta = pars[5]
    beta4 = pars[6]
    beta6 = pars[7]
    xdiff2 = xdiff**2
    ydiff2 = ydiff**2

    # convert sigx, sigy and theta to a, b, c terms
    cost2 = np.cos(theta) ** 2
    sint2 = np.sin(theta) ** 2
    sin2t = np.sin(2. * theta)
    xsig2 = xsig ** 2
    ysig2 = ysig ** 2
    a = ((cost2 / xsig2) + (sint2 / ysig2))
    b = ((sin2t / xsig2) - (sin2t / ysig2))    
    c = ((sint2 / xsig2) + (cost2 / ysig2))
    
    z2 = 0.5*(a*xdiff2 + b*xdiff*ydiff + c*ydiff2)
    gxy = (1+z2+0.5*beta4*z2**2+(1.0/6.0)*beta6*z2**3)
    g = amp / gxy

    
    # Compute derivative as well
    if deriv is True:

        # How many derivative terms to return
        if nderiv is not None:
            if nderiv <=0:
                nderiv = 8
        else:
            nderiv = 8
        
        derivative = []
        if nderiv>=1:
            dg_dA = g / amp
            derivative.append(np.sum(np.sum(dg_dA,axis=0),axis=0)/osamp2)
        if nderiv>=2:
            dg_dx_0 = g * (1+beta4*z2+0.5*beta6*z2**2)*(2*a*xdiff+b*ydiff) / gxy
            derivative.append(np.sum(np.sum(dg_dx_0,axis=0),axis=0)/osamp2)
        if nderiv>=3:
            dg_dy_0 = g * (1+beta4*z2+0.5*beta6*z2**2)*(2*c*ydiff+b*xdiff) / gxy            
            derivative.append(np.sum(np.sum(dg_dy_0,axis=0),axis=0)/osamp2)
        if nderiv>=4:
            xsig3 = xsig ** 3
            da_dxsig = -cost2 / xsig3
            db_dxsig = -sin2t / xsig3            
            dc_dxsig = -sint2 / xsig3         
            dg_dxsig = -g * (1+beta4*z2+0.5*beta6*z2**2)*(da_dxsig*xdiff2+db_dxsig*xdiff*ydiff+dc_dxsig*ydiff2) / gxy     
            derivative.append(np.sum(np.sum(dg_dxsig,axis=0),axis=0)/osamp2)
        if nderiv>=5:
            ysig3 = ysig ** 3
            da_dysig = -sint2 / ysig3
            db_dysig = sin2t / ysig3            
            dc_dysig = -cost2 / ysig3        
            dg_dysig = -g * (1+beta4*z2+0.5*beta6*z2**2)*(da_dysig*xdiff2+db_dysig*xdiff*ydiff+dc_dysig*ydiff2) / gxy     
            derivative.append(np.sum(np.sum(dg_dysig,axis=0),axis=0)/osamp2)
        if nderiv>=6:
            sint = np.sin(theta)
            cost = np.cos(theta)
            cos2t = np.cos(2.0*theta)
            da_dtheta = (sint * cost * ((1. / ysig2) - (1. / xsig2)))
            db_dtheta = (cos2t / xsig2) - (cos2t / ysig2)            
            dc_dtheta = -da_dtheta 
            dg_dtheta = -g * (1+beta4*z2+0.5*beta6*z2**2)*(da_dtheta*xdiff2+db_dtheta*xdiff*ydiff+dc_dtheta*ydiff2) / gxy     
            derivative.append(np.sum(np.sum(dg_dtheta,axis=0),axis=0)/osamp2)
        if nderiv>=7:            
            dg_dbeta4 = -g * (0.5*z2**2) / gxy
            derivative.append(np.sum(np.sum(dg_dbeta4,axis=0),axis=0)/osamp2)
        if nderiv>=8:            
            dg_dbeta6 = -g * ((1.0/6.0)*z2**3) / gxy
            derivative.append(np.sum(np.sum(dg_dbeta6,axis=0),axis=0)/osamp2)
            
        g = np.sum(np.sum(g,axis=0),axis=0)/osamp2

        # Reshape
        if ndim>1:
            g = g.reshape(shape)
            derivative = [d.reshape(shape) for d in derivative]
            
        return g,derivative

    # No derivative
    else:
        g = np.sum(np.sum(g,axis=0),axis=0)/osamp2
        # Reshape
        if ndim>1:
            g = g.reshape(shape)
        return g


def empirical(x, y, pars, mpars, mcube, deriv=False, nderiv=None):
    """Empirical look-up table"""
    npars = len(pars)
    if mcube.ndim==2:
        nypsf,nxpsf = mcube.shape   # python images are (Y,X)
        nel = 1
    else:
        nypsf,nxpsf,nel = mcube.shape

    # Parameters for the profile
    amp = pars[0]
    x0 = pars[1]
    y0 = pars[2]

    # Center of image
    xcenter = mpars[0]
    ycenter = mpars[1]
        
    # Relative positions
    dx = x - x0
    dy = y - y0
    
    # Constant component
    # Interpolate to the X/Y values
    f_psf = RectBivariateSpline(np.arange(nxpsf)-nxpsf//2, np.arange(nypsf)-nypsf//2, mcube[:,:,0],kx=3,ky=3,s=0)
    cterm = f_psf(dx,dy,grid=False)
    g = cterm.copy()
    
    # Spatially-varying component
    if nel>1:
        reldeltax = (x0-xcenter)/(2*xcenter)
        reldeltay = (y0-ycenter)/(2*ycenter) 
        # X-term
        
        
        # X*Y-term

        # Y-term
    
    
    if deriv is True:
        derivative = []
        derivative.append( g/amp )
        derivative.append( np.gradient(g, axis=(0,1)) )

        return g,derivative
            
    # No derivative
    else:        
        return g


def psfmodel(name,pars=None,**kwargs):
    """ Select PSF model based on the name."""
    if str(name).lower() in _models.keys():
        return _models[str(name).lower()](pars,**kwargs)
    else:
        raise ValueError('PSF type '+str(name)+' not supported.  Select '+', '.join(_models.keys()))
    

#######################
# PSF classes
#######################


# PSF base class
class PSFBase:

    def __init__(self,mpars,npix=101,binned=False,verbose=False):
        """
        Initialize the PSF model object.

        Parameters
        ----------
        mpars : numpy array
          PSF model parameters array.
        npix : int, optional
          Number of pixels to model [Npix,Npix].  Must be odd. Default is 101 pixels.
        binned : boolean, optional
        
        verbose : boolean, optional
          Verbose output when performing operations.  Default is False.

        """
        # npix must be odd
        if npix%2==0: npix += 1
        self._params = np.atleast_1d(mpars)
        self.binned = binned
        self.npix = npix
        self.radius = npix//2
        self.verbose = verbose
        self.niter = 0
        self._bounds = None
        self._unitfootflux = None  # unit flux in footprint
        
        # add a precomputed circular mask here to mask out the corners??


        
    # ADD AN EMPIRICAL ATTRIBUTE THAT SPECIFIES IF THERE IS AN EMPIRICAL
    # LOOK-UP TABLE, separate parameter for what variation level (constant or linear)
        
    @property
    def params(self):
        """ Return the PSF model parameters."""
        return self._params

    @params.setter
    def params(self,value):
        self._params = value

    def starbbox(self,coords,imshape,radius=None):
        """
        Return the boundary box for a star given radius and image size.
        
        Parameters
        ----------
        coords: list or tuple
           Central coordinates (xcen,ycen) of star (*absolute* values).
        imshape: list or tuple
            Image shape (ny,nx) values.  Python images are (Y,X).
        radius: float, optional
            Radius in pixels.  Default is psf.npix//2.
        
        Returns
        -------
        bbox : BoundingBox object
          Bounding box of the x/y ranges.
          Upper values are EXCLUSIVE following the python convention.

        """

        # Star coordinates
        xcen,ycen = coords
        ny,nx = imshape   # python images are (Y,X)
        if radius is None:
            radius = self.npix//2
        xlo = np.maximum(int(np.floor(xcen-radius)),0)
        xhi = np.minimum(int(np.ceil(xcen+radius+1)),nx)
        ylo = np.maximum(int(np.floor(ycen-radius)),0)
        yhi = np.minimum(int(np.ceil(ycen+radius+1)),ny)
        
        return BoundingBox(xlo,xhi,ylo,yhi)
        
    
    def bbox2xy(self,bbox):
        """
        Convenience method to convert boundary box of X/Y limits to 2-D X and Y arrays.  The upper limits
        are EXCLUSIVE following the python convention.
        """
        if isinstance(bbox,BoundingBox):
            x0,x1 = bbox.xrange
            y0,y1 = bbox.yrange
        else:
            x0,x1 = bbox[0]
            y0,y1 = bbox[1]            
        dx = np.arange(x0,x1)
        nxpix = len(dx)
        dy = np.arange(y0,y1)
        nypix = len(dy)
        # Python images are (Y,X)
        x = dx.reshape(1,-1)+np.zeros(nypix,int).reshape(-1,1)   # broadcasting is faster
        y = dy.reshape(-1,1)+np.zeros(nxpix,int)     
        return x,y
        
    def __call__(self,x=None,y=None,pars=None,mpars=None,bbox=None,deriv=False,**kwargs):
        """
        Generate a model PSF for the input X/Y value and parameters.  If no inputs
        are given, then a postage stamp PSF image is returned.

        Parameters
        ----------
        x and y: numpy array, optional
            The X and Y values for the images pixels for which you want to
            generate the model. These can be 1D or 2D arrays.
            The "bbox" parameter can be used instead of "x" and "y" if a rectangular
            region is desired.
        pars : numpy array
            Model parameter values [height, xcen, ycen, sky].
        bbox: list or BoundingBox
            Boundary box giving range in X and Y for a rectangular region to generate the model.
            This can be BoundingBox object or a 2x2 list/tuple [[x0,x1],[y0,y1]].
            Upper values are EXCLUSIVE following the python convention.
        mpars : numpy array, optional
            PSF model parameters to use.  The default behavior is to use the
            model pararmeters of the PSFobject.
        deriv : boolean, optional
            Return the derivative (Jacobian) as well as the model (model, deriv).
        binned : boolean, optional
            Sum the flux across a pixel.  This is normally set when the PSF
            object is initialized, but can be modified directly in the call.

        Returns
        -------
        model : numpy array
          Array of (1-D) model values for the input xdata and parameters.

        Example
        -------

        m = psf(x,y,pars)

        """

        # Nothing input, PSF postage stamp
        if x is None and y is None and pars is None and bbox is None:
            pars = [1.0, self.npix//2, self.npix//2]
            pix = np.arange(self.npix)
            # Python images are (Y,X)
            y = pix.reshape(-1,1)+np.zeros(self.npix,int)     # broadcasting is faster
            x = y.copy().T

        # Get coordinates from BBOX
        if x is None and y is None and bbox is not None:
            x,y = self.bbox2xy(bbox)

        if x is None or y is None:
            raise ValueError("X and Y or BBOX must be input")
        if pars is None:
            raise ValueError("PARS must be input")
        if len(pars)<3 or len(pars)>4:
            raise ValueError("PARS must have 3 or 4 elements")
        
        # Make sure they are numpy arrays
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)             

        # No model parameters input, use saved ones
        if mpars is None: mpars = self.params               

        # Sky value input
        sky = None
        if len(pars)==4:
            sky = pars[3]
            pars = pars[0:3]
        
        # Make parameters for the function, STELLAR + MODEL parameters
        inpars = np.hstack((pars,mpars))

        # Evaluate
        out = self.evaluate(x,y,inpars,deriv=deriv,**kwargs)

        # Mask any corner pixels
        rr = np.sqrt((x-inpars[1])**2+(y-inpars[2])**2)
        out[rr>self.radius] = 0
        
        # Add sky to model
        if sky is not None:
            # With derivative
            if deriv is True:
                modelim = out[0]
                modelim += sky
                out = (modelim,)+out[1:]
            else:
                out += sky

        return out


    def model(self,xdata,*args,allpars=False,**kwargs):
        """
        Function to use with curve_fit() to fit a single stellar profile.

        Parameters
        ----------
        xdata : numpy array
            X and Y values in a [2,N] array.
        args : float
            Model parameter values as separate positional input parameters,
            [height, xcen, ycen, sky].  If allpars=True, then the model
            parameters are added at the end, i.e. 
            [height, xcen, ycen, sky, model parameters].
        allpars : boolean, optional
            PSF model parameters have been input as well (behind the stellar
            parameters).  Default is False.

        Returns
        -------
        model : numpy array
          Array of (1-D) model values for the input xdata and parameters.

        Example
        -------

        m = psf.model(xdata,*pars)

        """
        # PARS should be [height,x0,y0,sky]
        ## curve_fit separates each parameter while
        ## psf expects on pars array
        
        pars = args
        if self.verbose: print('model: ',pars)

        self.niter += 1
        
        # Just stellar parameters
        if not allpars:
            return self(xdata[0],xdata[1],pars,**kwargs)
        
        # Stellar + Model parameters
        #   PARS should be [height,x0,y0,sky, model parameters]
        else:
            allpars = args
            nmpars = len(self.params)
            npars = len(allpars)-nmpars
            pars = allpars[0:npars]
            mpars = allpars[-nmpars:]        
            return self(xdata[0],xdata[1],pars,mpars=mpars,**kwargs)


    def jac(self,xdata,*args,retmodel=False,allpars=False,**kwargs):
        """
        Method to return Jacobian matrix.

        Parameters
        ----------
        xdata : numpy array
            X and Y values in a [2,N] array.
        args : float
            Model parameter values as separate positional input parameters,
            [height, xcen, ycen, sky]. If allpars=True, then the model
            parameters are added at the end, i.e. 
            [height, xcen, ycen, sky, model parameters].
        retmodel : boolean, optional
            Return the model as well.  Default is retmodel=False.
        allpars : boolean, optional
            PSF model parameters have been input as well (behind the stellar
            parameters).  Default is False.

        Returns
        -------
        if retmodel==False
        jac : numpy array
          Jacobian matrix of partial derivatives [N,Npars].
        model : numpy array
          Array of (1-D) model values for the input xdata and parameters.
          If retmodel==True, then (model,jac) are returned.

        Example
        -------

        jac = psf.jac(xdata,*pars)

        """

        # CAN WE MODIFY THIS TO ALLOW FOR MULTIPLE STAR PARAMETERS
        # TO BE INPUT AS A CATALOG
        # So produce a model for a group of stars?
        
        # PARS should be [height,x0,y0,sky]        
        ## curve_fit separates each parameter while
        ## psf expects on pars array
        pars = np.array(args)
        if self.verbose: print('jac: ',pars)
        # Make parameters for the function, STELLAR + MODEL parameters
        #   *without* sky
        if not allpars:
            if len(pars)==3:
                inpars = np.hstack((pars,self.params))
                sky = None
            elif len(pars)==4:
                # drop sky which is the fourth parameter
                inpars = np.hstack((pars[0:3],self.params))
                sky = pars[3]
            else:
                raise ValueError('PARS must have 3 or 4 parameters')
            nderiv = 3
        # All parameters
        else:
            nmpars = len(self.params)
            mpars = pars[-nmpars:]
            npars = len(pars)-nmpars
            if npars==4:  # sky input, drop it
                sky = pars[3]
                inpars = np.hstack((pars[0:3],mpars)) 
            else:
                inpars = pars.copy()
                sky = None
            nderiv = None  # want all the derivatives
        # Get the derivatives
        if retmodel:  # want model as well
            m,deriv = self.evaluate(xdata[0],xdata[1],inpars,deriv=True,nderiv=nderiv,**kwargs)
            if sky is not None: m += sky  # add sky
        else:
            deriv = self.deriv(xdata[0],xdata[1],inpars,nderiv=nderiv,**kwargs)
        deriv = np.array(deriv).T
        # Initialize jacobian matrix
        #   the parameters are [height,xmean,ymean,sky]
        #   if allpars, parameters are [height,xmean,ymean,sky,model parameters]
        jac = np.zeros((len(xdata[0]),len(pars)),float)
        if sky is None:
            jac[:,:] = deriv
        else:  # add sky derivative in
            jac[:,0:3] = deriv[:,0:3]
            jac[:,3] = 1
            jac[:,4:] = deriv[:,3:]
            
        # Return
        if retmodel:   # return model as well
            return m,jac
        else:
            return jac
    
    def modelall(self,xdata,*args,**kwargs):
        """ Convenience function to use with curve_fit() to fit all parameters of a single stellar profile."""
        # PARS should be [height,x0,y0,sky, model parameters]
        return self.model(xdata,*args,allpars=True,**kwargs)
        
    def jacall(self,xdata,*args,retmodel=False,**kwargs):
        """ Convenience function to use with curve_fit() to fit all parameters of a single stellar profile."""
        # PARS should be [height,x0,y0,sky, model parameters]        
        return self.jac(xdata,*args,allpars=True,**kwargs)

    
    def fit(self,im,pars,niter=2,radius=None,allpars=False,method='qr',nosky=False,
            minpercdiff=0.5,absolute=False,retpararray=False,retfullmodel=False,
            bounds=None,verbose=False):
        """
        Method to fit a single star using the PSF model.

        Parameters
        ----------
        im : CCDData object
            Image to use for fitting.
        pars : numpy array, list or catalog
            Initial parameters.  If numpy array or list the values should be [height, xcen, ycen].
            If a catalog is input then it must at least the "x" and "y" columns.
        niter : int, optional
            Number of iterations to perform.  Default is 2.
        radius : float, optional
            Fitting radius in pixels.  Default is to use the PSF FWHM.
        allpars : boolean, optional
            Fit PSF model parameters as well.  Default is to only fit the stellar parameters
            of [height, xcen, ycen, sky].
        method : str, optional
            Method to use for solving the non-linear least squares problem: "cholesky",
            "qr", "svd", and "curve_fit".  Default is "qr".
        minpercdiff : float, optional
           Minimum percent change in the parameters to allow until the solution is
           considered converged and the iteration loop is stopped.  Default is 0.5.
        nosky : boolean, optional
            Do not fit the sky, only [height, xcen, and ycen].  Default is False.
        weight : boolean, optional
            Weight the data by 1/error**2.  Default is weight=True.
        absolute : boolean, optional
            Input and output coordinates are in "absolute" values using the image bounding box.
              Default is False, everything is relative.
        retpararray : boolean, optional
            Return best-fit parameter values as an array.  Default is to return parameters
              as a catalog.
        retfullmodel : boolean, optional
            Return model over the full PSF region.  Default is False.
        bounds : list, optional
            Input lower and upper bounds/constraints on the fitting parameters (tuple of two
              lists (e.g., ([height_lo,x_low,y_low],[height_hi,x_hi,y_hi])).
        verbose : boolean, optional
            Verbose output to the screen.  Default is False.

        Returns
        -------
        outcat : catalog or numpy array
            Output catalog of best-fit values (id, height, height_error, x, x_error, y, y_error,
              sky, sky_error, niter).  If retpararray=True, then the parameters and parameter
              uncertainties will be output as numpy arrays.
        perror : numpy array
            Array of uncertainties of the best-fit values.  Only if retpararray=True is set.
        model : CCDData object
            The best-fitting model. This only includes the model for the region that was used
              in the fit.  To return the model for the full image set retfullmodel=True.
        mpars : numpy array
            Best-fit model parameter values.  Only if allpars=True and retpararray=False are set.

        Example
        -------

        outcat,model = psf.fit(image,[1002.0,520.0,734.0])

        or

        pars,perror,model = psf.fit(image,[1002.0,520.0,734.0],retpararray=True)

        or

        outcat,model,mpars = psf.fit(image,[1002.0,520.0,734.0],allpars=True)


        """
        
        # PARS: initial guesses for Xo and Yo parameters.
        if isinstance(pars,Table):
            for n in ['x','y','height']:
                if n not in pars.columns:
                    raise ValueError('PARS must have [HEIGHT, X, Y]')
            cat = {'height':pars['height'][0],'x':pars['x'][0],'y':pars['y'][0]}
        elif isinstance(pars,np.ndarray):
            for n in ['x','y','height']:
                if n not in pars.dtype.names:
                    raise ValueError('PARS must have [HEIGHT, X, Y]')            
            cat = {'height':pars['height'][0],'x':pars['x'][0],'y':pars['y'][0]}
        elif isinstance(pars,dict):
            if 'x' in pars.keys()==False or 'y' in pars.keys() is False:
                raise ValueError('PARS dictionary must have x and y')
            cat = pars
        else:            
            if len(pars)<3:
                raise ValueError('PARS must have [HEIGHT, X, Y]')
            cat = {'height':pars[0],'x':pars[1],'y':pars[2]}

        method = str(method).lower()

        # Input bounds
        if bounds is not None:
            inbounds = copy.deepcopy(bounds)
        else:
            inbounds = None
        
        # Image offset for absolute X/Y coordinates
        if absolute:
            imx0 = im.bbox.xrange[0]
            imy0 = im.bbox.yrange[0]

        xc = cat['x']
        yc = cat['y']
        if absolute:  # offset
            xc -= imx0
            yc -= imy0
            if bounds is not None:
                bounds[0][1] -= imx0  # lower
                bounds[0][2] -= imy0
                bounds[1][1] -= imx0  # upper
                bounds[1][2] -= imy0                
        if radius is None:
            radius = np.maximum(self.fwhm(),1)
        bbox = self.starbbox((xc,yc),im.shape,radius)
        X,Y = self.bbox2xy(bbox)

        # Get subimage of pixels to fit
        # xc/yc might be offset
        flux = im.data[bbox.slices]
        err = im.error[bbox.slices]
        wt = 1.0/np.maximum(err,1)**2  # weights
        skyim = im.sky[bbox.slices]
        xc -= bbox.ixmin  # offset for the subimage
        yc -= bbox.iymin
        X -= bbox.ixmin
        Y -= bbox.iymin
        if bounds is not None:
            bounds[0][1] -= bbox.ixmin  # lower
            bounds[0][2] -= bbox.iymin
            bounds[1][1] -= bbox.ixmin  # upper
            bounds[1][2] -= bbox.iymin            
        xdata = np.vstack((X.ravel(), Y.ravel()))        
        sky = np.median(skyim)
        if nosky: sky=0.0
        height = flux[int(np.round(yc)),int(np.round(xc))]-sky   # python images are (Y,X)
        initpar = [height,xc,yc,sky]            
        
        # Fit PSF parameters as well
        if allpars:
            initpar = np.hstack(([height,xc,yc,sky],self.params.copy()))

        # Remove sky column
        if nosky:
            initpar = np.delete(initpar,3,axis=0)

        # Initialize the output catalog
        dt = np.dtype([('id',int),('height',float),('height_error',float),('x',float),
                       ('x_error',float),('y',float),('y_error',float),('sky',float),
                       ('sky_error',float),('flux',float),('flux_error',float),
                       ('mag',float),('mag_error',float),('niter',int),
                       ('nfitpix',int),('rms',float),('chisq',float)])
        outcat = np.zeros(1,dtype=dt)
        outcat['id'] = 1

        # Make bounds
        if bounds is None:
            bounds = self.mkbounds(initpar,flux.shape)
        
        # Curve_fit
        if method=='curve_fit':
            self.niter = 0
            if allpars==False:
                bestpar,cov = curve_fit(self.model,xdata,flux.ravel(),sigma=err.ravel(),
                                        p0=initpar,jac=self.jac,bounds=bounds)
                perror = np.sqrt(np.diag(cov))
                model = self.model(xdata,*bestpar)
                count = self.niter
                
            # Fit all parameters
            else:
                bestpar,cov = curve_fit(self.modelall,xdata,flux.ravel(),sigma=err.ravel(),
                                        p0=initpar,jac=self.jacall)
                perror = np.sqrt(np.diag(cov))
                model = self.modelall(xdata,*bestpar)
                count = self.niter

        # All other methods:
        else:
            # Iterate
            count = 0
            bestpar = initpar.copy()
            maxpercdiff = 1e10
            maxsteps = self.steps(initpar,bounds,star=True)  # maximum steps
            while (count<niter and maxpercdiff>minpercdiff):
                # Use Cholesky, QR or SVD to solve linear system of equations
                if allpars:
                    m,jac = self.jac(xdata,*bestpar,allpars=True,retmodel=True)
                else:
                    m,jac = self.jac(xdata,*bestpar,retmodel=True)
                dy = flux.ravel()-m.ravel()
                # Solve Jacobian
                dbeta = lsq.jac_solve(jac,dy,method=method,weight=wt.ravel())
                dbeta[~np.isfinite(dbeta)] = 0.0  # deal with NaNs
                
                # Update parameters
                oldpar = bestpar.copy()
                # limit the steps to the maximum step sizes and boundaries
                bestpar = self.newpars(bestpar,dbeta,bounds,maxsteps)
                #bestpar += dbeta
                # Check differences and changes
                diff = np.abs(bestpar-oldpar)
                denom = np.maximum(np.abs(oldpar.copy()),0.0001)
                percdiff = diff.copy()/denom*100  # percent differences
                percdiff[1:3] = diff[1:3]*100               # x/y
                maxpercdiff = np.max(percdiff)
                
                if verbose:
                    print('N = ',count)
                    print('bestpars = ',bestpar)
                    print('dbeta = ',dbeta)
                
                count += 1

            # Get covariance and errors
            if allpars:
                model,jac = self.jac(xdata,*bestpar,allpars=True,retmodel=True)
            else:
                model,jac = self.jac(xdata,*bestpar,retmodel=True)
            dy = flux.ravel()-m.ravel()
            cov = lsq.jac_covariance(jac,dy,wt.ravel())
            perror = np.sqrt(np.diag(cov))

        # Offset the final coordinates for the subimage offset
        bestpar[1] += bbox.ixmin
        bestpar[2] += bbox.iymin        
            
        # Image offsets for absolute X/Y coordinates
        if absolute:
            bestpar[1] += imx0
            bestpar[2] += imy0
            
        # Put values in catalog
        outcat['height'] = bestpar[0]
        outcat['height_error'] = perror[0]
        outcat['x'] = bestpar[1]
        outcat['x_error'] = perror[1]
        outcat['y'] = bestpar[2]
        outcat['y_error'] = perror[2]
        if not nosky:
            outcat['sky'] = bestpar[3]
            outcat['sky_error'] = perror[3]
        outcat['flux'] = bestpar[0]*self.fwhm()
        outcat['flux_error'] = perror[0]*self.fwhm()        
        outcat['mag'] = -2.5*np.log10(np.maximum(outcat['flux'],1e-10))+25.0
        outcat['mag_error'] = (2.5/np.log(10))*outcat['flux_error']/outcat['flux']
        outcat['niter'] = count
        outcat['nfitpix'] = flux.size
        outcat['chisq'] = np.sum((flux-model.reshape(flux.shape))**2/err**2)/len(flux)
        # chi value, RMS of the residuals as a fraction of the height
        rms = np.sqrt(np.mean(((flux-model.reshape(flux.shape))/bestpar[0])**2))
        outcat['rms'] = rms
        
        # Return full model
        if retfullmodel:
            bbox = self.starbbox((bestpar[1],bestpar[2]),im.shape)
            model = self(pars=bestpar,bbox=bbox)
            model = CCDData(model,bbox=bbox,unit=im.unit)
        else:
            # Reshape model and make CCDData image with proper bbox
            model = model.reshape(flux.shape)
            model = CCDData(model,bbox=bbox,unit=im.unit)

        # Set input bounds back
        bounds = copy.deepcopy(inbounds)
            
        # Return catalog
        if not retpararray:
            if allpars:
                mpars = bestpar[-len(self.params):]   # model parameters
                return outcat,model,mpars
            else:
                return outcat,model
        # Return parameter array
        else:
            return bestpar,perror,model

    
    def sub(self,im,cat,sky=False,radius=None):
        """
        Method to subtract a single star using the PSF model.

        Parameters
        ----------
        im : CCDData object
            Image to use for fitting.
        cat : catalog
            Catalog of stellar parameters.  Columns must include height, x, y and sky.
        sky : boolean, optional
            Include sky in the model that is subtracted.  Default is False.
        radius : float, optional
            PSF radius to use.  The default is to use the full size of the PSF.

        Returns
        -------
        subim : CCDData object
            Image with stellar models subtracted.

        Example
        -------

        subim = psf.sub(image,cat)


        """

        if isinstance(cat,np.ndarray):
            columns = cat.dtype.names
        elif isinstance(cat,dict):
            columns = cat.keys()
        elif isinstance(cat,Table):
            columns = cat.columns
        else:
            raise ValueError('Only ndarray, astropy Table or dictionaries supported for catalogs')

        for n in ['height','x','y','sky']:
            if not n in columns:
                raise ValueError('Catalog must have height, x, y and sky columns')
            
        ny,nx = im.shape    # python images are (Y,X)
        nstars = np.array(cat).size
        hpix = self.npix//2
        if radius is None:
            radius = self.radius
        else:
            radius = np.minimum(self.radius,radius)
        subim = im.data.copy()
        for i in range(nstars):
            pars = [cat['height'][i],cat['x'][i],cat['y'][i]]
            if sky:
                pars.append(cat['sky'][i])
            bbox = self.starbbox((pars[1],pars[2]),im.shape,radius)
            #x0 = int(np.maximum(0,np.floor(pars[1]-radius)))
            #x1 = int(np.minimum(np.ceil(pars[1]+radius),nx))
            #y0 = int(np.maximum(0,np.floor(pars[2]-radius)))
            #y1 = int(np.minimum(np.ceil(pars[2]+radius),ny))
            #bbox = [[x0,x1],[y0,y1]]
            im1 = self(pars=pars,bbox=bbox)
            #subim[x0:x1+1,y0:y1+1] -= im1
            subim[bbox.slices] -= im1            
        return subim
                    
    
    def __str__(self):
        return self.__class__.__name__+'('+str(list(self.params))+',binned='+str(self.binned)+')'

    def __repr__(self):
        return self.__class__.__name__+'('+str(list(self.params))+',binned='+str(self.binned)+')'        

    @property
    def unitfootflux(self):
        """ Return the unit flux inside the footprint."""
        if self._unitfootflux is None:
            self._unitfootflux = np.sum(self()) # sum up footprint flux
        return self._unitfootflux 
            
    def fwhm(self):
        """ Return the FWHM of the model function. Must be defined by subclass"""
        pass

    def flux(self,pars=None,footprint=False):
        """ Return the flux/volume of the model given the height.  Must be defined by subclass."""
        pass

    # Do we also want the flux within the footprint!
    # could calculate the unit flux within the footprint the first time it's
    # called and save that.
    # could use fluxtot for the total flux
    # or fluxfoot for the footprint flux
    # or even have footprint=True to use the footprint flux
    
    def steps(self,pars=None,bounds=None,star=False):
        """ Return step sizes to use when fitting the PSF model parameters (at least initial sizes)."""
        # star=True indicates that we have stellar parameters
        # Check the initial steps against the parameters to make sure that don't
        #   go past the boundaries
        if pars is None:
            pars = self.params
        if bounds is None:
            bounds = self.mkbounds(pars)
        npars = len(pars)
        nmpars = len(self.params)
        # Either
        # 1) model parameters only (npars==mpars and star==False)
        # 2) star parameters only (star==True)
        # 3) star + model parameters (npars>mpars)
        # Have stellar parameters
        if npars>nmpars or star:
            initsteps = np.zeros(npars,float)
            initsteps[0:3] = [pars[0]*0.5,0.5,0.5]  # amp, x, y
            if npars-nmpars==4 or npars==4:  # with sky
                initsteps[3] = pars[3]*0.5          # sky
            # we also have model parameters
            if npars>4 or star==False:
                initsteps[-nmpars:] = self._steps       # model parameters
        # Only model parameters
        else:
            initsteps = self._steps
        # Now compare to the boundaries
        # NOTE: it's okay if the step crosses the boundary
        #   newpars() can figure out that it needs to limit
        #   any new parameter value at the boundary
        lcheck = self.checkbounds(pars-initsteps,bounds)
        ucheck = self.checkbounds(pars+initsteps,bounds)
        # Final steps
        fsteps = initsteps.copy()
        # bad negative step, crosses lower boundary
        badneg = (lcheck!=0)
        nbadneg = np.sum(badneg)
        # reduce the step sizes until they are within bounds
        maxiter = 2
        count = 0
        while (np.sum(badneg)>0 and count<=maxiter):
            fsteps[badneg] /= 2
            lcheck = self.checkbounds(pars-fsteps,bounds)
            badneg = (lcheck!=0)
            count += 1
            
        # bad positive step, crosses upper boundary
        badpos = (ucheck!=0)
        # reduce the steps sizes until they are within bounds
        count = 0
        while (np.sum(badpos)>0 and count<=maxiter):
            fsteps[badpos] /= 2
            ucheck = self.checkbounds(pars+fsteps,bounds)
            badpos = (ucheck!=0)            
            count += 1
            
        return fsteps
    
    @property
    def bounds(self):
        """ Return the lower and upper bounds for the parameters."""
        return self._bounds

    def mkbounds(self,pars,imshape):
        """ Make bounds for a set of input parameters."""

        npars = len(pars)
        nmpars = len(self.params)
        
        # figure out if nosky is set
        if (npars==3) or (npars-nmpars==3):
            nosky = True
        else:
            nosky = False
        ny,nx = imshape
        
        # Make bounds
        lbounds = np.zeros(npars,float)
        ubounds = np.zeros(npars,float)
        ubounds[0:3] = [np.inf,nx-1,ny-1]
        if nosky==False:
            lbounds[3] = -np.inf
            ubounds[3] = np.inf
        if npars>4:
            mlbounds,mubounds = self.bounds
            if nosky:
                 lbounds[3:] = mlbounds
                 ubounds[3:] = mubounds
            else:
                 lbounds[4:] = mlbounds
                 ubounds[4:] = mubounds            
        bounds = (lbounds,ubounds)
                 
        return bounds

    def checkbounds(self,pars,bounds=None):
        """ Check the parameters against the bounds."""
        # 0 means it's fine
        # 1 means it's beyond the lower bound
        # 2 means it's beyond the upper bound
        if bounds is None:
            bounds = self.mkbounds(pars)
        npars = len(pars)
        lbounds,ubounds = bounds
        check = np.zeros(npars,int)
        check[pars<=lbounds] = 1
        check[pars>=ubounds] = 2
        return check
        
    def limbounds(self,pars,bounds=None):
        """ Limit the parameters to the boundaries."""
        if bounds is None:
            bounds = self.mkbounds(pars)
        lbounds,ubounds = bounds
        outpars = np.minimum(np.maximum(pars,lbounds),ubounds)
        return outpars

    def limsteps(self,steps,maxsteps):
        """ Limit the parameter steps to maximum step sizes."""
        signs = np.sign(steps)
        outsteps = np.minimum(np.abs(steps),maxsteps)
        outsteps *= signs
        return outsteps

    def newpars(self,pars,steps,bounds,maxsteps):
        """ Get new parameters given initial parameters, steps and constraints."""

        # Limit the steps to maxsteps
        limited_steps = self.limsteps(steps,maxsteps)
        # Make sure that these don't cross the boundaries
        lbounds,ubounds = bounds
        check = self.checkbounds(pars+limited_steps,bounds)
        # Reduce step size for any parameters to go beyond the boundaries
        badpars = (check!=0)
        # reduce the step sizes until they are within bounds
        newsteps = limited_steps.copy()
        count = 0
        maxiter = 2
        while (np.sum(badpars)>0 and count<=maxiter):
            newsteps[badpars] /= 2
            newcheck = self.checkbounds(pars+newsteps,bounds)
            badpars = (newcheck!=0)
            count += 1
            
        # Final parameters
        newpars = pars + newsteps
            
        # Make sure to limit them to the boundaries
        check = self.checkbounds(newpars,bounds)
        badpars = (check!=0)
        if np.sum(badpars)>0:
            # add a tiny offset so it doesn't fit right on the boundary
            newpars = np.minimum(np.maximum(newpars,lbounds+1e-30),ubounds-1e-30)
        return newpars
    
    def evaluate(self):
        """ Evaluate the function.  Must be defined by subclass."""
        pass

    def deriv(self):
        """ Return the derivate of the function.  Must be defined by subclass."""
        pass
        
    def copy(self):
        """ Create a new copy of this LSF object."""
        return copy.deepcopy(self)        

    @classmethod
    def read(cls,filename):
        """ Load a PSF file."""
        if os.path.exists(filename)==False:
            raise ValueError(filename+' NOT FOUND')
        data,head = fits.getdata(filename,header=True)
        psftype = head.get('PSFTYPE')
        if psftype is None:
            raise ValueError('No PSFTYPE found in header')
        kwargs = {}        
        binned = head.get('BINNED')
        if binned is not None: kwargs['binned'] = binned
        npix = head.get('NPIX')
        if npix is not None: kwargs['npix'] = npix        
        return psfmodel(psftype,data,**kwargs)

    def tohdu(self):
        """ Convert the PSF object to an HDU. Defined by subclass."""
        pass
    
    def write(self,filename,overwrite=True):
        """ Write a PSF to a file.  Defined by subclass."""
        pass

    def thumbnail(self,filename=None,figsize=6):
        """ Generate a thumbnail image of the PSF."""

        if filename is None:
            filename = 'psf.png'
        if os.path.exists(filename): os.remove(filename)

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        font = {'size' : 11}
        matplotlib.rc('font', **font)
        fig, ax = plt.subplots(1,1,figsize=(figsize,0.9*figsize))
        ax1 = ax.imshow(self(),origin='lower')
        ax.set_xlabel('X (pixels')
        ax.set_ylabel('Y (pixels')
        plt.title(str(self))
        cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().y1-ax.get_position().y0])
        fig.colorbar(ax1, cax=cax)
        plt.savefig(filename,bbox_inches='tight')
        plt.close(fig)
    
    
# PSF Gaussian class
class PSFGaussian(PSFBase):

    # Initalize the object
    def __init__(self,mpars=None,npix=101,binned=False):
        # MPARS are the model parameters
        if mpars is None:
            mpars = np.array([1.0,1.0,0.0])
        if len(mpars)!=3:
            raise ValueError('3 parameters required')
        # mpars = [xsigma, ysigma, theta]
        if mpars[0]<=0 or mpars[1]<=0:
            raise ValueError('sigma parameters must be >0')
        super().__init__(mpars,npix=npix,binned=binned)
        # Set the bounds
        self._bounds = (np.array([0.0,0.0,-np.inf]),
                        np.array([np.inf,np.inf,np.inf]))
        # Set step sizes
        self._steps = np.array([0.5,0.5,0.2])
        
    def fwhm(self,pars=None):
        """ Return the FWHM of the model."""
        if pars is None:
            pars = np.hstack(([1.0,0.0,0.0],self.params))
        return gaussian2d_fwhm(pars)

    def flux(self,pars=None,footprint=False):
        """ Return the flux/volume of the model given the height or parameters."""
        if pars is None:
            pars = np.hstack(([1.0, 0.0, 0.0], self.params))
        else:
            pars = np.atleast_1d(pars)
            if pars.size==1:
                pars = np.hstack(([pars[0], 0.0, 0.0], self.params))
        if footprint:
            return self.unitfootflux*pars[0]
        else:
            return gaussian2d_flux(pars)        
    
    def evaluate(self,x, y, pars, binned=None, deriv=False, nderiv=None):
        """Two dimensional Gaussian model function"""
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta]
        if binned is None: binned = self.binned
        if binned is True:
            return gaussian2d_integrate(x, y, pars, deriv=deriv, nderiv=nderiv)
        else:
            return gaussian2d(x, y, pars, deriv=deriv, nderiv=nderiv)
    
    def deriv(self,x, y, pars, binned=None, nderiv=None):
        """Two dimensional Gaussian model derivative with respect to parameters"""
        if binned is None: binned = self.binned        
        if binned is True:
            g, derivative = gaussian2d_integrate(x, y, pars, deriv=True, nderiv=nderiv)
        else:
            g, derivative = gaussian2d(x, y, pars, deriv=True, nderiv=nderiv)
        return derivative            

    def tohdu(self):
        """
        Convert the PSF object to an HDU so it can be written to a file.

        Returns
        -------
        hdu : fits HDU object
          The FITS HDU object.

        Example
        -------

        hdu = psf.tohdu()

        """
        hdu = fits.PrimaryHDU(self.params)
        hdu.header['PSFTYPE'] = 'Gaussian'
        hdu.header['BINNED'] = self.binned
        hdu.header['NPIX'] = self.npix
        return hdu
    
    def write(self,filename,overwrite=True):
        """ Write a PSF to a file."""
        if os.path.exists(filename) and overwrite==False:
            raise ValueError(filename+' already exists and overwrite=False')
        hdulist = fits.HDUList()
        hdulist.append(fits.PrimaryHDU(self.params))
        hdulist[0].header['PSFTYPE'] = 'Gaussian'
        hdulist[0].header['BINNED'] = self.binned
        hdulist[0].header['NPIX'] = self.npix
        hdulist.writeto(filename,overwrite=overwrite)
        hdulist.close()

        
# PSF Moffat class
class PSFMoffat(PSFBase):

    # add separate X/Y sigma values and cross term like in DAOPHOT
    
    
    # Initalize the object
    def __init__(self,mpars=None,npix=101,binned=False):
        # MPARS are model parameters
        # mpars = [xsig, ysig, theta, beta]
        if mpars is None:
            mpars = np.array([1.0,1.0,0.0,2.5])
        if len(mpars)!=4:
            old = np.array(mpars).copy()
            mpars = np.array([1.0,1.0,0.0,2.5])
            mpars[0:len(old)] = old
        if mpars[0]<=0 or mpars[1]<=0:
            raise ValueError('sigma must be >0')
        if mpars[3]<0 or mpars[3]>6:
            raise ValueError('beta must be >0 and <6')
        super().__init__(mpars,npix=npix,binned=binned)
        # Set the bounds
        self._bounds = (np.array([0.0,0.0,-np.inf,0.01]),
                        np.array([np.inf,np.inf,np.inf,6.0]))
        # Set step sizes
        self._steps = np.array([0.5,0.5,0.2,0.2])
        
    def fwhm(self,pars=None):
        """ Return the FWHM of the model."""
        if pars is None:
            pars = np.hstack(([1.0,0.0,0.0],self.params))
        return moffat2d_fwhm(pars)

    def flux(self,pars=None,footprint=False):
        """ Return the flux/volume of the model given the height or parameters."""
        if pars is None:
            pars = np.hstack(([1.0, 0.0, 0.0], self.params))
        else:
            pars = np.atleast_1d(pars)
            if pars.size==1:
                pars = np.hstack(([pars[0], 0.0, 0.0], self.params))
        if footprint:
            return self.unitfootflux*pars[0]
        else: 
            return moffat2d_flux(pars)
    
    def evaluate(self,x, y, pars, binned=None, deriv=False, nderiv=None):
        """Two dimensional Moffat model function"""
        # pars = [amplitude, x0, y0, xsig, ysig, theta, beta]
        if binned is None: binned = self.binned
        if binned is True:
            return moffat2d_integrate(x, y, pars, deriv=deriv, nderiv=nderiv)
        else:
            return moffat2d(x, y, pars, deriv=deriv, nderiv=nderiv)

    def deriv(self,x, y, pars, binned=None, nderiv=None):
        """Two dimensional Moffat model derivative with respect to parameters"""
        if binned is None: binned = self.binned
        if binned is True:
            g, derivative = moffat2d_integrate(x, y, pars, deriv=True, nderiv=nderiv)
        else:
            g, derivative = moffat2d(x, y, pars, deriv=True, nderiv=nderiv)
        return derivative

    def tohdu(self):
        """
        Convert the PSF object to an HDU so it can be written to a file.

        Returns
        -------
        hdu : fits HDU object
          The FITS HDU object.

        Example
        -------

        hdu = psf.tohdu()

        """
        hdu = fits.PrimaryHDU(self.params)
        hdu.header['PSFTYPE'] = 'Moffat'
        hdu.header['BINNED'] = self.binned
        hdu.header['NPIX'] = self.npix
        return hdu
    
    def write(self,filename,overwrite=True):
        """ Write a PSF to a file."""
        if os.path.exists(filename) and overwrite==False:
            raise ValueError(filename+' already exists and overwrite=False')
        hdulist = fits.HDUList()
        hdulist.append(fits.PrimaryHDU(self.params))
        hdulist[0].header['PSFTYPE'] = 'Moffat'
        hdulist[0].header['BINNED'] = self.binned
        hdulist[0].header['NPIX'] = self.npix        
        hdulist.writeto(filename,overwrite=overwrite)
        hdulist.close()
    
    
# PSF Penny class
class PSFPenny(PSFBase):
    """ Gaussian core and Lorentzian wings, only Gaussian is tilted."""
    # PARS are model parameters
    
    # Initalize the object
    def __init__(self,mpars=None,npix=101,binned=False):
        # mpars = [xsig,ysig,theta, relamp,sigma]
        if mpars is None:
            mpars = np.array([1.0,1.0,0.0,0.10,5.0])
        if len(mpars)!=5:
            old = np.array(mpars).copy()
            mpars = np.array([1.0,1.0,0.0,0.10,5.0])
            mpars[0:len(old)] = old
        if mpars[0]<=0:
            raise ValueError('sigma must be >0')
        if mpars[3]<0 or mpars[3]>1:
            raise ValueError('relative amplitude must be >=0 and <=1')
        super().__init__(mpars,npix=npix,binned=binned)
        # Set the bounds
        self._bounds = (np.array([0.0,0.0,-np.inf,0.00,0.0]),
                        np.array([np.inf,np.inf,np.inf,1.0,np.inf]))
        # Set step sizes
        self._steps = np.array([0.5,0.5,0.2,0.1,0.5])
        
    def fwhm(self,pars=None):
        """ Return the FWHM of the model."""
        if pars is None:
            pars = np.hstack(([1.0,0.0,0.0],self.params))
        return penny2d_fwhm(pars)

    def flux(self,pars=None,footprint=False):
        """ Return the flux/volume of the model given the height or parameters."""
        if pars is None:
            pars = np.hstack(([1.0, 0.0, 0.0], self.params))
        else:
            pars = np.atleast_1d(pars)
            if pars.size==1:
                pars = np.hstack(([pars[0], 0.0, 0.0], self.params))
        if footprint:
            return self.unitfootflux*pars[0]
        else:
            return penny2d_flux(pars)
        
    def evaluate(self,x, y, pars=None, binned=None, deriv=False, nderiv=None):
        """Two dimensional Penny model function"""
        # pars = [amplitude, x0, y0, xsig, ysig, theta, relamp, sigma]
        if pars is None: pars = self.params
        if binned is None: binned = self.binned
        if binned is True:
            return penny2d_integrate(x, y, pars, deriv=deriv, nderiv=nderiv)
        else:
            return penny2d(x, y, pars, deriv=deriv, nderiv=nderiv)

    def deriv(self,x, y, pars=None, binned=None, nderiv=None):
        """Two dimensional Penny model derivative with respect to parameters"""
        if pars is None: pars = self.params
        if binned is None: binned = self.binned
        if binned is True:
            g, derivative = penny2d_integrate(x, y, pars, deriv=True, nderiv=nderiv)
        else:
            g, derivative = penny2d(x, y, pars, deriv=True, nderiv=nderiv)
        return derivative

    def tohdu(self):
        """
        Convert the PSF object to an HDU so it can be written to a file.

        Returns
        -------
        hdu : fits HDU object
          The FITS HDU object.

        Example
        -------

        hdu = psf.tohdu()

        """
        hdu = fits.PrimaryHDU(self.params)
        hdu.header['PSFTYPE'] = 'Penny'
        hdu.header['BINNED'] = self.binned
        hdu.header['NPIX'] = self.npix
        return hdu
    
    def write(self,filename,overwrite=True):
        """ Write a PSF to a file."""
        if os.path.exists(filename) and overwrite==False:
            raise ValueError(filename+' already exists and overwrite=False')
        hdulist = fits.HDUList()
        hdulist.append(fits.PrimaryHDU(self.params))
        hdulist[0].header['PSFTYPE'] = 'Penny'
        hdulist[0].header['BINNED'] = self.binned
        hdulist[0].header['NPIX'] = self.npix        
        hdulist.writeto(filename,overwrite=overwrite)
        hdulist.close()

       
# PSF Ellipower class
class PSFGausspow(PSFBase):
    """ DoPHOT PSF, sum of Gaussian ellipses."""
    # PARS are model parameters
    
    # Initalize the object
    def __init__(self,mpars=None,npix=101,binned=False):
        # mpars = [sigx,sigy,sigxy,beta4,beta6]
        if mpars is None:
            mpars = np.array([1.0,2.5,2.5,0.0,1.0,1.0])
        if len(mpars)!=5:
            old = np.array(mpars).copy()
            mpars = np.array([1.0,2.5,2.5,0.0,1.0,1.0])
            mpars[0:len(old)] = old
        if mpars[0]<=0 or mpars[1]<=0:
            raise ValueError('sigma parameters must be >0')        
        super().__init__(mpars,npix=npix,binned=binned)
        # Set the bounds
        #  should be allow negative values for beta4 and beta6??
        self._bounds = (np.array([0.0,0.0,-np.inf,0.00,0.0]),
                        np.array([np.inf,np.inf,np.inf,np.inf,np.inf]))
        # Set step sizes
        self._steps = np.array([0.5,0.5,0.2,0.1,0.1])
        
    def fwhm(self,pars=None):
        """ Return the FWHM of the model."""
        if pars is None:
            pars = np.hstack(([1.0,0.0,0.0],self.params))
        return gausspow2d_fwhm(pars)

    def flux(self,pars=None,footprint=False):
        """ Return the flux/volume of the model given the height or parameters."""
        if pars is None:
            pars = np.hstack(([1.0, 0.0, 0.0], self.params))
        else:
            pars = np.atleast_1d(pars)
            if pars.size==1:
                pars = np.hstack(([pars[0], 0.0, 0.0], self.params))
        if footprint:
            return self.unitfootflux*pars[0]
        else:              
            return gausspow2d_flux(pars)
        
    def evaluate(self,x, y, pars=None, binned=None, deriv=False, nderiv=None):
        """Two dimensional DoPHOT Gausspow model function"""
        # pars = [amplitude, x0, y0, xsig, ysig, theta, relamp, sigma]
        if pars is None: pars = self.params
        if binned is None: binned = self.binned
        if binned is True:
            return gausspow2d_integrate(x, y, pars, deriv=deriv, nderiv=nderiv)
        else:
            return gausspow2d(x, y, pars, deriv=deriv, nderiv=nderiv)

    def deriv(self,x, y, pars=None, binned=None, nderiv=None):
        """Two dimensional DoPHOT Gausspow model derivative with respect to parameters"""
        if pars is None: pars = self.params
        if binned is None: binned = self.binned
        if binned is True:
            g, derivative = gausspow2d_integrate(x, y, pars, deriv=True, nderiv=nderiv)
        else:
            g, derivative = gausspow2d(x, y, pars, deriv=True, nderiv=nderiv)
        return derivative

    def tohdu(self):
        """
        Convert the PSF object to an HDU so it can be written to a file.

        Returns
        -------
        hdu : fits HDU object
          The FITS HDU object.

        Example
        -------

        hdu = psf.tohdu()

        """
        hdu = fits.PrimaryHDU(self.params)
        hdu.header['PSFTYPE'] = 'Gausspow'
        hdu.header['BINNED'] = self.binned
        hdu.header['NPIX'] = self.npix
        return hdu
    
    def write(self,filename,overwrite=True):
        """ Write a PSF to a file."""
        if os.path.exists(filename) and overwrite==False:
            raise ValueError(filename+' already exists and overwrite=False')
        hdulist = fits.HDUList()
        hdulist.append(fits.PrimaryHDU(self.params))
        hdulist[0].header['PSFTYPE'] = 'Gausspow'
        hdulist[0].header['BINNED'] = self.binned
        hdulist[0].header['NPIX'] = self.npix        
        hdulist.writeto(filename,overwrite=overwrite)
        hdulist.close()
    
        
    
class PSFEmpirical(PSFBase):
    """ Empirical look-up table PSF, can vary spatially."""

    # Initalize the object
    def __init__(self,mpars=None,npix=101):
        if mpars is None:
            raise ValueError('Must input images')
        # MPARS should be a two-element tuple with (parameters, psf cube)
        self.cube = mpars[1]
        ny,nx,npars = cube.shape    # Python images are (Y,X)
        super().__init__(mpars[0],npix=npix)
        self._bounds = None
        self._steps = None

    def fwhm(self):
        """ Return the FWHM of the model."""
        pass
        
    def evaluate(self,x, y, pars=None, cube=None, deriv=False, nderiv=None):
        """Empirical look-up table"""
        # pars = [amplitude, x0, y0, sigma, beta]
        if pars is None: pars = self.params
        if cube is None: cube = self.cube
        return empirical(x, y, pars, cube=cube, deriv=deriv, nderiv=nderiv)

    def deriv(self,x, y, pars=None, cube=None, nderiv=None):
        """Empirical look-up table derivative with respect to parameters"""
        if pars is None: pars = self.params
        if cube is None: cube = self.cube
        return empirical(x, y, pars, cube=cube, nderiv=nderiv)   

    def tohdu(self):
        """
        Convert the PSF object to an HDU so it can be written to a file.

        Returns
        -------
        hdu : fits HDU object
          The FITS HDU object.

        Example
        -------

        hdu = psf.tohdu()

        """
        hdu = fits.PrimaryHDU(self.params)
        hdu.header['PSFTYPE'] = 'Empirical'
        hdu.header['BINNED'] = self.binned
        hdu.header['NPIX'] = self.npix
        return hdu
    
    def write(self,filename,overwrite=True):
        """ Write a PSF to a file."""
        if os.path.exists(filename) and overwrite==False:
            raise ValueError(filename+' already exists and overwrite=False')
        hdulist = fits.HDUList()
        hdulist.append(fits.PrimaryHDU(self.params))
        hdulist[0].header['PSFTYPE'] = 'Empirical'
        hdulist[0].header['BINNED'] = self.binned
        hdulist[0].header['NPIX'] = self.npix        
        hdulist.writeto(filename,overwrite=overwrite)
        hdulist.close()


# Read function
read = PSFBase.read    
    
# List of models
_models = {'gaussian':PSFGaussian,'moffat':PSFMoffat,'penny':PSFPenny,'gausspow':PSFGausspow,'empirical':PSFEmpirical}

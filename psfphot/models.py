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
from . import getpsf

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

    # Use Error function

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


def empirical(x, y, pars, mpars, mcube, deriv=False, nderiv=None):
    """Empirical look-up table"""
    npars = len(pars)
    if mcube.ndim==2:
        nxpsf,nypsf = mcube.shape
        nel = 1
    else:
        nxpsf,nypsf,nel = mcube.shape

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

        # add a precomputed circular mask here to mask out the corners??
        
    @property
    def params(self):
        """ Return the PSF model parameters."""
        return self._params

    @params.setter
    def params(self,value):
        self._params = value

    
    def xylim2xy(self,xy):
        """
        Convenience method to convert 2x2 list of X/Y limits [[X0,X1],[Y0,Y1]]
        to 2-D X and Y arrays.  The upper limits are inclusive.
        """
        x0,x1 = xy[0]
        y0,y1 = xy[1]
        dx = np.arange(x0,x1+1)
        nxpix = len(dx)
        dy = np.arange(y0,y1+1)
        nypix = len(dy)
        x = dx.reshape(-1,1)+np.zeros(nypix,int)   # broadcasting is faster
        y = dy.reshape(1,-1)+np.zeros(nxpix,int).reshape(-1,1)     
        return x,y
        
    def __call__(self,x=None,y=None,pars=None,mpars=None,xy=None,deriv=False,**kwargs):
        """
        Generate a model PSF for the input X/Y value and parameters.  If no inputs
        are given, then a postage stamp PSF image is returned.

        Parameters
        ----------
        x and y: numpy array, optional
            The X and Y values for the images pixels for which you want to
            generate the model. These can be 1D or 2D arrays.
            The "xy" parameter can be used instead of "x" and "y" if a rectangular
            region is desired.
        pars : numpy array
            Model parameter values [height, xcen, ycen, sky].
        xy: list
            Limits in X and Y for a rectangular region to generate the model.
              XY = [[X0,X1],[Y0,Y1]]
            X/Y and XY are absolute pixel values, NOT relative ones.
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
        if x is None and y is None and pars is None and xy is None:
            pars = [1.0, self.npix//2, self.npix//2]
            pix = np.arange(self.npix)
            x = pix.reshape(-1,1)+np.zeros(self.npix)  # broadcasting is faster
            y = x.copy().T
            #x = np.repeat(pix,self.npix).reshape(self.npix,self.npix)
            #y = np.repeat(pix,self.npix).reshape(self.npix,self.npix).T

        # Get coordinates from XY
        if x is None and y is None and xy is not None:
            x,y = self.xylim2xy(xy)
            #x0,x1 = xy[0]
            #y0,y1 = xy[1]
            #dx = np.arange(x0,x1+1).astype(float)
            #nxpix = len(dx)
            #dy = np.arange(y0,y1+1).astype(float)
            #nypix = len(dy)
            #x = dx.reshape(-1,1)+np.zeros(nypix)   # broadcasting is faster
            #y = dy.reshape(1,-1)+np.zeros(nxpix).reshape(-1,1) 

        if x is None or y is None:
            raise ValueError("X and Y or XY must be input")
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

    
    def fit(self,im,pars,niter=1,radius=None,allpars=False,method='qr',nosky=False,weight=True):
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
            Number of iterations to perform.  Default is 1.
        radius : float, optional
            Fitting radius in pixels.  Default is to use the PSF FWHM.
        allpars : boolean, optional
            Fit PSF model parameters as well.  Default is to only fit the stellar parameters
            of [height, xcen, ycen, sky].
        method : str, optional
            Method to use to solve the system of equations: "qr", "svd", or "curve_fit".
            Default is "QR".
        nosky : boolean, optional
            Do not fit the sky, only [height, xcen, and ycen].  Default is False.
        weight : boolean, optional
            Weight the data by 1/error**2.  Default is weight=True.

        Returns
        -------
        bestpars : numpy array
          Array of best-fit values [height, xcen, ycen, sky].

        Example
        -------

        pars = psf.fit(image,[1002.0,520.0,734.0])


        """
        # PARS: initial guesses for Xo and Yo parameters.
        if isinstance(pars,dict)==False:
            if len(pars)<3:
                raise ValueError('PARS must have [HEIGHT, XCEN, YCEN]')
            cat = {'height':pars[0],'x':pars[1],'y':pars[2]}
        else:
            if 'x' in pars.keys()==False or 'y' in pars.keys() is False:
                raise ValueError('PARS dictionary must have x and y')
            cat = pars

        nx,ny = im.shape
        xc = cat['x']
        yc = cat['y']
        if radius is None:
            radius = self.fwhm()
        x0 = int(np.maximum(0,np.floor(xc-radius)))
        x1 = int(np.minimum(np.ceil(xc+radius),nx-1))
        y0 = int(np.maximum(0,np.floor(yc-radius)))
        y1 = int(np.minimum(np.ceil(yc+radius),ny-1))
        X,Y = self.xylim2xy([[x0,x1],[y0,y1]])
        xdata = np.vstack((X.ravel(), Y.ravel()))
        
        flux = im.data[x0:x1+1,y0:y1+1]
        err = im.uncertainty.array[x0:x1+1,y0:y1+1]
        sky = np.median(im.data[x0:x1+1,y0:y1+1])
        if nosky: sky=0.0
        height = im.data[int(np.round(xc)),int(np.round(yc))]-sky
        initpar = [height,xc,yc,sky]            
        
        # Fit PSF parameters as well
        if allpars:
            initpar = np.hstack(([height,xc,yc,sky],self.params.copy()))

        # Remove sky column
        if nosky:
            initpar = np.delete(initpar,3,axis=0)
            
        # Use weights
        if weight:
            wt = 1.0/np.maximum(err,1)**2
        
        # Iterate
        count = 0
        bestpar = initpar.copy()
        while (count<niter):
            # Use QR or SVD to solve linear system of equations
            if allpars:
                m,jac = self.jac(xdata,*bestpar,allpars=True,retmodel=True)
            else:
                m,jac = self.jac(xdata,*bestpar,retmodel=True)
            dy = flux.flatten()-m.flatten()
            # Multipy by weights dy and jac by weights
            if weight:
                dy *= wt.flatten()
                jac = jac * wt.flatten().reshape(-1,1)
            # QR decomposition
            if str(method).lower()=='qr':
                q,r = np.linalg.qr(jac)
                rinv = np.linalg.inv(r)
                dbeta = rinv @ (q.T @ dy)
            # SVD:
            elif str(method).lower()=='svd':
                u,s,vt = np.linalg.svd(jac)
                # u: [Npix,Npix]
                # s: [Npars]
                # vt: [Npars,Npars]
                # dy: [Npix]
                sinv = s.copy()*0  # pseudo-inverse
                sinv[s!=0] = 1/s[s!=0]
                npars = len(s)
                dbeta = vt.T @ ((u.T @ dy)[0:npars]*sinv)
            # Curve_fit
            elif str(method).lower()=='curve_fit':
                if allpars==False:
                    outpars,cov = curve_fit(self.model,xdata,flux.ravel(),sigma=err.ravel(),p0=bestpar,jac=self.jac)
                    perror = np.sqrt(np.diag(cov))
                    return outpars,perror
                # Fit all parameters
                else:
                    outpars,cov = curve_fit(self.modelall,xdata,flux.ravel(),sigma=err.ravel(),p0=bestpar,jac=self.jacall)
                    perror = np.sqrt(np.diag(cov))
                    return outpars,perror
            else:
                raise ValueError('Only SVD or QR methods currently supported')
            
            oldpar = bestpar.copy()
            bestpar += dbeta
            count += 1
                
        return bestpar

    def sub(self,im,cat,sky=False,radius=None):
        """
        Method to fit a single star using the PSF model.

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
            
        nx,ny = im.shape
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
            x0 = int(np.maximum(0,np.floor(pars[1]-radius)))
            x1 = int(np.minimum(np.ceil(pars[1]+radius),nx-1))
            y0 = int(np.maximum(0,np.floor(pars[2]-radius)))
            y1 = int(np.minimum(np.ceil(pars[2]+radius),ny-1))
            xy = [[x0,x1],[y0,y1]]
            im1 = self(pars=pars,xy=xy)
            subim[x0:x1+1,y0:y1+1] -= im1
        return subim
                    
    
    def __str__(self):
        return self.__class__.__name__+'('+str(list(self.params))+',binned='+str(self.binned)+')'

    def __repr__(self):
        return self.__class__.__name__+'('+str(list(self.params))+',binned='+str(self.binned)+')'        

    def fwhm(self):
        """ Return the FWHM of the model function. Must be defined by subclass"""
        pass

    def flux(self,pars=None):
        """ Return the flux/volume of the model given the height.  Must be defined by subclass."""
        pass
    
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

    #@property
    def fwhm(self,pars=None):
        """ Return the FWHM of the model."""
        if pars is None:
            pars = np.hstack(([1.0,0.0,0.0],self.params))
        return gaussian2d_fwhm(pars)

    def flux(self,pars=None):
        """ Return the flux/volume of the model given the height or parameters."""
        if pars is None:
            pars = np.hstack(([1.0, 0.0, 0.0], self.params))
        else:
            pars = np.atleast_1d(pars)
            if pars.size==1:
                pars = np.hstack(([pars[0], 0.0, 0.0], self.params))            
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
            raise ValueError('4 parameters required')
        if mpars[0]<=0 or mpars[1]<=0:
            raise ValueError('sigma must be >0')
        if mpars[3]<0 or mpars[3]>6:
            raise ValueError('beta must be >0 and <6')
        super().__init__(mpars,npix=npix,binned=binned)

    def fwhm(self,pars=None):
        """ Return the FWHM of the model."""
        if pars is None:
            pars = np.hstack(([1.0,0.0,0.0],self.params))
        return moffat2d_fwhm(pars)

    def flux(self,pars=None):
        """ Return the flux/volume of the model given the height or parameters."""
        if pars is None:
            pars = np.hstack(([1.0, 0.0, 0.0], self.params))
        else:
            pars = np.atleast_1d(pars)
            if pars.size==1:
                pars = np.hstack(([pars[0], 0.0, 0.0], self.params))            
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

    
# PSF Penny class
class PSFPenny(PSFBase):
    """ Gaussian core and Lorentzian wings, only Gaussian is tilted."""
    # PARS are model parameters
    
    # Initalize the object
    def __init__(self,mpars=None,npix=101,binned=False):
        # mpars = [xsig,ysig,theta, relamp,sigma]
        if mpars is None:
            mpars = np.array([1.0,2.5,2.5,0.0,0.02,5.0])
        if len(mpars)!=5:
            raise ValueError('5 parameters required')
        if mpars[0]<=0:
            raise ValueError('sigma must be >0')
        if mpars[3]<0 or mpars[3]>1:
            raise ValueError('relative amplitude must be >=0 and <=1')
        super().__init__(mpars,npix=npix,binned=binned)

    def fwhm(self,pars=None):
        """ Return the FWHM of the model."""
        if pars is None:
            pars = np.hstack(([1.0,0.0,0.0],self.params))
        return penny2d_fwhm(pars)

    def flux(self,pars=None):
        """ Return the flux/volume of the model given the height or parameters."""
        if pars is None:
            pars = np.hstack(([1.0, 0.0, 0.0], self.params))
        else:
            pars = np.atleast_1d(pars)
            if pars.size==1:
                pars = np.hstack(([pars[0], 0.0, 0.0], self.params))            
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

    
class PSFEmpirical(PSFBase):
    """ Empirical look-up table PSF, can vary spatially."""

    # Initalize the object
    def __init__(self,mpars=None,npix=101):
        if mpars is None:
            raise ValueError('Must input images')
        # MPARS should be a two-element tuple with (parameters, psf cube)
        self.cube = mpars[1]
        nx,ny,npars = cube.shape
        super().__init__(mpars[0],npix=npix)        

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


    

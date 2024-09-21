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
from scipy.optimize import curve_fit, least_squares, line_search, root_scalar
from scipy.interpolate import interp1d
from skimage import measure
from dlnpyutils import utils as dln, bindata, ladfit, coords
from scipy.interpolate import RectBivariateSpline
from scipy.special import gamma, gammaincinv, gammainc
import copy
import logging
import time
import matplotlib
from . import getpsf, utils
from .ccddata import BoundingBox,CCDData
from . import leastsquares as lsq

#import pdb
#stop = pdb.set_trace()

# A bunch of the Gaussian2D and Moffat2D code comes from astropy's modeling module
# https://docs.astropy.org/en/stable/_modules/astropy/modeling/functional_models.html

# Maybe x0/y0 should NOT be part of the parameters, and
# x/y should actually just be dx/dy (relative to x0/y0)

warnings.filterwarnings('ignore')

def hfluxrad(im):
    """
    Calculate the half-flux radius of a star in an image.

    Parameters
    ----------
    im : numpy array
       The image of a star.

    Returns
    -------
    hfluxrad: float
       The half-flux radius.

    Example
    -------

    hfrad = hfluxrad(im)

    """
    ny,nx = im.shape
    xx,yy = np.meshgrid(np.arange(nx)-nx//2,np.arange(ny)-ny//2)
    rr = np.sqrt(xx**2+yy**2)
    si = np.argsort(rr.ravel())
    rsi = rr.ravel()[si]
    fsi = im.ravel()[si]
    totf = np.sum(fsi)
    cumf = np.cumsum(fsi)/totf
    hfluxrad = rsi[np.argmin(np.abs(cumf-0.5))]
    return hfluxrad
    
def contourfwhm(im):
    """
    Measure the FWHM of a PSF or star image using contours.

    Parameters
    ----------
    im : numpy array
     The 2D image of a star.

    Returns
    -------
    fwhm : float
       The full-width at half maximum.

    Example
    -------

    fwhm = contourfwhm(im)

    """
    # get contour at half max and then get average radius
    ny,nx = im.shape
    xcen = nx//2
    ycen = ny//2
    xx,yy = np.meshgrid(np.arange(nx)-nx//2,np.arange(ny)-ny//2)
    rr = np.sqrt(xx**2+yy**2)
    
    # Get half-flux radius
    hfrad = hfluxrad(im)
    # mask the image to only 2*half-flux radius
    mask = (rr<2*hfrad)
    
    # Find contours at a constant value of 0.5
    contours = measure.find_contours(im*mask, 0.5*np.max(im))
    # If there are multiple contours, find the one that
    #   encloses the center
    if len(contours)>1:
        for i in range(len(contours)):
            x1 = contours[i][:,0]
            y1 = contours[i][:,1]
            inside = coords.isPointInPolygon(x1,y1,xcen,ycen)
            if inside:
                contours = contours[i]
                break
    else:
        contours = contours[0]   # first level
    xc = contours[:,0]
    yc = contours[:,1]
    r = np.sqrt((xc-nx//2)**2+(yc-ny//2)**2)
    fwhm = np.mean(r)*2
    return fwhm

def imfwhm(im):
    """
    Measure the FWHM of a PSF or star image.

    Parameters
    ----------
    im : numpy array
      The image of a star.

    Returns
    -------
    fwhm : float
      The full-width at half maximum of the star.

    Example
    -------

    fwhm = imfwhm(im)

    """
    ny,nx = im.shape
    xx,yy = np.meshgrid(np.arange(nx)-nx//2,np.arange(ny)-ny//2)
    rr = np.sqrt(xx**2+yy**2)
    centerf = im[ny//2,nx//2]
    si = np.argsort(rr.ravel())
    rsi = rr.ravel()[si]
    fsi = im.ravel()[si]
    ind, = np.where(fsi<0.5*centerf)
    bestr = np.min(rsi[ind])
    bestind = ind[np.argmin(rsi[ind])]
    # fit a robust line to the neighboring points
    gd, = np.where(np.abs(rsi-bestr) < 1.0)
    coef,absdev = ladfit.ladfit(rsi[gd],fsi[gd])
    # where does the line cross y=0.5
    bestr2 = (0.5-coef[0])/coef[1]
    fwhm = 2*bestr2
    return fwhm

def starbbox(coords,imshape,radius):
    """
    Return the boundary box for a star given radius and image size.
        
    Parameters
    ----------
    coords: list or tuple
       Central coordinates (xcen,ycen) of star (*absolute* values).
    imshape: list or tuple
       Image shape (ny,nx) values.  Python images are (Y,X).
    radius: float
       Radius in pixels.
        
    Returns
    -------
    bbox : BoundingBox object
       Bounding box of the x/y ranges.
       Upper values are EXCLUSIVE following the python convention.

    """

    # Star coordinates
    xcen,ycen = coords
    ny,nx = imshape   # python images are (Y,X)
    xlo = np.maximum(int(np.floor(xcen-radius)),0)
    xhi = np.minimum(int(np.ceil(xcen+radius+1)),nx)
    ylo = np.maximum(int(np.floor(ycen-radius)),0)
    yhi = np.minimum(int(np.ceil(ycen+radius+1)),ny)
        
    return BoundingBox(xlo,xhi,ylo,yhi)

def bbox2xy(bbox):
    """
    Convenience method to convert boundary box of X/Y limits to 2-D X and Y arrays.  The upper limits
    are EXCLUSIVE following the python convention.

    Parameters
    ----------
    bbox : BoundingBox object
      A BoundingBox object defining a rectangular region of an image.

    Returns
    -------
    x : numpy array
      The 2D array of X-values of the bounding box region.
    y : numpy array
      The 2D array of Y-values of the bounding box region.

    Example
    -------

    x,y = bbox2xy(bbox)

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

def gaussian2d(x,y,pars,deriv=False,nderiv=None):
    """
    Two dimensional Gaussian model function.
    
    Parameters
    ----------
    x : numpy array
      Array of X-values of points for which to compute the Gaussian model.
    y : numpy array
      Array of Y-values of points for which to compute the Gaussian model.
    pars : numpy array or list
       Parameter list. pars = [amplitude, x0, y0, xsigma, ysigma, theta]
    deriv : boolean, optional
       Return the derivatives as well.
    nderiv : int, optional
       The number of derivatives to return.  The default is None
        which means that all are returned if deriv=True.

    Returns
    -------
    g : numpy array
      The Gaussian model for the input x/y values and parameters (same
        shape as x/y).
    derivative : list
      List of derivatives of g relative to the input parameters.
        This is only returned if deriv=True.

    Example
    -------

    g = gaussian2d(x,y,pars)

    or

    g,derivative = gaussian2d(x,y,pars,deriv=True)

    """
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
    """
    Return the FWHM of a 2D Gaussian.

    Parameters
    ----------
    pars : numpy array or list
       Parameter list. pars = [amplitude, x0, y0, xsigma, ysigma, theta]

    Returns
    -------
    fwhm : float
       The full-width at half maximum of the Gaussian.
    
    Example
    -------

    fwhm = gaussian2d_fwhm(pars)

    """
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
    """
    Return the total flux (or volume) of a 2D Gaussian.

    Parameters
    ----------
    pars : numpy array or list
       Parameter list. pars = [amplitude, x0, y0, xsigma, ysigma, theta]

    Returns
    -------
    flux : float
       Total flux or volumne of the 2D Gaussian.
    
    Example
    -------

    flux = gaussian2d_flux(pars)

    """
    # Volume is 2*pi*A*sigx*sigy
    amp = pars[0]
    xsig = pars[3]
    ysig = pars[4]    
    volume = 2*np.pi*amp*xsig*ysig
    
    return volume
    

def gaussian2d_sigtheta2abc(xstd,ystd,theta):
    """
    Convert 2D Gaussian sigma_x, sigma_y and theta to a, b, c coefficients.
    f(x,y) = A*exp(-0.5 * (a*xdiff**2 + b*xdiff*ydiff + c*ydiff**2))

    Parameters
    ----------
    xstd : float
      The Gaussian sigma in the x-dimension.
    ystd : float
      The Gaussian sigma in the y-dimension.
    theta : float
      The orientation angle of the elliptical 2D Gaussian (radians).

    Returns
    -------
    a : float
      The x**2 coefficient in the 2D elliptical Gaussian equation.
    b : float
      The y**2 coefficient in the 2D elliptical Gaussian equation.
    c : float
      The x*y coefficient in the 2D elliptical Gaussian equation.

    Example
    -------
 
    a,b,c = gaussian2d_sigtheta2abc(xstd,ystd,theta)

    """
    
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
    """
    Convert 2D Gaussian a, b, c coefficients to sigma_x, sigma_y and theta.
    The inverse of guassian2d_sigtheta2abc().
    f(x,y) = A*exp(-0.5 * (a*xdiff**2 + b*xdiff*ydiff + c*ydiff**2))

    Parameters
    ----------
    a : float
      The x**2 coefficient in the 2D elliptical Gaussian equation.
    b : float
      The y**2 coefficient in the 2D elliptical Gaussian equation.
    c : float
      The x*y coefficient in the 2D elliptical Gaussian equation.

    Returns
    -------
    xstd : float
      The Gaussian sigma in the x-dimension.
    ystd : float
      The Gaussian sigma in the y-dimension.
    theta : float
      The orientation angle of the elliptical 2D Gaussian (radians).

    Example
    -------

    xstd,ystd,stheta = gaussian2d_abc2sigtheta(a,b,c)

    """
    
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
    """
    Two dimensional Gaussian model function integrated over the pixels.

   
    Parameters
    ----------
    x : numpy array
      Array of X-values of points for which to compute the Gaussian model.
    y : numpy array
      Array of Y-values of points for which to compute the Gaussian model.
    pars : numpy array or list
       Parameter list. pars = [amplitude, x0, y0, xsigma, ysigma, theta]
    deriv : boolean, optional
       Return the derivatives as well.
    nderiv : int, optional
       The number of derivatives to return.  The default is None
        which means that all are returned if deriv=True.
    osamp : int, optional
       The oversampling of the pixel when doing the integrating.
          Default is 4.

    Returns
    -------
    g : numpy array
      The Gaussian model for the input x/y values and parameters (same
        shape as x/y).
    derivative : list
      List of derivatives of g relative to the input parameters.
        This is only returned if deriv=True.

    Example
    -------

    g = gaussian2d_integrate(x,y,pars)

    or

    g,derivative = gaussian2d_integrate(x,y,pars,deriv=True)

    """

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
    theta = pars[5]
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
    """
    Two dimensional Moffat model function.

    Parameters
    ----------
    x : numpy array
      Array of X-values of points for which to compute the Moffat model.
    y : numpy array
      Array of Y-values of points for which to compute the Moffat model.
    pars : numpy array or list
       Parameter list. pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta]
    deriv : boolean, optional
       Return the derivatives as well.
    nderiv : int, optional
       The number of derivatives to return.  The default is None
        which means that all are returned if deriv=True.

    Returns
    -------
    g : numpy array
      The Moffat model for the input x/y values and parameters (same
        shape as x/y).
    derivative : list
      List of derivatives of g relative to the input parameters.
        This is only returned if deriv=True.

    Example
    -------

    g = moffat2d(x,y,pars)

    or

    g,derivative = moffat2d(x,y,pars,deriv=True)

    """
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
    """
    Return the FWHM of a 2D Moffat function.

    Parameters
    ----------
    pars : numpy array or list
       Parameter list. pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta]

    Returns
    -------
    fwhm : float
       The full-width at half maximum of the Moffat.
    
    Example
    -------

    fwhm = moffat2d_fwhm(pars)

    """

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
    """
    Return the total Flux of a 2D Moffat.

    Parameters
    ----------
    pars : numpy array or list
       Parameter list. pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta]

    Returns
    -------
    flux : float
       Total flux or volumne of the 2D Moffat.
    
    Example
    -------

    flux = moffat2d_flux(pars)

    """

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
    """
    Two dimensional Moffat model function integrated over the pixels.

    Parameters
    ----------
    x : numpy array
      Array of X-values of points for which to compute the Moffat model.
    y : numpy array
      Array of Y-values of points for which to compute the Moffat model.
    pars : numpy array or list
       Parameter list. pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta]
    deriv : boolean, optional
       Return the derivatives as well.
    nderiv : int, optional
       The number of derivatives to return.  The default is None
        which means that all are returned if deriv=True.
    osamp : int, optional
       The oversampling of the pixel when doing the integrating.
          Default is 4.

    Returns
    -------
    g : numpy array
      The Moffat model for the input x/y values and parameters (same
        shape as x/y).
    derivative : list
      List of derivatives of g relative to the input parameters.
        This is only returned if deriv=True.

    Example
    -------

    g = moffat2d_integrate(x,y,pars)

    or

    g,derivative = moffat2d_integrate(x,y,pars,deriv=True)

    """
    # pars = [amplitude, x0, y0, xstd, ystd, theta, beta]

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
    """
    Gaussian core and Lorentzian-like wings, only Gaussian is tilted.

    Parameters
    ----------
    x : numpy array
      Array of X-values of points for which to compute the Penny model.
    y : numpy array
      Array of Y-values of points for which to compute the Penny model.
    pars : numpy array or list
       Parameter list.
        pars = [amplitude, x0, y0, xsigma, ysigma, theta, relamp, sigma]
    deriv : boolean, optional
       Return the derivatives as well.
    nderiv : int, optional
       The number of derivatives to return.  The default is None
        which means that all are returned if deriv=True.

    Returns
    -------
    g : numpy array
      The Penny model for the input x/y values and parameters (same
        shape as x/y).
    derivative : list
      List of derivatives of g relative to the input parameters.
        This is only returned if deriv=True.

    Example
    -------

    g = penny2d(x,y,pars)

    or

    g,derivative = penny2d(x,y,pars,deriv=True)

    """
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
    relamp = np.minimum(np.maximum(pars[6],0),1.0)  # 0<relamp<1
    # Gaussian component
    g = amp * (1-relamp) * np.exp(-0.5*((a * xdiff ** 2) + (b * xdiff*ydiff) +
                                        (c * ydiff ** 2)))
    # Add Lorentzian/Moffat beta=1.2 wings
    sigma = np.maximum(pars[7],0)
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
    """
    Return the FWHM of a 2D Penny function.

    Parameters
    ----------
    pars : numpy array or list
       Parameter list.
        pars = [amplitude, x0, y0, xsigma, ysigma, theta, relamp, sigma]

    Returns
    -------
    fwhm : float
       The full-width at half maximum of the Penny function.
    
    Example
    -------

    fwhm = penny2d_fwhm(pars)

    """

    # [amplitude, x0, y0, xsig, ysig, theta, relative amplitude, sigma]

    amp = pars[0]
    xsig = pars[3]
    ysig = pars[4]
    relamp = np.minimum(np.maximum(pars[6],0),1.0)  # 0<relamp<1
    sigma = np.maximum(pars[7],0)
    beta = 1.2   # Moffat

    if np.sum(~np.isfinite(np.array(pars)))>0:
        raise ValueError('PARS cannot be inf or nan')
    
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
    """
    Return the total Flux of a 2D Penny function.

    Parameters
    ----------
    pars : numpy array or list
       Parameter list.
        pars = [amplitude, x0, y0, xsigma, ysigma, theta, relamp, sigma]

    Returns
    -------
    flux : float
       Total flux or volumne of the 2D Penny function.
    
    Example
    -------

    flux = penny2d_flux(pars)

    """
    # [amplitude, x0, y0, xsig, ysig, theta, relative amplitude, sigma]    

    # Volume is 2*pi*A*sigx*sigy
    # area of 1D moffat function is pi*alpha**2 / (beta-1)
    # maybe the 2D moffat volume is (xsig*ysig*pi**2/(beta-1))**2

    amp = pars[0]
    xsig = pars[3]
    ysig = pars[4]
    relamp = np.minimum(np.maximum(pars[6],0),1.0)  # 0<relamp<1
    sigma = np.maximum(pars[7],0)
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
    """
    Gaussian core and Lorentzian-like wings, only Gaussian is tilted
    integrated over the pixels.

    Parameters
    ----------
    x : numpy array
      Array of X-values of points for which to compute the Penny model.
    y : numpy array
      Array of Y-values of points for which to compute the Penny model.
    pars : numpy array or list
       Parameter list.
        pars = [amplitude, x0, y0, xsigma, ysigma, theta, relamp, sigma]
    deriv : boolean, optional
       Return the derivatives as well.
    nderiv : int, optional
       The number of derivatives to return.  The default is None
        which means that all are returned if deriv=True.
    osamp : int, optional
       The oversampling of the pixel when doing the integrating.
          Default is 4.

    Returns
    -------
    g : numpy array
      The Penny model for the input x/y values and parameters (same
        shape as x/y).
    derivative : list
      List of derivatives of g relative to the input parameters.
        This is only returned if deriv=True.

    Example
    -------

    g = penny2d_integrate(x,y,pars)

    or

    g,derivative = penny2d_integrate(x,y,pars,deriv=True)


    """
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
    relamp = np.minimum(np.maximum(pars[6],0),1.0)  # 0<relamp<1

    # Gaussian component
    g = amp * (1-relamp) * np.exp(-0.5*((a * xdiff ** 2) + (b * xdiff*ydiff) +
                                        (c * ydiff ** 2)))
    # Add Lorentzian/Moffat beta=1.2 wings
    sigma = np.maximum(pars[7],0)
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
    """
    DoPHOT PSF, sum of elliptical Gaussians.

    Parameters
    ----------
    x : numpy array
      Array of X-values of points for which to compute the Gausspow model.
    y : numpy array
      Array of Y-values of points for which to compute the Gausspow  model.
    pars : numpy array or list
       Parameter list.
        pars = [amplitude, x0, y0, sigx, sigy, theta, beta4, beta6]
    deriv : boolean, optional
       Return the derivatives as well.
    nderiv : int, optional
       The number of derivatives to return.  The default is None
        which means that all are returned if deriv=True.

    Returns
    -------
    g : numpy array
      The Gausspow model for the input x/y values and parameters (same
        shape as x/y).
    derivative : list
      List of derivatives of g relative to the input parameters.
        This is only returned if deriv=True.

    Example
    -------

    g = gausspow2d(x,y,pars)

    or

    g,derivative = gausspow2d(x,y,pars,deriv=True)

    """

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
    """
    Return the FWHM of a 2D DoPHOT Gausspow function.

    Parameters
    ----------
    pars : numpy array or list
       Parameter list.
        pars = [amplitude, x0, y0, sigx, sigy, theta, beta4, beta6]

    Returns
    -------
    fwhm : float
       The full-width at half maximum of the Penny function.
    
    Example
    -------

    fwhm = gausspow2d_fwhm(pars)

    """

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
    """
    Return the flux of a 2D DoPHOT Gausspow function.

    Parameters
    ----------
    pars : numpy array or list
       Parameter list.
        pars = [amplitude, x0, y0, sigx, sigy, theta, beta4, beta6]

    Returns
    -------
    flux : float
       Total flux or volumne of the 2D Gausspow function.
    
    Example
    -------

    flux = gausspow2d_flux(pars)

    """

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
    """
    DoPHOT PSF, integrated over the pixels.

    Parameters
    ----------
    x : numpy array
      Array of X-values of points for which to compute the Gausspow model.
    y : numpy array
      Array of Y-values of points for which to compute the Gausspow  model.
    pars : numpy array or list
       Parameter list.
        pars = [amplitude, x0, y0, sigx, sigy, theta, beta4, beta6]
    deriv : boolean, optional
       Return the derivatives as well.
    nderiv : int, optional
       The number of derivatives to return.  The default is None
        which means that all are returned if deriv=True.
    osamp : int, optional
       The oversampling of the pixel when doing the integrating.
          Default is 4.

    Returns
    -------
    g : numpy array
      The Gausspow model for the input x/y values and parameters (same
        shape as x/y).
    derivative : list
      List of derivatives of g relative to the input parameters.
        This is only returned if deriv=True.

    Example
    -------

    g = gausspow2d_integrate(x,y,pars)

    or

    g,derivative = gausspow2d_integreate(x,y,pars,deriv=True)

    """

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

   
def sersic2d(x, y, pars, deriv=False, nderiv=None):
    """
    Sersic profile and can be elliptical and rotated.

    Parameters
    ----------
    x : numpy array
      Array of X-values of points for which to compute the Sersic model.
    y : numpy array
      Array of Y-values of points for which to compute the Sersic model.
    pars : numpy array or list
       Parameter list.
        pars = [amp,x0,y0,k,alpha,recc,theta]
    deriv : boolean, optional
       Return the derivatives as well.
    nderiv : int, optional
       The number of derivatives to return.  The default is None
        which means that all are returned if deriv=True.

    Returns
    -------
    g : numpy array
      The Sersic model for the input x/y values and parameters (same
        shape as x/y).
    derivative : list
      List of derivatives of g relative to the input parameters.
        This is only returned if deriv=True.

    Example
    -------

    g = sersic2d(x,y,pars)

    or

    g,derivative = sersic2d(x,y,pars,deriv=True)

    """
    # pars = [amp,x0,y0,k,alpha,recc,theta]

    # Sersic radial profile
    # I(R) = I0 * exp(-k*R**(1/n))
    # n is the sersic index
    # I'm going to use alpha = 1/n instead
    # I(R) = I0 * exp(-k*R**alpha)    
    # most galaxies have indices in the range 1/2 < n < 10
    # n=4 is the de Vaucouleurs profile
    # n=1 is the exponential
    
    xdiff = x - pars[1]
    ydiff = y - pars[2]
    amp = pars[0]
    kserc = pars[3]
    alpha = pars[4]
    recc = pars[5]               # b/a
    theta = pars[6]    
    cost2 = np.cos(theta) ** 2
    sint2 = np.sin(theta) ** 2
    sin2t = np.sin(2. * theta)
    xsig2 = 1.0           # major axis
    ysig2 = recc ** 2     # minor axis
    a = (cost2 + (sint2 / ysig2))
    b = (sin2t - (sin2t / ysig2))    
    c = (sint2 + (cost2 / ysig2))

    rr = np.sqrt( (a * xdiff ** 2) + (b * xdiff * ydiff) + (c * ydiff ** 2) )
    g = amp * np.exp(-kserc*rr**alpha)
   
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
            dg_dx_mean = g * (kserc*alpha)*(rr**(alpha-2))*0.5*((2 * a * xdiff) + (b * ydiff))
            dg_dx_mean[rr==0] = 0
            derivative.append(dg_dx_mean)
        if nderiv>=3:
            dg_dy_mean = g * (kserc*alpha)*(rr**(alpha-2))*0.5*((2 * c * ydiff) + (b * xdiff))
            dg_dx_mean[rr==0] = 0           
            derivative.append(dg_dy_mean)
        if nderiv>=4:
            dg_dk = -g * rr**alpha
            derivative.append(dg_dk)
        if nderiv>=5:
            dg_dalpha = -g * kserc*np.log(rr) * rr**alpha
            dg_dalpha[rr==0] = 0
            derivative.append(dg_dalpha)
        if nderiv>=6:
            xdiff2 = xdiff ** 2
            ydiff2 = ydiff ** 2
            recc3 = recc**3
            da_drecc = -2*sint2 / recc3
            db_drecc =  2*sin2t / recc3            
            dc_drecc = -2*cost2 / recc3            
            dg_drecc = -g*(kserc*alpha)*(rr**(alpha-2))*0.5*(da_drecc * xdiff2 +
                                                             db_drecc * xdiff * ydiff +
                                                             dc_drecc * ydiff2)
            dg_drecc[rr==0] = 0
            derivative.append(dg_drecc)
        if nderiv>=7:
            sint = np.sin(theta)
            cost = np.cos(theta)
            cos2t = np.cos(2.0*theta)
            da_dtheta = (sint * cost * ((1. / ysig2) - (1. / xsig2)))
            db_dtheta = (cos2t / xsig2) - (cos2t / ysig2)            
            dc_dtheta = -da_dtheta            
            dg_dtheta = -g*(kserc*alpha)*(rr**(alpha-2))*(da_dtheta * xdiff2 +
                                                          db_dtheta * xdiff * ydiff +
                                                          dc_dtheta * ydiff2)
            dg_dtheta[rr==0] = 0
            derivative.append(dg_dtheta)

        # special case if alpha=2???
            
        return g,derivative
            
    # No derivative
    else:        
        return g

def sersic2d_fwhm(pars):
    """
    Return the FWHM of a 2D Sersic function.

    Parameters
    ----------
    pars : numpy array or list
       Parameter list.
        pars = [amp,x0,y0,k,alpha,recc,theta]

    Returns
    -------
    fwhm : float
       The full-width at half maximum of the Sersic function.
    
    Example
    -------

    fwhm = sersic2d_fwhm(pars)

    """

    # pars = [amp,x0,y0,k,alpha,recc,theta]
    # x0,y0 and theta are irrelevant

    amp = pars[0]
    kserc = pars[3]
    alpha = pars[4]
    recc = pars[5]               # b/a

    if np.sum(~np.isfinite(np.array(pars)))>0:
        raise ValueError('PARS cannot be inf or nan')

    # Radius of half maximum
    # I(R) = I0 * exp(-k*R**alpha) 
    # 0.5*I0 = I0 * exp(-k*R**alpha)
    # 0.5 = exp(-k*R**alpha)
    # ln(0.5) = -k*R**alpha
    # R = (-ln(0.5)/k)**(1/alpha)
    rhalf = (-np.log(0.5)/kserc)**(1/alpha)
    
    # The mean radius of an ellipse is: (2a+b)/3
    sig_major = rhalf
    sig_minor = rhalf*recc
    mnsig = (2.0*sig_major+sig_minor)/3.0
    # Convert sigma to FWHM
    # FWHM = 2*sqrt(2*ln(2))*sig ~ 2.35482*sig
    fwhm = mnsig*2.35482

    # Generate a small profile
    #x = np.arange( gfwhm/2.35/2, gfwhm, 0.5)
    #f = amp * np.exp(-kserc*x**alpha)
    #hwhm = np.interp(0.5,f[::-1],x[::-1])
    #fwhm = 2*hwhm
    
    return fwhm

# https://gist.github.com/bamford/b657e3a14c9c567afc4598b1fd10a459
def sersic_b(n):
    # Normalisation constant
    # bn ~ 2n-1/3 for n>8
    return gammaincinv(2*n, 0.5)

def create_sersic_function(Ie, re, n):
    # Not required for integrals - provided for reference
    # This returns a "closure" function, which is fast to call repeatedly with different radii
    neg_bn = -b(n)
    reciprocal_n = 1.0/n
    f = neg_bn/re**reciprocal_n
    def sersic_wrapper(r):
        return Ie * np.exp(f * r ** reciprocal_n - neg_bn)
    return sersic_wrapper
    
def sersic_lum(Ie, re, n):
    # total luminosity (integrated to infinity)
    bn = sersic_b(n)
    g2n = gamma(2*n)
    return Ie * re**2 * 2*np.pi*n * np.exp(bn)/(bn**(2*n)) * g2n

def sersic_full2half(I0,kserc,alpha):
    # Convert Io and k to Ie and Re
    # Ie = Io * exp(-bn)
    # Re = (bn/k)**n
    n = 1/alpha
    bn = sersic_b(n)
    Ie = I0 * np.exp(-bn)
    Re = (bn/kserc)**n
    return Ie,Re

def sersic_half2full(Ie,Re,alpha):
    # Convert Ie and Re to Io and k
    # Ie = Io * exp(-bn)
    # Re = (bn/k)**n
    n = 1/alpha
    bn = sersic_b(n)
    I0 = Ie * np.exp(bn)
    kserc = bn/Re**alpha
    return I0,kserc

def sersic2d_flux(pars):
    """
    Return the total Flux of a 2D Sersic function.

    Parameters
    ----------
    pars : numpy array or list
       Parameter list.  pars = [amp,x0,y0,k,alpha,recc,theta]

    Returns
    -------
    flux : float
       Total flux or volumne of the 2D Sersic function.
    
    Example
    -------

    flux = sersic2d_flux(pars)

    """

    # pars = [amp,x0,y0,k,alpha,recc,theta]

    # Volume is 2*pi*A*sigx*sigy
    # area of 1D moffat function is pi*alpha**2 / (beta-1)
    # maybe the 2D moffat volume is (xsig*ysig*pi**2/(beta-1))**2

    # x0, y0 and theta are irrelevant
    
    amp = pars[0]
    kserc = pars[3]
    alpha = pars[4]
    recc = pars[5]               # b/a

    # https://ned.ipac.caltech.edu/level5/March05/Graham/Graham2.html
    # integrating over an area out to R gives
    # I0 * Re**2 * 2*pi*n*(e**bn)/(bn)**2n * gammainc(2n,x)

    # Re encloses half of the light
    # Convert Io and k to Ie and Re
    Ie,Re = sersic_full2half(amp,kserc,alpha)
    n = 1/alpha
    volume = sersic_lum(Ie,Re,n)

    # Mulitply by recc (b/a)
    volume *= recc
    
    # Volume for 2D Gaussian is 2*pi*A*sigx*sigy
    
    return volume


def sersic2d_estimates(pars):
    # calculate estimates for the Sersic parameters using
    # peak, x0, y0, flux, asemi, bsemi, theta
    # Sersic Parameters are [amp,x0,y0,k,alpha,recc,theta]
    peak = pars[0]
    x0 = pars[1]
    y0 = pars[2]
    flux = pars[3]
    asemi = pars[4]
    bsemi = pars[5]
    theta = pars[6]
    recc = bsemi/asemi
    
    # Calculate FWHM
    # The mean radius of an ellipse is: (2a+b)/3
    mnsig = (2.0*asemi+bsemi)/3.0
    # Convert sigma to FWHM
    # FWHM = 2*sqrt(2*ln(2))*sig ~ 2.35482*sig
    fwhm = mnsig*2.35482
    rhalf = 0.5*fwhm
    
    # Solve half-max radius equation for kserc
    # I(R) = I0 * exp(-k*R**alpha) 
    # 0.5*I0 = I0 * exp(-k*R**alpha)
    # 0.5 = exp(-k*R**alpha)
    # ln(0.5) = -k*R**alpha
    # R = (-ln(0.5)/k)**(1/alpha)
    # rhalf = (-np.log(0.5)/kserc)**(1/alpha)
    # kserc = -np.log(0.5)/rhalf**alpha
   
    # Solve flux equation for kserc
    # bn = sersic_b(n)
    # g2n = gamma(2*n)
    # flux =  recc * Ie * Re**2 * 2*np.pi*n * np.exp(bn)/(bn**(2*n)) * g2n
    # Re = np.sqrt(flux/(recc * Ie * 2*np.pi*n * np.exp(bn)/(bn**(2*n)) * g2n))
    # kserc = bn/Re**alpha    
    # kserc = bn * ((recc * Ie * 2*np.pi*n * np.exp(bn)/(bn**(2*n)) * g2n)/flux)**(alpha/2)

    # Setting the two equal and then putting everything to one side
    # 0 = np.log(0.5)/rhalf**alpha + bn * ((recc * Ie * 2*np.pi*n * np.exp(bn)/(bn**(2*n)) * g2n)/flux)**(alpha/2)
    def alphafunc(alpha):
        # rhalf, recc, flux are defined above
        n = 1/alpha
        bn = sersic_b(n)
        g2n = gamma(2*n)
        Ie,_ = sersic_full2half(peak,1.0,alpha)
        return np.log(0.5)/rhalf**alpha + bn * ((recc * Ie * 2*np.pi*n * np.exp(bn)/(bn**(2*n)) * g2n)/flux)**(alpha/2)
    
    # Solve for the roots
    res = root_scalar(alphafunc,x0=1.0,x1=0.5)
    if res.converged:
        alpha = res.root
    else:
        alphas = np.arange(0.1,2.0,0.05)
        vals = np.zeros(len(alphas),float)
        for i in range(len(alphas)):
            vals[i] = alphafunc(alphas[i])
        bestind = np.argmin(np.abs(vals))
        alpha = alphas[bestind]
                            
    # Now solve for ksersic
    # rhalf = (-np.log(0.5)/kserc)**(1/alpha)
    kserc = -np.log(0.5)/rhalf**alpha
    
    # Put all the parameters together
    spars = [peak,x0,y0,kserc,alpha,recc,theta]
    
    return spars


def sersic2d_integrate(x, y, pars, deriv=False, nderiv=None, osamp=4):
    """
    Sersic profile and can be elliptical and rotated, integrated over the pixels.

    Parameters
    ----------
    x : numpy array
      Array of X-values of points for which to compute the Sersic model.
    y : numpy array
      Array of Y-values of points for which to compute the Sersic model.
    pars : numpy array or list
       Parameter list.
        pars = [amp,x0,y0,k,alpha,recc,theta]
    deriv : boolean, optional
       Return the derivatives as well.
    nderiv : int, optional
       The number of derivatives to return.  The default is None
        which means that all are returned if deriv=True.
    osamp : int, optional
       The oversampling of the pixel when doing the integrating.
          Default is 4.

    Returns
    -------
    g : numpy array
      The Sersic model for the input x/y values and parameters (same
        shape as x/y).
    derivative : list
      List of derivatives of g relative to the input parameters.
        This is only returned if deriv=True.

    Example
    -------

    g = sersic2d_integrate(x,y,pars)

    or

    g,derivative = sersic2d_integrate(x,y,pars,deriv=True)

    """
    # pars = [amp,x0,y0,k,alpha,recc,theta]
    
    # Sersic radial profile
    # I(R) = I0 * exp(-k*R**(1/n))
    # n is the sersic index
    # I'm going to use alpha = 1/n instead
    # I(R) = I0 * exp(-k*R**alpha)    
    # most galaxies have indices in the range 1/2 < n < 10
    # n=4 is the de Vaucouleurs profile
    # n=1 is the exponential

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
    kserc = pars[3]
    alpha = pars[4]
    recc = pars[5]               # b/a
    theta = pars[6]    
    cost2 = np.cos(theta) ** 2
    sint2 = np.sin(theta) ** 2
    sin2t = np.sin(2. * theta)
    xsig2 = 1.0           # major axis
    ysig2 = recc ** 2     # minor axis
    a = (cost2 + (sint2 / ysig2))
    b = (sin2t - (sin2t / ysig2))    
    c = (sint2 + (cost2 / ysig2))

    # Gaussian component
    rr = np.sqrt( (a * xdiff ** 2) + (b * xdiff * ydiff) + (c * ydiff ** 2) )
    g = amp * np.exp(-kserc*rr**alpha)
   
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
            derivative.append(np.sum(np.sum(dg_dA,axis=0),axis=0)/osamp2)
        if nderiv>=2:        
            dg_dx_mean = g * (kserc*alpha)*(rr**(alpha-2))*0.5*((2 * a * xdiff) + (b * ydiff))
            dg_dx_mean[rr==0] = 0
            derivative.append(np.sum(np.sum(dg_dx_mean,axis=0),axis=0)/osamp2)
        if nderiv>=3:
            dg_dy_mean = g * (kserc*alpha)*(rr**(alpha-2))*0.5*((2 * c * ydiff) + (b * xdiff))
            dg_dx_mean[rr==0] = 0           
            derivative.append(np.sum(np.sum(dg_dy_mean,axis=0),axis=0)/osamp2)
        if nderiv>=4:
            dg_dk = -g * rr**alpha
            derivative.append(np.sum(np.sum(dg_dk,axis=0),axis=0)/osamp2)
        if nderiv>=5:
            dg_dalpha = -g * kserc*np.log(rr) * rr**alpha
            dg_dalpha[rr==0] = 0
            derivative.append(np.sum(np.sum(dg_dalpha,axis=0),axis=0)/osamp2)
        if nderiv>=6:
            xdiff2 = xdiff ** 2
            ydiff2 = ydiff ** 2
            recc3 = recc**3
            da_drecc = -2*sint2 / recc3
            db_drecc =  2*sin2t / recc3            
            dc_drecc = -2*cost2 / recc3            
            dg_drecc = -g*(kserc*alpha)*(rr**(alpha-2))*0.5*(da_drecc * xdiff2 +
                                                             db_drecc * xdiff * ydiff +
                                                             dc_drecc * ydiff2)
            dg_drecc[rr==0] = 0
            derivative.append(np.sum(np.sum(dg_drecc,axis=0),axis=0)/osamp2)
        if nderiv>=7:
            sint = np.sin(theta)
            cost = np.cos(theta)
            cos2t = np.cos(2.0*theta)
            da_dtheta = (sint * cost * ((1. / ysig2) - (1. / xsig2)))
            db_dtheta = (cos2t / xsig2) - (cos2t / ysig2)            
            dc_dtheta = -da_dtheta            
            dg_dtheta = -g*(kserc*alpha)*(rr**(alpha-2))*(da_dtheta * xdiff2 +
                                                          db_dtheta * xdiff * ydiff +
                                                          dc_dtheta * ydiff2)
            dg_dtheta[rr==0] = 0
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
    
    
def relcoord(x,y,shape):
    """
    Convert absolute X/Y coordinates to relative ones to use
    with the lookup table.

    Parameters
    ----------
    x : numpy array
      Input x-values of positions in an image.
    y : numpy array
      Input Y-values of positions in an image.
    shape : tuple or list
      Two-element tuple or list of the (Ny,Nx) size of the image.

    Returns
    -------
    relx : numpy array
      The relative x-values ranging from -1 to +1.
    rely : numpy array
      The relative y-values ranging from -1 to +1.

    Example
    -------

    relx,rely = relcoord(x,y,shape)

    """

    midpt = [shape[0]//2,shape[1]//2]
    relx = (x-midpt[1])/shape[1]*2
    rely = (y-midpt[0])/shape[0]*2
    return relx,rely
    
def empirical(x, y, pars, data, shape=None, deriv=False, korder=3):
    """
    Evaluate an empirical PSF.

    Parameters
    ----------
    x : numpy array
      Array of X-values of points for which to compute the empirical model.
    y : numpy array
      Array of Y-values of points for which to compute the empirical model.
    pars : numpy array or list
       Parameter list.  pars = [amplitude, x0, y0].
    deriv : boolean, optional
       Return the derivatives as well.
    nderiv : int, optional
       The number of derivatives to return.  The default is None
        which means that all are returned if deriv=True.

    Returns
    -------
    g : numpy array
      The empirical model for the input x/y values and parameters (same
        shape as x/y).
    derivative : list
      List of derivatives of g relative to the input parameters.
        This is only returned if deriv=True.

    Example
    -------

    g = empirical(x,y,pars)

    or

    g,derivative = empirical(x,y,pars,deriv=True)

    """

    npars = len(pars)

    # Parameters for the profile
    amp = pars[0]
    x0 = pars[1]
    y0 = pars[2]
        
    # Relative positions
    dx = x - x0
    dy = y - y0

    # Turn into a list of RectBivariateSpline objects
    ldata = []
    if isinstance(data,np.ndarray):
        if data.ndim==2:
            ldata = [np.copy(data)]
        elif data.ndim==3:
            ldata = []
            for i in range(data.shape[2]):
                ldata.append(data[:,:,i])
        else:
            raise ValueError('Data must be 2D or 3D numpy array')
    elif isinstance(data,RectBivariateSpline):
        ldata = [data]
    elif isinstance(data,list):
        ldata = data
    else:
        raise ValueError('Data type not understood')
    ndata = len(ldata)
    
    # Make list of RectBivariateSpline objects
    farr = []
    for i in range(ndata):
        if isinstance(ldata[i],RectBivariateSpline):
            farr.append(ldata[i])
        else:
            ny,nx = ldata[i].shape
            farr.append(RectBivariateSpline(np.arange(nx)-nx//2, np.arange(ny)-ny//2, ldata[i],kx=korder,ky=korder,s=0))

    # Higher-order X/Y terms
    if ndata>1:
        relx,rely = relcoord(x0,y0,shape)
        coeff = [1, relx, rely, relx*rely]
    else:
        coeff = [1]
        
    # Perform the interpolation
    g = np.zeros(dx.shape,float)
    # We must find the derivative with x0/y0 empirically
    if deriv:
        gxplus = np.zeros(dx.shape,float)
        gyplus = np.zeros(dx.shape,float)        
        xoff = 0.01
        yoff = 0.01
    for i in range(ndata):
        # spline is initialized with x,y, z(Nx,Ny)
        # and evaluated with f(x,y)
        # since we are using im(Ny,Nx), we have to evalute with f(y,x)
        g += farr[i](dy,dx,grid=False) * coeff[i]
        if deriv:
            gxplus += farr[i](dy,dx-xoff,grid=False) * coeff[i]
            gyplus += farr[i](dy-yoff,dx,grid=False) * coeff[i]            
    g *= amp
    if deriv:
        gxplus *= amp
        gyplus *= amp        
    
    if deriv is True:
        # We cannot use np.gradient() because the input x/y values
        # might not be a regular grid
        derivative = [g/amp,  (gxplus-g)/xoff, (gyplus-g)/yoff ]      
        return g,derivative
            
    # No derivative
    else:        
        return g


def psfmodel(name,pars=None,**kwargs):
    """
    Select PSF model based on the name.

    Parameters
    ----------
    name : str
      The type of PSF model to use: 'gaussian', 'moffat', 'penny', 'gausspow', 'empirical'.
    pars : numpy array
      The model parameters.
    kwargs : dictionary
      Any other keyword arguments that should be used to initialize the PSF model.

    Returns
    -------
    psf : PSF model
      The requested PSF model.

    Example
    -------

    psf = psfmodel('gaussian',[2.0, 2.5, 0.5])

    """

    if str(name).lower() in _models.keys():
        return _models[str(name).lower()](pars,**kwargs)
    else:
        raise ValueError('PSF type '+str(name)+' not supported.  Select '+', '.join(_models.keys()))
    

#######################
# PSF classes
#######################


# PSF base class
class PSFBase:

    def __init__(self,mpars,npix=51,binned=False,verbose=False):
        """
        Initialize the PSF model object.

        Parameters
        ----------
        mpars : numpy array
          PSF model parameters array.
        npix : int, optional
          Number of pixels to model [Npix,Npix].  Must be odd. Default is 51 pixels.
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
        self.lookup = None
        
        # add a precomputed circular mask here to mask out the corners??

        
    @property
    def params(self):
        """ Return the PSF model parameters."""
        return self._params

    @params.setter
    def params(self,value):
        """ Set the PSF model parameters."""
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
        if radius is None:
            radius = self.npix//2
        return starbbox(coords,imshape,radius)        
    
    def bbox2xy(self,bbox):
        """
        Convenience method to convert boundary box of X/Y limits to 2-D X and Y arrays.  The upper limits
        are EXCLUSIVE following the python convention.
        """
        return bbox2xy(bbox)
        
    def __call__(self,x=None,y=None,pars=None,mpars=None,bbox=None,nolookup=False,
                 deriv=False,**kwargs):
        """
        Generate a model PSF for the input X/Y value and parameters.  If no inputs
        are given, then a postage stamp PSF image is returned.
        This will include the contribution from the lookup table.

        Parameters
        ----------
        x and y: numpy array, optional
            The X and Y values for the images pixels for which you want to
            generate the model. These can be 1D or 2D arrays.
            The "bbox" parameter can be used instead of "x" and "y" if a rectangular
            region is desired.
        pars : numpy array, list or catalog
            Stellar arameters.  If numpy array or list the values should be [amp, xcen, ycen, sky].
            If a catalog is input then it must have amp, x, y and sky columns.
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
        nolookup : boolean, optional
            Do not use the lookup table if there is one.  Default is False.

        Returns
        -------
        model : numpy array
          Array of (1-D) model values for the input xdata and parameters.

        Example
        -------

        m = psf(x,y,pars)

        """

        # No coordinates input, PSF postage stamp
        if x is None and y is None and bbox is None:
            if pars is None:
                pars = [1.0, self.npix//2, self.npix//2]
            pix = np.arange(self.npix)-self.npix//2
            # Python images are (Y,X)
            y = (pix+pars[2]).reshape(-1,1)+np.zeros(self.npix,int)                # broadcasting is faster
            x = (pix+pars[1]).reshape(1,-1)+np.zeros(self.npix,int).reshape(-1,1)  # broadcasting is faster
            
        # Get coordinates from BBOX
        if x is None and y is None and bbox is not None:
            x,y = self.bbox2xy(bbox)

        if x is None or y is None:
            raise ValueError("X and Y or BBOX must be input")
        if pars is None:
            raise ValueError("PARS must be input")
        if type(pars) is Table:
            for n in ['amp','x','y','sky']:
                if n not in pars.columns:
                    raise ValueError('Input catalog must have amp, x, y, and sky columns')
            inpcat = pars
            pars = [inpcat['amp'][0],inpcat['x'][0],inpcat['y'][0],inpcat['sky'][0]]
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

        # Add the lookup component
        if self.haslookup and nolookup==False:
            lumodel = self.lookup(x,y,pars,deriv=deriv)
            if deriv:
                # [0] is the model, [1] is the list of derivatives
                out[0][:] += lumodel[0][:]
                out[0][:] = np.maximum(out[0],0.0)  # make sure it's non-negative                    
                for i in range(len(lumodel[1])):
                    out[1][i][:] += lumodel[1][i][:]
            else:
                out += lumodel
                out = np.maximum(out,0.0)  # make sure it's non-negative
                
        # Mask any corner pixels
        rr = np.sqrt((x-inpars[1])**2+(y-inpars[2])**2)
        if deriv:
            out[0][rr>self.radius] = 0
        else:
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
        This includes the contribution of the lookup table.

        Parameters
        ----------
        xdata : numpy array
            X and Y values in a [2,N] array.
        args : float
            Model parameter values as separate positional input parameters,
            [amp, xcen, ycen, sky].  If allpars=True, then the model
            parameters are added at the end, i.e. 
            [amp, xcen, ycen, sky, model parameters].
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
        # PARS should be [amp,x0,y0,sky]
        ## curve_fit separates each parameter while
        ## psf expects on pars array
        
        pars = np.copy(np.array(args))
        if self.verbose: print('model: ',pars)

        self.niter += 1

        # Just stellar parameters
        if not allpars:
            return self(xdata[0],xdata[1],pars,**kwargs)
        
        # Stellar + Model parameters
        #   PARS should be [amp,x0,y0,sky, model parameters]
        else:
            allpars = np.copy(np.array(args))
            nmpars = len(self.params)
            npars = len(allpars)-nmpars
            pars = allpars[0:npars]
            mpars = allpars[-nmpars:]
            # Constrain the model parameters to the PSF bounds
            lbnds,ubnds = self.bounds
            for i in range(len(self.params)):
                mpars[i] = np.minimum(np.maximum(mpars[i],lbnds[i]),ubnds[i])
            return self(xdata[0],xdata[1],pars,mpars=mpars,**kwargs)


    def jac(self,xdata,*args,retmodel=False,allpars=False,**kwargs):
        """
        Method to return Jacobian matrix.
        This includes the contribution of the lookup table.

        Parameters
        ----------
        xdata : numpy array
            X and Y values in a [2,N] array.
        args : float
            Model parameter values as separate positional input parameters,
            [amp, xcen, ycen, sky]. If allpars=True, then the model
            parameters are added at the end, i.e. 
            [amp, xcen, ycen, sky, model parameters].
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
        
        # PARS should be [amp,x0,y0,sky]        
        ## curve_fit separates each parameter while
        ## psf expects on pars array
        pars = np.array(args)
        if self.verbose: print('jac: ',pars)

        # Just stellar parameters
        if not allpars:
            if len(pars)==3:
                sky = None
            elif len(pars)==4:
                sky = pars[3]
            else:
                raise ValueError('PARS must have 3 or 4 parameters')
            nderiv = 3
            mpars = self.params
        # All parameters
        else:
            nmpars = len(self.params)
            mpars = pars[-nmpars:]
            npars = len(pars)-nmpars
            pars = pars[0:-nmpars]
            if npars==4:  # sky input
                sky = pars[3]
            else:
                sky = None           
            nderiv = None  # want all the derivatives
        
        # Get the derivatives and model
        m,deriv = self(xdata[0],xdata[1],pars,mpars=mpars,deriv=True,nderiv=nderiv,**kwargs)            
        deriv = np.array(deriv).T
        # Initialize jacobian matrix
        #   the parameters are [amp,xmean,ymean,sky]
        #   if allpars, parameters are [amp,xmean,ymean,sky,model parameters]
        jac = np.zeros((len(xdata[0]),len(args)),float)
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
        # PARS should be [amp,x0,y0,sky, model parameters]
        return self.model(xdata,*args,allpars=True,**kwargs)
        
    def jacall(self,xdata,*args,retmodel=False,**kwargs):
        """ Convenience function to use with curve_fit() to fit all parameters of a single stellar profile."""
        # PARS should be [amp,x0,y0,sky, model parameters]        
        return self.jac(xdata,*args,allpars=True,**kwargs)

    def linesearch(self,xdata,bestpar,dbeta,flux,wt,m,jac,allpars=False):
        # Perform line search along search gradient
        start_point = bestpar
        search_gradient = dbeta
        def obj_func(pp,m=None):
            """ chisq given the parameters."""
            if m is None:
                if allpars:
                    m = self.model(xdata,*pp,allpars=True)
                else:
                    m = self.model(xdata,*pp)                        
            chisq = np.sum((flux.ravel()-m.ravel())**2 * wt.ravel())
            return chisq
        def obj_grad(pp,m=None,jac=None):
            """ Gradient of chisq wrt the parameters."""
            if m is None and jac is None:
                if allpars:
                    m,jac = self.jac(xdata,*pp,allpars=True,retmodel=True)
                else:
                    m,jac = self.jac(xdata,*pp,retmodel=True)
            # d chisq / d parj = np.sum( 2*jac_ij*(m_i-d_i))/sig_i**2)
            dchisq = np.sum( 2*jac * (m.ravel()-flux.ravel()).reshape(-1,1)
                             * wt.ravel().reshape(-1,1),axis=0)
            return dchisq

        f0 = obj_func(start_point,m=m)
        # Do our own line search with three points and a quadratic fit.
        f1 = obj_func(start_point+0.5*search_gradient)
        f2 = obj_func(start_point+search_gradient)
        alpha = dln.quadratic_bisector(np.array([0.0,0.5,1.0]),np.array([f0,f1,f2]))
        #print('models alpha=',alpha)
        if ~np.isfinite(alpha):
            alpha = 1.0
        alpha = np.minimum(np.maximum(alpha,0.0),1.0)  # 0<alpha<1
        # Use scipy.optimize.line_search()
        #grad0 = obj_grad(start_point,m=m,jac=jac)
        #alpha,fc,gc,new_fval,old_fval,new_slope = line_search(obj_func, obj_grad, start_point, search_gradient, grad0,f0,maxiter=3)
        #if alpha is None:  # did not converge
        #    alpha = 1.0
        pars_new = start_point + alpha * search_gradient
        new_dbeta = alpha * search_gradient
        return alpha,new_dbeta
    
    def fit(self,im,pars,niter=2,radius=None,allpars=False,method='qr',nosky=False,
            minpercdiff=0.5,absolute=False,retpararray=False,retfullmodel=False,
            recenter=True,bounds=None,verbose=False):
        """
        Method to fit a single star using the PSF model.

        Parameters
        ----------
        im : CCDData object
            Image to use for fitting.
        pars : numpy array, list or catalog
            Initial parameters.  If numpy array or list the values should be [amp, xcen, ycen].
            If a catalog is input then it must at least the "x" and "y" columns.
        niter : int, optional
            Number of iterations to perform.  Default is 2.
        radius : float, optional
            Fitting radius in pixels.  Default is to use the PSF FWHM.
        allpars : boolean, optional
            Fit PSF model parameters as well.  Default is to only fit the stellar parameters
            of [amp, xcen, ycen, sky].
        method : str, optional
            Method to use for solving the non-linear least squares problem: "cholesky",
            "qr", "svd", and "curve_fit".  Default is "qr".
        minpercdiff : float, optional
           Minimum percent change in the parameters to allow until the solution is
           considered converged and the iteration loop is stopped.  Default is 0.5.
        nosky : boolean, optional
            Do not fit the sky, only [amp, xcen, and ycen].  Default is False.
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
        recenter : boolean, optional
            Allow the centroids to be fit.  Default is True.
        bounds : list, optional
            Input lower and upper bounds/constraints on the fitting parameters (tuple of two
              lists (e.g., ([amp_lo,x_low,y_low],[amp_hi,x_hi,y_hi])).
        verbose : boolean, optional
            Verbose output to the screen.  Default is False.

        Returns
        -------
        outcat : catalog or numpy array
            Output catalog of best-fit values (id, amp, amp_error, x, x_error, y, y_error,
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

        print = utils.getprintfunc() # Get print function to be used locally, allows for easy logging   
        
        # PARS: initial guesses for Xo and Yo parameters.
        if isinstance(pars,Table):
            for n in ['x','y','amp']:
                if n not in pars.columns:
                    raise ValueError('PARS must have [AMP, X, Y]')
            cat = {'amp':pars['amp'][0],'x':pars['x'][0],'y':pars['y'][0]}
        elif isinstance(pars,np.ndarray):
            for n in ['x','y','amp']:
                if n not in pars.dtype.names:
                    raise ValueError('PARS must have [AMP, X, Y]')            
            cat = {'amp':pars['amp'][0],'x':pars['x'][0],'y':pars['y'][0]}
        elif isinstance(pars,dict):
            if 'x' in pars.keys()==False or 'y' in pars.keys() is False:
                raise ValueError('PARS dictionary must have x and y')
            cat = pars
        else:            
            if len(pars)<3:
                raise ValueError('PARS must have [AMP, X, Y]')
            cat = {'amp':pars[0],'x':pars[1],'y':pars[2]}

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
        if 'amp' in cat:
            amp = cat['amp']
        else:
            amp = flux[int(np.round(yc)),int(np.round(xc))]-sky   # python images are (Y,X)
            amp = np.maximum(amp,1)  # make sure it's not negative
            
        initpar = [amp,xc,yc,sky]            
        
        # Fit PSF parameters as well
        if allpars:
            initpar = np.hstack(([amp,xc,yc,sky],self.params.copy()))

        # Remove sky column
        if nosky:
            initpar = np.delete(initpar,3,axis=0)

        # Initialize the output catalog
        dt = np.dtype([('id',int),('amp',float),('amp_error',float),('x',float),
                       ('x_error',float),('y',float),('y_error',float),('sky',float),
                       ('sky_error',float),('flux',float),('flux_error',float),
                       ('mag',float),('mag_error',float),('niter',int),
                       ('nfitpix',int),('rms',float),('chisq',float)])
        outcat = np.zeros(1,dtype=dt)
        outcat['id'] = 1

        # Make bounds
        if bounds is None:
            bounds = self.mkbounds(initpar,flux.shape)
        # Not fitting centroids
        if recenter==False:
            bounds[0][1] = initpar[1]-1e-7
            bounds[0][2] = initpar[2]-1e-7
            bounds[1][1] = initpar[1]+1e-7
            bounds[1][2] = initpar[2]+1e-7

        if verbose:
            print('initpar = ',initpar)
            
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
            if verbose:
                print('lbounds = ',bounds[0])
                print('ubounds = ',bounds[1])                
                print('maxsteps = ',maxsteps)
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
                chisq = np.sum(dy**2 * wt.ravel())/len(dy)
                
                # Perform line search
                alpha,new_dbeta = self.linesearch(xdata,bestpar,dbeta,flux,wt,m,jac,allpars=allpars)
                    
                # Update parameters
                oldpar = bestpar.copy()
                # limit the steps to the maximum step sizes and boundaries
                #bestpar = self.newpars(bestpar,dbeta,bounds,maxsteps)
                bestpar = self.newpars(bestpar,new_dbeta,bounds,maxsteps)                
                #bestpar += dbeta
                # Check differences and changes
                diff = np.abs(bestpar-oldpar)
                denom = np.maximum(np.abs(oldpar.copy()),0.0001)
                percdiff = diff.copy()/denom*100  # percent differences
                percdiff[1:3] = diff[1:3]*100               # x/y
                maxpercdiff = np.max(percdiff)
                
                if verbose:
                    print('N = '+str(count))
                    print('dbeta = '+str(dbeta))
                    print('bestpars = '+str(bestpar))
                    print('chisq = ',chisq)
                
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
        outcat['amp'] = bestpar[0]
        outcat['amp_error'] = perror[0]
        outcat['x'] = bestpar[1]
        outcat['x_error'] = perror[1]
        outcat['y'] = bestpar[2]
        outcat['y_error'] = perror[2]
        if not nosky:
            outcat['sky'] = bestpar[3]
            outcat['sky_error'] = perror[3]
        outcat['flux'] = bestpar[0]*self.flux()
        outcat['flux_error'] = perror[0]*self.flux()        
        outcat['mag'] = -2.5*np.log10(np.maximum(outcat['flux'],1e-10))+25.0
        outcat['mag_error'] = (2.5/np.log(10))*outcat['flux_error']/outcat['flux']
        outcat['niter'] = count
        outcat['nfitpix'] = flux.size
        outcat['chisq'] = np.sum((flux-model.reshape(flux.shape))**2/err**2)/len(flux)
        outcat = Table(outcat)
        # chi value, RMS of the residuals as a fraction of the amp
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


    def add(self,im,cat,sky=False,radius=None,nocopy=False):
        """
        Method to add stars using the PSF model from an image.

        Parameters
        ----------
        im : CCDData object
            Image to use for fitting.
        cat : catalog
            Catalog of stellar parameters.  Columns must include amp, x, y and sky.
        sky : boolean, optional
            Include sky in the model that is subtracted.  Default is False.
        radius : float, optional
            PSF radius to use.  The default is to use the full size of the PSF.
        nocopy: boolean, optional
            Return the original image with the stars added.  Default is False
              and a copy of the image will be returned.

        Returns
        -------
        addim : CCDData object
            Image with stellar models added.

        Example
        -------

        addim = psf.add(image,cat)

        """

        if isinstance(cat,np.ndarray):
            columns = cat.dtype.names
        elif isinstance(cat,dict):
            columns = cat.keys()
        elif isinstance(cat,Table):
            columns = cat.columns
        else:
            raise ValueError('Only ndarray, astropy Table or dictionaries supported for catalogs')

        for n in ['amp','x','y','sky']:
            if not n in columns:
                raise ValueError('Catalog must have amp, x, y and sky columns')
            
        ny,nx = im.shape    # python images are (Y,X)
        nstars = np.array(cat).size
        hpix = self.npix//2
        if radius is None:
            radius = self.radius
        else:
            radius = np.minimum(self.radius,radius)
        if nocopy:
            addim = im
        else:
            addim = np.copy(im.data)
        for i in range(nstars):
            pars = [cat['amp'][i],cat['x'][i],cat['y'][i]]
            if sky:
                pars.append(cat['sky'][i])
            bbox = self.starbbox((pars[1],pars[2]),im.shape,radius)
            im1 = self(pars=pars,bbox=bbox)
            addim[bbox.slices] += im1            
        return addim
                    
        
    def sub(self,im,cat,sky=False,radius=None,nocopy=False):
        """
        Method to subtract stars using the PSF model from an image.

        Parameters
        ----------
        im : CCDData object
            Image to use for fitting.
        cat : catalog
            Catalog of stellar parameters.  Columns must include amp, x, y and sky.
        sky : boolean, optional
            Include sky in the model that is subtracted.  Default is False.
        radius : float, optional
            PSF radius to use.  The default is to use the full size of the PSF.
        nocopy: boolean, optional
            Return the original image with the star subtracted.  Default is False
              and a copy of the image will be returned.

        Returns
        -------
        subim : numpy array
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

        for n in ['amp','x','y']:
            if not n in columns:
                raise ValueError('Catalog must have amp, x, and y columns')
        if sky and 'sky' not in columns:
            raise ValueError('Catalog must have sky column')
            
        ny,nx = im.shape    # python images are (Y,X)
        nstars = np.array(cat).size
        hpix = self.npix//2
        if radius is None:
            radius = self.radius
        else:
            radius = np.minimum(self.radius,radius)
        if nocopy:
            subim = im
        else:
            subim = np.copy(im.data)
        for i in range(nstars):
            pars = [cat['amp'][i],cat['x'][i],cat['y'][i]]
            if sky:
                pars.append(cat['sky'][i])
            bbox = self.starbbox((pars[1],pars[2]),im.shape,radius)
            im1 = self(pars=pars,bbox=bbox)
            subim[bbox.slices] -= im1
            
        return subim
                    

    def resid(self,cat,image,fillvalue=np.nan):
        """
        Produce a residual map of the cutout of the star (within the PSF footprint) and
        the best-fitting PSF.

        Parameters
        ----------
        cat : table
           The catalog of stars to use.  This should have "x" and "y" columns and
             preferably also "amp".
        image : CCDData object
           The image to use to generate the residuals images.
        fillvalue : float, optional
          The fill value to use for pixels that are bad are off the image.
            Default is np.nan.

        Returns
        -------
        resid : numpy array
           Three-dimension cube (Npix,Npix,Nstars) of the star images with the
             best-fitting PSF model subtracted.

        Example
        -------

        cube = psf.resid(cat,image)

        """

        # Get the residuals data
        nstars = len(cat)
        npix = self.npix
        nhpix = npix//2
        resid = np.zeros((npix,npix,nstars),float)
        xx,yy = np.meshgrid(np.arange(npix)-nhpix,np.arange(npix)-nhpix)
        rr = np.sqrt(xx**2+yy**2)        
        x = xx[0,:]
        y = yy[:,0]
        for i in range(nstars):
            xcen = cat['x'][i]            
            ycen = cat['y'][i]
            bbox = self.starbbox((xcen,ycen),image.shape,radius=nhpix)
            im = image[bbox.slices]
            flux = image.data[bbox.slices]-image.sky[bbox.slices]
            err = image.error[bbox.slices]
            if 'amp' in cat.columns:
                amp = cat['amp'][i]
            elif 'peak' in cat.columns:
                amp = cat['peak'][i]
            else:
                amp = flux[int(np.round(ycen)),int(np.round(xcen))]
            xim,yim = np.meshgrid(im.x,im.y)
            xim = xim.astype(float)-xcen
            yim = yim.astype(float)-ycen
            # We need to interpolate this onto the grid
            f = RectBivariateSpline(yim[:,0],xim[0,:],flux/amp)
            im2 = np.zeros((npix,npix),float)+np.nan
            xcover = (x>=bbox.ixmin-xcen) & (x<=bbox.ixmax-1-xcen)
            xmin,xmax = dln.minmax(np.where(xcover)[0])
            ycover = (y>=bbox.iymin-ycen) & (y<=bbox.iymax-1-ycen)
            ymin,ymax = dln.minmax(np.where(ycover)[0])            
            im2[ymin:ymax+1,xmin:xmax+1] = f(y[ycover],x[xcover],grid=True)
            # Get model
            model = self(pars=[1.0,0.0,0.0],bbox=[[-nhpix,nhpix+1],[-nhpix,nhpix+1]])
            # Stuff it into 3D array
            resid[:,:,i] = im2-model
        return resid
            
    def __str__(self):
        """ String representation of the PSF."""
        return self.__class__.__name__+'('+str(list(self.params))+',binned='+str(self.binned)+',npix='+str(self.npix)+',lookup='+str(self.haslookup)+') FWHM=%.2f' % (self.fwhm())

    def __repr__(self):
        """ String representation of the PSF."""        
        return self.__class__.__name__+'('+str(list(self.params))+',binned='+str(self.binned)+',npix='+str(self.npix)+',lookup='+str(self.haslookup)+') FWHM=%.2f' % (self.fwhm())

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
        """ Return the flux/volume of the model given the amp.  Must be defined by subclass."""
        pass

    # Do we also want the flux within the footprint!
    # could calculate the unit flux within the footprint the first time it's
    # called and save that.
    # could use fluxtot for the total flux
    # or fluxfoot for the footprint flux
    # or even have footprint=True to use the footprint flux
    
    def steps(self,pars=None,bounds=None,star=False):
        """
        Return step sizes to use when fitting the PSF model parameters (at least initial sizes).

        Parameters
        ----------
        pars : numpy array or list
          List or array of parameters for which to produce step sizes.
        bounds : tuple, optional
          Two-element tuple of lower and upper constrainst on pars.
        star : boolean, optional
          Stellar parameters are included.  Default is False.

        Returns
        -------
        steps : numpy array
          Array of step sizes.

        Example
        -------
        steps = psf.steps(pars)

        """
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
                initsteps[3] = np.maximum(pars[3]*0.5,50)     # sky
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
        """
        Make bounds for a set of input parameters.

        Parameters
        ----------
        pars : numpy array or list
          List or array of parameters for which to produce constraints.
        imshape : tuple
          Two-element tuple of the image size.

        Returns
        -------
        bounds : tuple
          Two-element tuple of lower and upper constraints in pars.

        Example
        -------

        bounds = psf.mkbounds(pars)

        """

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
        """
        Check the parameters against the bounds.

        Parameters
        ----------
        pars : numpy array or list
          List or array of parameters for which to check the constraints.
        bounds : tuple, optional
          Two-element tuple of lower and upper constrainst on pars.

        Returns
        -------
        check : numpy array
          Integer array indicating if the parameter crossed the boundaries.
           0-fine, 1-beyond the lower bound; 2-beyond the upper bound.

        Example
        -------

        check = psf.checkbounds(pars,bounds)

        """
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
        """
        Limit the parameters to the boundaries.

        Parameters
        ----------
        pars : numpy array or list
          List or array of parameters.
        bounds : tuple, optional
          Two-element tuple of lower and upper constrainst on pars.

        Returns
        -------
        outpars : numpy array
          Array of output parameters that are limited to the bounds.

        Example
        -------

        outpars = psf.limbounds(pars,bounds)

        """
        if bounds is None:
            bounds = self.mkbounds(pars)
        lbounds,ubounds = bounds
        outpars = np.minimum(np.maximum(pars,lbounds),ubounds)
        return outpars

    def limsteps(self,steps,maxsteps):
        """
        Limit the parameter steps to maximum step sizes.

        Parameters
        ----------
        steps : numpy array
          Array of step sizes to limit.
        maxstep : numpy array
          Array of maximum step sizes to limit the input steps to.

        Returns
        -------
        outsteps : numpy array
           array of step sizes that have been limited to the maximum values.

        Example
        -------

        outsteps = psf.limsteps(steps,maxsteps)

        """
        signs = np.sign(steps)
        outsteps = np.minimum(np.abs(steps),maxsteps)
        outsteps *= signs
        return outsteps

    def newpars(self,pars,steps,bounds,maxsteps):
        """
        Get new parameters given initial parameters, steps and constraints.
        
        Parameters
        ----------
        pars : numpy array or list
          List or array of initial parameters.
        steps : numpy array
          Array of steps to add to pars.
        bounds : tuple, optional
          Two-element tuple of lower and upper constrainst on pars.
        maxstep : numpy array
          Array of maximum step sizes to limit the input steps to.     

        Returns
        -------
        newpars : numpy array or list
          Array of new parameters that have been incremented by
           steps but limited by bounds and maxsteps.

        Example
        -------

        newpars = psf.newpars(pars,steps,bounds,maxsteps)

        """

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

    def trim(self,trimflux):
        """ Trim the PSF size to a radius where "trimflux" is removed."""
        xx,yy = np.meshgrid(np.arange(self.npix)-self.npix//2,np.arange(self.npix)-self.npix//2)
        rr = np.sqrt(xx**2+yy**2)
        im = self()
        r = rr.ravel()
        f = im.ravel()
        si = np.argsort(r)
        rsi = r[si]
        fsi = f[si]
        totflux = np.sum(im)
        cfsi = np.cumsum(fsi)/totflux
        ind = np.max(np.where(cfsi<(1-trimflux))[0])
        rtrim = rsi[ind]
        rtrim = int(np.ceil(rtrim))
        newnpix = np.minimum(2*rtrim+1,self.npix)
        if newnpix!=self.npix:
            newradius = newnpix//2
            off = (self.npix-newnpix)//2
            if type(self)==PSFEmpirical:
                data = self._data
                if data.ndim==2:
                    data = data[off:-off,off:-off]
                elif data.ndim==3:
                    data = data[off:-off,off:-off,:]
                # Re-initialize a temporary PSF to generate the splines
                temp = PSFEmpirical(data,imshape=self._shape,korder=self._fpars[0].degrees[0],
                                    npix=newnpix,order=self.order,lookup=self.lookup)
                self._data = data
                self._fpars = copy.deepcopy(temp._fpars)
                del temp
            self.npix = newnpix
            self.radius = newradius
            self._unitfootflux = np.sum(self())
        
    @property
    def haslookup(self):
        """ Check if there is a lookup table."""
        return (self.lookup is not None)
    
    @classmethod
    def read(cls,filename,exten=0):
        """ Load a PSF file."""
        if os.path.exists(filename)==False:
            raise ValueError(filename+' NOT FOUND')
        hdulist = fits.open(filename)
        data,head = hdulist[exten].data,hdulist[exten].header
        psftype = head.get('PSFTYPE')
        if psftype is None:
            raise ValueError('No PSFTYPE found in header')
        kwargs = {}        
        binned = head.get('BINNED')
        if binned is not None: kwargs['binned'] = binned
        npix = head.get('NPIX')
        if npix is not None: kwargs['npix'] = npix
        if psftype.lower()=='empirical':
            yshape = head.get('YSHAPE')
            xshape = head.get('XSHAPE')
            if yshape is not None and xshape is not None:
                imshape = (yshape,xshape)
            else:
                imshape = None
            kwargs['imshape'] = imshape
            kwargs['korder'] = head.get('KORDER')
        newpsf = psfmodel(psftype,data,**kwargs)
        # Lookup table
        if head.get('HSLOOKUP') and len(hdulist)>1:
            ludata,luhead = hdulist[exten+1].data,hdulist[exten+1].header
            lukwargs = {}
            yshape = luhead.get('YSHAPE')
            xshape = luhead.get('XSHAPE')
            if yshape is not None and xshape is not None:
                imshape = (yshape,xshape)
            else:
                imshape = None
            lukwargs['imshape'] = imshape
            lukwargs['korder'] = luhead.get('KORDER')
            lookup = psfmodel('empirical',ludata,**lukwargs,lookup=True)
            newpsf.lookup = lookup
        return newpsf

    def tohdu(self):
        """ Convert the PSF object to an HDU. Defined by subclass."""
        pass
    
    def write(self,filename,overwrite=True):
        """ Write a PSF to a file.  Defined by subclass."""
        pass

    def thumbnail(self,filename=None,figsize=6):
        """
        Generate a thumbnail image of the PSF.

        Parameters
        ----------
        filename : str, optional
           Filename of the output thumbnail file.  Default is "psf.png".
        figsize : float, optional
           The figure size in inches.  Default is 6.

        Returns
        -------
        The PSF thumbnail is saved to a file.

        Example
        -------

        psf.thumbnail('thumbnai.png')

        """
        
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
    def __init__(self,mpars=None,npix=51,binned=False):
        # MPARS are the model parameters
        #  mpars = [xsigma, ysigma, theta]
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
        # Labels
        self.labels = ['xsigma','ysigma','theta']
        
    def fwhm(self,pars=None):
        """ Return the FWHM of the model."""
        if pars is None:
            pars = np.hstack(([1.0,0.0,0.0],self.params))
        return gaussian2d_fwhm(pars)

    def flux(self,pars=None,footprint=False):
        """ Return the flux/volume of the model given the amp or parameters."""
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
        This does not include the lookup table.

        Returns
        -------
        hdu : fits HDU object
          The FITS HDU object.

        Example
        -------

        hdu = psf.tohdu()

        """
        hdulist = []   # HDUList always makes the first HDU a PrimaryHDU which can problems
        hdulist.append(fits.ImageHDU(self.params))
        hdulist[0].header['EXTNAME'] = 'PSF MODEL'      
        hdulist[0].header['COMMENT'] = 'Prometheus PSF model'
        hdulist[0].header['PSFTYPE'] = 'Gaussian'
        hdulist[0].header['BINNED'] = self.binned
        hdulist[0].header['NPIX'] = self.npix
        hdulist[0].header['HSLOOKUP'] = self.haslookup
        if self.haslookup:
            luhdu = self.lookup.tohdu()
            hdulist.append(luhdu)
            hdulist[1].header['EXTNAME'] = 'PSF MODEL LOOKUP'
            hdulist[1].header['COMMENT'] = 'Prometheus PSF model lookup table'
            hdulist[1].header['LOOKUP'] = 1
        return fits.HDUList(hdulist)
    
    def write(self,filename,overwrite=True):
        """ Write a PSF to a file."""
        if os.path.exists(filename) and overwrite==False:
            raise ValueError(filename+' already exists and overwrite=False')
        hdulist = fits.HDUList()
        hdulist.append(fits.PrimaryHDU(self.params))
        hdulist[0].header['PSFTYPE'] = 'Gaussian'
        hdulist[0].header['BINNED'] = self.binned
        hdulist[0].header['NPIX'] = self.npix
        hdulist[0].header['HSLOOKUP'] = self.haslookup
        if self.haslookup:
            luhdu = self.lookup.tohdu()
            hdulist.append(luhdu)
            hdulist[1].header['LOOKUP'] = 1
        hdulist.writeto(filename,overwrite=overwrite)
        hdulist.close()

        
# PSF Moffat class
class PSFMoffat(PSFBase):

    # add separate X/Y sigma values and cross term like in DAOPHOT
    
    
    # Initalize the object
    def __init__(self,mpars=None,npix=51,binned=False):
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
        # Labels
        self.labels = ['xsigma','ysigma','theta','beta']
        
    def fwhm(self,pars=None):
        """ Return the FWHM of the model."""
        if pars is None:
            pars = np.hstack(([1.0,0.0,0.0],self.params))
        return moffat2d_fwhm(pars)

    def flux(self,pars=None,footprint=False):
        """ Return the flux/volume of the model given the amp or parameters."""
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
        This does not include the lookup table.

        Returns
        -------
        hdu : fits HDU object
          The FITS HDU object.

        Example
        -------

        hdu = psf.tohdu()

        """
        hdulist = []   # HDUList always makes the first HDU a PrimaryHDU which can problems
        hdulist.append(fits.ImageHDU(self.params))
        hdulist[0].header['PSFTYPE'] = 'Moffat'
        hdulist[0].header['BINNED'] = self.binned
        hdulist[0].header['NPIX'] = self.npix
        hdulist[0].header['HSLOOKUP'] = self.haslookup
        if self.haslookup:
            luhdu = self.lookup.tohdu()
            hdulist.append(luhdu)
            hdulist[1].header['LOOKUP'] = 1
        return fits.HDUList(hdulist)

    def write(self,filename,overwrite=True):
        """ Write a PSF to a file."""
        if os.path.exists(filename) and overwrite==False:
            raise ValueError(filename+' already exists and overwrite=False')
        hdulist = fits.HDUList()
        hdulist.append(fits.PrimaryHDU(self.params))
        hdulist[0].header['PSFTYPE'] = 'Moffat'
        hdulist[0].header['BINNED'] = self.binned
        hdulist[0].header['NPIX'] = self.npix
        hdulist[0].header['HSLOOKUP'] = self.haslookup
        if self.haslookup:
            luhdu = self.lookup.tohdu()
            hdulist.append(luhdu)
            hdulist[1].header['LOOKUP'] = 1                
        hdulist.writeto(filename,overwrite=overwrite)
        hdulist.close()
    
    
# PSF Penny class
class PSFPenny(PSFBase):
    """ Gaussian core and Lorentzian wings, only Gaussian is tilted."""
    # PARS are model parameters
    
    # Initalize the object
    def __init__(self,mpars=None,npix=51,binned=False):
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
        self._bounds = (np.array([0.0,0.0,-np.inf,0.00,0.01]),
                        np.array([np.inf,np.inf,np.inf,1.0,np.inf]))
        # Set step sizes
        self._steps = np.array([0.5,0.5,0.2,0.1,0.5])
        # Labels
        self.labels = ['xsigma','ysigma','theta','relamp','sigma']
        
    def fwhm(self,pars=None):
        """ Return the FWHM of the model."""
        if pars is None:
            pars = np.hstack(([1.0,0.0,0.0],self.params))
        return penny2d_fwhm(pars)

    def flux(self,pars=None,footprint=False):
        """ Return the flux/volume of the model given the amp or parameters."""
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
        This does not include the lookup table.

        Returns
        -------
        hdu : fits HDU object
          The FITS HDU object.

        Example
        -------

        hdu = psf.tohdu()

        """
        hdulist = []   # HDUList always makes the first HDU a PrimaryHDU which can problems
        hdulist.append(fits.ImageHDU(self.params))
        hdulist[0].header['PSFTYPE'] = 'Penny'
        hdulist[0].header['BINNED'] = self.binned
        hdulist[0].header['NPIX'] = self.npix
        hdulist[0].header['HSLOOKUP'] = self.haslookup
        if self.haslookup:
            luhdu = self.lookup.tohdu()
            hdulist.append(luhdu)
            hdulist[1].header['LOOKUP'] = 1
        return fits.HDUList(hdulist)

    def write(self,filename,overwrite=True):
        """ Write a PSF to a file."""
        if os.path.exists(filename) and overwrite==False:
            raise ValueError(filename+' already exists and overwrite=False')
        hdulist = fits.HDUList()
        hdulist.append(fits.PrimaryHDU(self.params))
        hdulist[0].header['PSFTYPE'] = 'Penny'
        hdulist[0].header['BINNED'] = self.binned
        hdulist[0].header['NPIX'] = self.npix
        hdulist[0].header['HSLOOKUP'] = self.haslookup
        if self.haslookup:
            luhdu = self.lookup.tohdu()
            hdulist.append(luhdu)
            hdulist[1].header['LOOKUP'] = 1        
        hdulist.writeto(filename,overwrite=overwrite)
        hdulist.close()

       
# PSF Ellipower class
class PSFGausspow(PSFBase):
    """ DoPHOT PSF, sum of Gaussian ellipses."""
    # PARS are model parameters
    
    # Initalize the object
    def __init__(self,mpars=None,npix=51,binned=False):
        # mpars = [sigx,sigy,sigxy,beta4,beta6]
        if mpars is None:
            mpars = np.array([2.5,2.5,0.0,1.0,1.0])
        if len(mpars)!=5:
            old = np.array(mpars).copy()
            mpars = np.array([2.5,2.5,0.0,1.0,1.0])
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
        # Labels
        self.labels = ['xsigma','ysigma','xysigma','beta4','beta6']
        
    def fwhm(self,pars=None):
        """ Return the FWHM of the model."""
        if pars is None:
            pars = np.hstack(([1.0,0.0,0.0],self.params))
        return gausspow2d_fwhm(pars)

    def flux(self,pars=None,footprint=False):
        """ Return the flux/volume of the model given the amp or parameters."""
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
        hdulist = []   # HDUList always makes the first HDU a PrimaryHDU which can problems
        hdulist.append(fits.ImageHDU(self.params))
        hdulist[0].header['PSFTYPE'] = 'Gausspow'
        hdulist[0].header['BINNED'] = self.binned
        hdulist[0].header['NPIX'] = self.npix
        hdulist[0].header['HSLOOKUP'] = self.haslookiup
        if self.haslookup:
            luhdu = self.lookup.tohdu()
            hdulist.append(luhdu)
            hdulist[1].header['LOOKUP'] = 1
        return fits.HDUList(hdulist)
    
    def write(self,filename,overwrite=True):
        """ Write a PSF to a file."""
        if os.path.exists(filename) and overwrite==False:
            raise ValueError(filename+' already exists and overwrite=False')
        hdulist = fits.HDUList()
        hdulist.append(fits.PrimaryHDU(self.params))
        hdulist[0].header['PSFTYPE'] = 'Gausspow'
        hdulist[0].header['BINNED'] = self.binned
        hdulist[0].header['NPIX'] = self.npix
        hdulist[0].header['HSLOOKUP'] = self.haslookup
        if self.haslookup:
            luhdu = self.lookup.tohdu()
            hdulist.append(luhdu)
            hdulist[1].header['LOOKUP'] = 1        
        hdulist.writeto(filename,overwrite=overwrite)
        hdulist.close()
    
    
# Sersic class
class Sersic(PSFBase):
    """ Sersic function."""
    # PARS are model parameters
    
    # Initalize the object
    def __init__(self,mpars=None,npix=51,binned=False):
        # mpars = [k,alpha,recc,theta]
        if mpars is None:
            mpars = np.array([1.0,1.0,1.0,0.0])
        if len(mpars)!=4:
            old = np.array(mpars).copy()
            mpars = np.array([1.0,1.0,0.8,0.0])
            mpars[0:len(old)] = old
        if mpars[0]<=0:
            raise ValueError('k must be >0')
        if mpars[1]<=0:
            raise ValueError('alpha must be >0')
        if mpars[2]<0 or mpars[2]>1:
            raise ValueError('relative amplitude must be >=0 and <=1')
        super().__init__(mpars,npix=npix,binned=binned)
        # Set the bounds
        self._bounds = (np.array([0.0,0.0,0.0,-np.inf]),
                        np.array([np.inf,np.inf,1.0,np.inf]))
        # Set step sizes
        self._steps = np.array([0.5,0.5,0.1,0.2])
        # Labels
        self.labels = ['k','alpha','recc','theta']
        
    def fwhm(self,pars=None):
        """ Return the FWHM of the model."""
        if pars is None:
            pars = np.hstack(([1.0,1.0,1.0,0.0],self.params))
        return sersic2d_fwhm(pars)

    def flux(self,pars=None,footprint=False):
        """ Return the flux/volume of the model given the amp or parameters."""
        if pars is None:
            pars = np.hstack(([1.0,1.0,1.0,0.0], self.params))
        else:
            pars = np.atleast_1d(pars)
            if pars.size==1:
                pars = np.hstack(([pars[0],1.0,1.0,0.0], self.params))
        if footprint:
            return self.unitfootflux*pars[0]
        else:
            return sersic2d_flux(pars)

    def estimates(self,epars):
        # calculate estimates for the Sersic parameters using
        # epars = [peak, x0, y0, flux, asemi, bsemi, theta]
        # Sersic Parameters are [amp,x0,y0,k,alpha,recc,theta]
        return sersic2d_estimates(epars)
        
    def evaluate(self,x, y, pars=None, binned=None, deriv=False, nderiv=None):
        """Two dimensional Sersicy model function"""
        # pars = [amplitude, x0, y0, k, alpha, recc, theta]
        if pars is None: pars = self.params
        if binned is None: binned = self.binned
        if binned is True:
            return sersic2d_integrate(x, y, pars, deriv=deriv, nderiv=nderiv)
        else:
            return sersic2d(x, y, pars, deriv=deriv, nderiv=nderiv)

    def deriv(self,x, y, pars=None, binned=None, nderiv=None):
        """Two dimensional Sersic model derivative with respect to parameters"""
        if pars is None: pars = self.params
        if binned is None: binned = self.binned
        if binned is True:
            g, derivative = sersic2d_integrate(x, y, pars, deriv=True, nderiv=nderiv)
        else:
            g, derivative = sersic2d(x, y, pars, deriv=True, nderiv=nderiv)
        return derivative

    def tohdu(self):
        """
        Convert the Sersic object to an HDU so it can be written to a file.
        This does not include the lookup table.

        Returns
        -------
        hdu : fits HDU object
          The FITS HDU object.

        Example
        -------

        hdu = psf.tohdu()

        """
        hdu = fits.ImageHDU(self.params)
        hdu.header['MTYPE'] = 'Sersic'
        hdu.header['BINNED'] = self.binned
        hdu.header['NPIX'] = self.npix
        return hdu
    
    def write(self,filename,overwrite=True):
        """ Write a Sersic object to a file."""
        if os.path.exists(filename) and overwrite==False:
            raise ValueError(filename+' already exists and overwrite=False')
        hdulist = fits.HDUList()
        hdulist.append(fits.PrimaryHDU(self.params))
        hdulist[0].header['MTYPE'] = 'Sersic'
        hdulist[0].header['BINNED'] = self.binned
        hdulist[0].header['NPIX'] = self.npix
        hdulist[0].header['HSLOOKUP'] = True
        hdulist.writeto(filename,overwrite=overwrite)
        hdulist.close()

    
class PSFEmpirical(PSFBase):
    """ Empirical look-up table PSF, can vary spatially."""

    # Initalize the object
    def __init__(self,mpars,imshape=None,korder=3,npix=51,binned=True,order=0,lookup=False):
        if mpars is None:
            mpars = PSFGaussian(npix=npix)()  # initialize with Gaussian of 1
            if order==1:
                mpars0 = mpars.copy()
                mpars = np.zeros((npix,npix,4),float)
                mpars[:,:,0] = mpars0
                if imshape is None:
                    imshape = (2048,2048)  # dummy image shape
        # List of RectBivariateSpline objects input
        if type(mpars) is list:
            nx = mpars[0].tck[0].shape
            ny = mpars[0].tck[1].shape
            npix = ny
            npars = len(mpars)
            if npars==1:
                order = 0
            elif npars==4:
                order = 1
            else:
                raise ValueError('Only order = 0 (1 term) or 1 (4 terms) supported at this time')
        elif mpars.ndim==2:
            npix,nx = mpars.shape
            npars = 1
            order = 0
        elif mpars.ndim==3:
            npix,nx,npars = mpars.shape    # Python images are (Y,X)                    
            order = 1
        else:
            raise ValueError('Input not understood')
        # Need image shape if there are higher-order terms
        if order==1 and imshape is None:
            raise ValueError('Image shape must be input for spatially varying PSF')
        super().__init__([],npix=npix,binned=True)
        self.npix = npix
        self.order = order
        self.islookup = lookup  # is this a lookup table
        self._data = mpars
        self._npars = npars
        self._korder = korder
        fpars = []
        x = np.arange(npix)-npix//2
        for i in range(npars):
            # spline is initialized with x,y, z(Nx,Ny)
            # and evaluated with f(x,y)
            # since we are using im(Ny,Nx), we have to evaluate with f(y,x)
            if mpars.ndim==2:
                fpars.append(RectBivariateSpline(x,x,mpars,kx=korder,ky=korder,s=0))
            else:
                fpars.append(RectBivariateSpline(x,x,mpars[:,:,i],kx=korder,ky=korder,s=0))
        self._fpars = fpars
        # image shape
        if imshape is not None:
            self._shape = imshape
        else:
            self._shape = None
        self._bounds = None
        self._steps = None

    def __str__(self):
        if self.islookup==False:
            return self.__class__.__name__+'(npix='+str(self.npix)+') FWHM=%.2f' % (self.fwhm())
        else:
            return self.__class__.__name__+'(npix='+str(self.npix)+')'            
        
    def __repr__(self):
        if self.islookup==False:
            return self.__class__.__name__+'(npix='+str(self.npix)+') FWHM=%.2f' % (self.fwhm())
        else:
            return self.__class__.__name__+'(npix='+str(self.npix)+')'                    
        
    def fwhm(self):
        """ Return the FWHM of the model."""
        # get contour at half max and then get average radius
        if self.islookup==False:
            return contourfwhm(self())
        else:
            return 0.0

    def flux(self,pars=None,footprint=True):
        """ Return the flux/volume of the model given the amp or parameters."""
        if pars is None:
            pars = [1.0, 0.0, 0.0]
        else:
            pars = np.atleast_1d(pars)
            if pars.size==1:
                pars = [pars[0], 0.0, 0.0]
        if self._unitfootflux is None:
            dum = self.unitfootflux
        return self.unitfootflux*pars[0]
    
    def evaluate(self,x, y, pars=None, data=None, deriv=False, nderiv=None):
        """Empirical look-up table"""
        # pars = [amplitude, x0, y0]
        if pars is None:
            raise ValueError('PARS must be input')
        if data is None: data = self._fpars
        return empirical(x, y, pars, data=data, shape=self._shape, deriv=deriv)

    def deriv(self,x, y, pars=None, data=None, nderiv=None):
        """Empirical look-up table derivative with respect to parameters"""
        if pars is None:
            raise ValueError('PARS must be input')        
        if data is None: data = self._fpars
        g,derivative = empirical(x, y, pars, data=data, shape=self._shape, deriv=True)
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
        hdu = fits.ImageHDU(self._data)
        hdu.header['PSFTYPE'] = 'Empirical'
        hdu.header['NPIX'] = self.npix
        hdu.header['BINNED'] = self.binned
        hdu.header['ORDER'] = self.order
        if self._shape is not None:
            hdu.header['YSHAPE'] = self._shape[0]
            hdu.header['XSHAPE'] = self._shape[1]     
        hdu.header['KORDER'] = self._korder       
        return hdu
    
    def write(self,filename,overwrite=True):
        """ Write a PSF to a file."""
        if os.path.exists(filename) and overwrite==False:
            raise ValueError(filename+' already exists and overwrite=False')
        hdulist = fits.HDUList()
        hdulist.append(fits.PrimaryHDU(self._data))
        hdulist[0].header['PSFTYPE'] = 'Empirical'
        hdulist[0].header['NPIX'] = self.npix
        hdulist[0].header['BINNED'] = self.binned
        hdulist[0].header['ORDER'] = self.order
        if self._shape is not None:
            hdulist[0].header['YSHAPE'] = self._shape[0]
            hdulist[0].header['XSHAPE'] = self._shape[1]     
        hdulist[0].header['KORDER'] = self._korder        
        hdulist.writeto(filename,overwrite=overwrite)
        hdulist.close()


# Read function
read = PSFBase.read    
    
# List of models
_models = {'gaussian':PSFGaussian,'moffat':PSFMoffat,'penny':PSFPenny,'gausspow':PSFGausspow,'empirical':PSFEmpirical}

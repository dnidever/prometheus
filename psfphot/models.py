
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
import getpsf

# A bunch of the Gaussian2D and Moffat2D code comes from astropy's modeling module
# https://docs.astropy.org/en/stable/_modules/astropy/modeling/functional_models.html

# Maybe x0/y0 should NOT be part of the parameters, and
# x/y should actually just be dx/dy (relative to x0/y0)

def gaussian2d(x,y,pars,deriv=False,nderiv=None):
    """Two dimensional Gaussian model function"""
    # pars = [amplitude, x0, y0, a, b, c]

    # Input A, B and C parameters instead
    # it will speed up the function since we don't have
    #  to compute anything
    # We can always compute sigma_minor, sigma_major, theta later
    #  if we want.

    xdiff = x - pars[1]
    ydiff = y - pars[2]
    amp = pars[0]
    a = pars[3]
    b = pars[4]
    c = pars[5]
    g = pars[0] * np.exp(-0.5*((a * xdiff ** 2) + (b * ydiff ** 2) +
                               (c * xdiff * ydiff)))

    
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
            dg_dx_mean = g * 0.5*((2 * a * xdiff) + (c * ydiff))
            derivative.append(dg_dx_mean)
        if nderiv>=3:
            dg_dy_mean = g * 0.5*((2 * b * ydiff) + (c * xdiff))
            derivative.append(dg_dy_mean)
        if nderiv>=4:       
            dg_da = g * (-0.5) * xdiff ** 2
            derivative.append(dg_da)
        if nderiv>=5:       
            dg_db = g * (-0.5) * ydiff ** 2
            derivative.append(dg_db)
        if nderiv>=6:       
            dg_dc = g * (-0.5) * xdiff * ydiff
            derivative.append(dg_dc)

        return g,derivative
            
    # No derivative
    else:        
        return g

def gaussian2d_abc2sigtheta(a,b,c):
    """ Convert 2D Gaussian a, b, c coefficients to sigma_x, sigma_y and theta."""
    
    # xdiff = x-x0
    # ydiff = y-y0
    # f(x,y) = A*exp(-0.5 * (a*xdiff**2 + b*ydiff**2 + c*xdiff*ydiff))
    
    # a is x**2 term
    # b is y**2 term
    # c is x*y term

    #cost2 = np.cos(theta) ** 2
    #sint2 = np.sin(theta) ** 2
    #sin2t = np.sin(2. * theta)
    #xstd2 = x_stddev ** 2
    #ystd2 = y_stddev ** 2
    #a = ((cost2 / xstd2) + (sint2 / ystd2))
    #b = ((sint2 / xstd2) + (cost2 / ystd2))    
    #c = ((sin2t / xstd2) - (sin2t / ystd2))

    # a+b = 1/xstd2 + 1/ystd2
    # c = sin2t * (1/xstd2 + 1/ystd2)
    # tan 2*theta = c/(a-b)
    theta = np.arctan2(c,a-b)/2.0

    sin2t = np.sin(2.0*theta)
    # c/sin2t + (a+b) = 2/xstd2
    # xstd2 = 2.0/(c/sin2t + (a+b))
    xstd = np.sqrt( 2.0/(c/sin2t + (a+b)) )

    # a+b = 1/xstd2 + 1/ystd2
    ystd = np.sqrt( 1/(a+b-1/xstd**2) )

    return xstd,ystd,theta


def gaussian2d_fwhm(pars):
    """ Return the FWHM of a 2D Gaussian."""
    # pars = [amplitude, x0, y0, a, b, c]

    # xdiff = x-x0
    # ydiff = y-y0
    # f(x,y) = A*exp(-0.5 * (a*xdiff**2 + b*ydiff**2 + c*xdiff*ydiff))
    
    # a is x**2 term
    # b is y**2 term
    # c is x*y term
    
    #cost2 = np.cos(theta) ** 2
    #sint2 = np.sin(theta) ** 2
    #sin2t = np.sin(2. * theta)
    #xstd2 = x_stddev ** 2
    #ystd2 = y_stddev ** 2
    #a = ((cost2 / xstd2) + (sint2 / ystd2))
    #b = ((sint2 / xstd2) + (cost2 / ystd2))    
    #c = ((sin2t / xstd2) - (sin2t / ystd2))

    a = pars[3]
    b = pars[4]
    c = pars[5]

    xstd,ystd,theta = gaussian2d_abc2sigtheta(a,b,c)

    # The mean radius of an ellipse is: (2a+b)/3
    sig_major = np.max([xstd,ystd])
    sig_minor = np.min([xstd,ystd])
    mnsig = (2.0*sig_major+sig_minor)/3.0
    # Convert sigma to FWHM
    # FWHM = 2*sqrt(2*ln(2))*sig ~ 2.35482*sig
    fwhm = mnsig*2.35482

    return fwhm


def gaussian2d_flux(pars):
    """ Return the total Flux of a 2D Gaussian."""
    # Volume is 2*pi*A*sigx*sigy

    amp = pars[0]
    a = pars[3]
    b = pars[4]
    c = pars[5]

    xstd,ystd,theta = gaussian2d_abc2sigtheta(a,b,c)

    volume = 2*np.pi*amp*xstd*ystd
    
    return volume
    
    
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
    # pars = [amplitude, x0, y0, a, b, c, beta]

    amp = pars[0]
    xdiff = x - pars[1]
    ydiff = y - pars[2]
    a = pars[3]
    b = pars[4]
    c = pars[5]
    beta = pars[6]

    rr_gg = (a * xdiff ** 2) + (b * ydiff ** 2) + (c * xdiff * ydiff)
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
            dg_dx_0 = beta*g/(1+rr_gg) * (2*a*xdiff + c*ydiff)
            derivative.append(dg_dx_0)            
        if nderiv>=3:
            dg_dy_0 = beta*g/(1+rr_gg) * (2*b*ydiff + c*xdiff)
            derivative.append(dg_dy_0)            
        if nderiv>=4:
            dg_da = (-beta)*g/(1+rr_gg) * xdiff**2
            derivative.append(dg_da)
        if nderiv>=5:
            dg_db = (-beta)*g/(1+rr_gg) * ydiff**2
            derivative.append(dg_db)            
        if nderiv>=6:
            dg_dc = (-beta)*g/(1+rr_gg) * xdiff*ydiff
            derivative.append(dg_dc)
        if nderiv>=7:            
            dg_dbeta = -g * np.log(1 + rr_gg)
            derivative.append(dg_dbeta) 
            
        return g,derivative

    # No derivative
    else:
        return g

def moffat2d_fwhm(pars):
    """ Return the FWHM of a 2D Moffat function."""
    # [amplitude, x0, y0, a, b, c, beta]
    # https://nbviewer.jupyter.org/github/ysbach/AO_2017/blob/master/04_Ground_Based_Concept.ipynb#1.2.-Moffat

    a = pars[3]
    b = pars[4]
    c = pars[5]
    beta = pars[6]
    
    xstd,ystd,theta = gaussian2d_abc2sigtheta(a,b,c)

    # The mean radius of an ellipse is: (2a+b)/3
    sig_major = np.max([xstd,ystd])
    sig_minor = np.min([xstd,ystd])
    mnsig = (2.0*sig_major+sig_minor)/3.0
    
    return 2.0 * np.abs(mnsig) * np.sqrt(2.0 ** (1.0/beta) - 1.0)


def moffat2d_flux(pars):
    """ Return the total Flux of a 2D Moffat."""
    # [amplitude, x0, y0, a, b, c, beta]
    # Volume is 2*pi*A*sigx*sigy
    # area of 1D moffat function is pi*alpha**2 / (beta-1)
    # maybe the 2D moffat volume is (xsig*ysig*pi**2/(beta-1))**2

    amp = pars[0]
    a = pars[3]
    b = pars[4]
    c = pars[5]
    beta = pars[6]

    xstd,ystd,theta = gaussian2d_abc2sigtheta(a,b,c)

    # This worked for beta=2.5, but was too high by ~1.05-1.09 for beta=1.5
    #volume = amp * xstd*ystd*np.pi/(beta-1)
    volume = amp * xstd*ystd*np.pi/(beta-1)
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


# Don't need this!  Moffat with beta=1
#def lorentz2d(x, y, pars, deriv=False, nderiv=None):
#    """Two dimensional Lorentz model function"""
#    # 1/(1+(r**2/alpha**2)**beta)
#    alpha = pars[0]
#    beta = pars[1]
#
#    amp = pars[0]
#    xdiff = x - pars[1]
#    ydiff = y - pars[2]
#    a = pars[3]
#    b = pars[4]
#    c = pars[5]
#    beta = pars[6]
#
#    rr_gg = (a * xdiff ** 2) + (b * xdiff * ydiff) + (c * ydiff ** 2)
#    g = amp * (1 + rr_gg) ** (-beta)

    
def penny2d(x, y, pars, deriv=False, nderiv=None):
    """ Gaussian core and Lorentzian wings, only Gaussian is tilted."""
    # Maybe Lorentzian wings need to be azimuthally symmetric.


    xdiff = x - pars[1]
    ydiff = y - pars[2]
    amp = pars[0]
    a = pars[3]
    b = pars[4]
    c = pars[5]
    # Gaussian component
    g = amp * np.exp(-0.5*((a * xdiff ** 2) + (b * ydiff ** 2) +
                           (c * xdiff * ydiff)))
    # Add Lorentzian wings
    relamp = pars[6]
    sigma = pars[7]
    rr_gg = (xdiff ** 2 + ydiff ** 2) / sigma ** 2
    l = amp * relamp / (1 + rr_gg)
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
            df_dx_mean = ( g * 0.5*((2 * a * xdiff) + (c * ydiff)) +
                           2*l*xdiff/(sigma**2 * (1+rr_gg)) )
            derivative.append(df_dx_mean)
        if nderiv>=3:
            df_dy_mean = ( g * 0.5*((2 * b * ydiff) + (c * xdiff)) +
                           2*l*ydiff/(sigma**2 * (1+rr_gg)) )
            derivative.append(df_dy_mean)
        if nderiv>=4:       
            df_da = g * (-0.5) * xdiff ** 2
            derivative.append(df_da)
        if nderiv>=5:       
            df_db = g * (-0.5) * ydiff ** 2
            derivative.append(df_db)
        if nderiv>=6:       
            df_dc = g * (-0.5) * xdiff * ydiff
            derivative.append(df_dc)
        if nderiv>=7:       
            df_drelamp = l / relamp
            derivative.append(df_drelamp)
        if nderiv>=8:
            df_dsigma = l/(1+rr_gg) * 2*rr_gg/sigma
            derivative.append(df_dsigma)
            
        return f,derivative
            
    # No derivative
    else:        
        return f

    

def empirical(x, y, pars, deriv=False, nderiv=None):
    """Empirical look-up table"""
    nx,ny,npars = pars.shape

    pass


    
# PSF classes


# PSF base class
class PSFBase:

    def __init__(self,mpars,npix=101,binned=False):
        # npix must be odd
        if npix%2==1: npix += 1
        self._params = np.atleast_1d(mpars)
        self.binned = binned
        self.npix = npix

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self,value):
        self._params = value
        
    def __call__(self,x=None,y=None,pars=None,mpars=None,xy=None,deriv=False,**kwargs):
        # X/Y and XY are absolute pixel values NOT relative ones
        # PARS are the stellar parameters [height, xcen, ycen]
        # MPARS are the model parameters
        
        # Nothing input, PSF postage stamp
        if x is None and y is None and pars is None and xy is None:
            pars = [1.0, self.npix//2, self.npix//2]
            pix = np.arange(self.npix)
            x = np.repeat(pix,self.npix).reshape(self.npix,self.npix)
            y = np.repeat(pix,self.npix).reshape(self.npix,self.npix).T

        # Get coordinates from XY
        if x is None and y is None and xy is not None:
            x0,x1 = xy[0]
            y0,y1 = xy[1]
            dx = np.arange(x0,x1+1).astype(float)
            nxpix = len(dx)
            dy = np.arange(y0,y1+1).astype(float)
            nypix = len(dy)
            x = np.repeat(dx,nypix).reshape(nxpix,nypix)
            y = np.repeat(dy,nxpix).reshape(nypix,nxpix).T 

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

        # Add sky to model
        if sky is not None:
            # With derivative
            if deriv is True:
                out[0] = out[0]+sky
            else:
                out += sky

        return out


    def model(xdata,*args,**kwargs):
        """ Function to use with curve_fit() to fit a single stellar profile."""
        ## curve_fit separates each parameter while
        ## psf expects on pars array
        pars = args2
        print(pars)
        return self(xdata[0],xdata[1],pars,**kwargs2)
    
    def modelall(xdata,*args,**kwargs):
        """ Function to use with curve_fit() to fit all parameters of a single stellar profile."""
        allpars = args2
        print(allpars)
        nmpars = len(func.params)
        mpars = allpars[-nmpars:]
        pars = allpars[0:-nmpars]
        return self(xdata[0],xdata[1],pars,mpars=mpars,**kwargs2)

    def fit(im,pars):
        """ Convenience function to fit a single star model."""
        cat = {'X':pars[1],'Y':pars[2]}
        return getpsf.fitstar(im,cat,self)
    
    #def allpars(x,y,pars,**kwargs):
    #    """" can input STELLAR + MODEL parameters in one array."""
    #    nmpars = len(self.params)
    #    pars = pars[0:-nmpars]
    #    mpars = pars[-nmpars:]
    #    return self(x,y,pars,mpars=mpars,**kwargs)
    
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
        if len(mpars)<3:
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
    
    def deriv(self,x, y, pars, nderiv=None):
        """Two dimensional Gaussian model derivative with respect to parameters"""
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
        # [a, b, c, beta]
        if mpars is None:
            mpars = np.array([1.0,1.0,0.0,2.5])
        if len(mpars)<4:
            raise ValueError('2 parameters required')
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
        # pars = [amplitude, x0, y0, sigma, beta]
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

# PSF Lorentz class
class PSFLorentz(PSFBase):
       
    # Initalize the object
    def __init__(self,mpars=None,npix=101,binned=False):
        if mpars is None:
            mpars = np.array([1.0,2.5])
        if len(mpars)<2:
            raise ValueError('2 parameters required')
        # mpars = [sigma, beta]
        if mpars[0]<=0:
            raise ValueError('sigma must be >0')
        super().__init__(mpars,npix=npix,binned=binned)

    def fwhm(self):
        """ Return the FWHM of the model."""
        pass
        
    def evaluate(self,x, y, pars, binned=None, deriv=False, nderiv=None):
        """Two dimensional Lorentz model function"""
        # pars = [amplitude, x0, y0, sigma, beta]
        if binned is None: binned = self.binned
        if binned is True:
            return lorentz2d_integrate(x, y, pars, binned=binned, deriv=deriv, nderiv=nderiv)
        else:
            return lorentz2d(x, y, pars, binned=binned, deriv=deriv, nderiv=nderiv)        

        
    def deriv(self,x, y, pars, binned=None, nderiv=None):
        """Two dimensional Lorentz model derivative with respect to parameters"""
        if binned is None: binned = self.binned
        if binned is True:
            g, derivative = lorentz2d_integrate(x, y, pars, binned=binned, deriv=True, nderiv=nderiv)
        else:
            g, derivative = lorentz2d(x, y, pars, binned=binned, deriv=True, nderiv=nderiv)        
        return derivative
    
# PSF Penny class
class PSFPenny(PSFBase):
    """ Gaussian core and Lorentzian wings, only Gaussian is tilted."""
    # PARS are model parameters
    
    # Initalize the object
    def __init__(self,mpars=None,npix=101,binned=False):
        if mpars is None:
            mpars = np.array([1.0,2.5])
        if len(mpars)<2:
            raise ValueError('2 parameters required')
        # mpars = [sigma, beta]
        if mpars[0]<=0:
            raise ValueError('sigma must be >0')
        super().__init__(mpars,npix=npix,binned=binned)

    def fwhm(self):
        """ Return the FWHM of the model."""
        pass
        
    def evaluate(self,x, y, pars=None, binned=None, deriv=False, nderiv=None):
        """Two dimensional Penny model function"""
        # pars = [amplitude, x0, y0, sigma, beta]
        if pars is None: pars = self.params
        if binned is None: binned = self.binned
        return penny2d(x, y, pars, binned=binned, deriv=deriv, nderiv=nderiv)

    def deriv(self,x, y, pars=None, binned=None, nderiv=None):
        """Two dimensional Penny model derivative with respect to parameters"""
        if pars is None: pars = self.params
        if binned is None: binned = self.binned        
        return penny2d_deriv(x, y, pars, binned=binned, nderiv=nderiv)


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


    

import os
import numpy as np
from numba import njit,types,from_dtype
from numba.experimental import jitclass
from . import models_numba as mnb

# Fit a PSF model to multiple stars in an image

@njit
def median2d(data,axis=-1,ignore_nan=False):
    """Get the median."""
    if axis==-1:
        raise Exception('Use median() if not using axis parameter')
    if data.ndim != 2:
        raise Exception('data must be 2D')
    ndim = data.ndim
    shape = data.shape
    loopdims = np.delete(np.arange(ndim),axis)
    loopshape = np.delete(np.array(shape),axis)
    med = np.zeros(loopshape[0],float)
    for i in range(loopshape[0]):
        if axis==0:
            if ignore_nan:
                med[i] = nanmedian(data[:,i])
            else:
                med[i] = np.median(data[:,i])
        else:
            if ignore_nan:
                med[i] = nanmedian(data[i,:])
            else:
                med[i] = np.median(data[i,:])
    return med

@njit
def median3d(data,axis=-1,ignore_nan=False):
    """Get the median."""
    if axis==-1:
        raise Exception('Use median() if not using axis parameter')
    if data.ndim != 3:
        raise Exception('data must be 3D')
    ndim = data.ndim
    shape = data.shape
    loopdims = np.delete(np.arange(ndim),axis)
    loopshape = np.delete(np.array(shape),axis)
    med = np.zeros((loopshape[0],loopshape[1]),float)
    for i in range(loopshape[0]):
        for j in range(loopshape[1]):
            if axis==0:
                if ignore_nan:
                    med[i,j] = nanmedian(data[:,i,j])
                else:
                    med[i,j] = np.median(data[:,i,j])
            elif axis==1:
                if ignore_nan:
                    med[i,j] = nanmedian(data[i,:,j])
                else:
                    med[i,j] = np.median(data[i,:,j])
            else:
                if ignore_nan:
                    med[i,j] = nanmedian(data[i,j,:])
                else:
                    med[i,j] = np.median(data[i,j,:])
    return med

@njit
def nanmedian(data):
    """ Get the median ignoring nans """
    data1d = data.ravel()
    gd, = np.where(np.isfinite(data1d)==True)
    if len(gd)==0:
        med = np.nan
    else:
        med = np.median(data1d[gd])
    return med

@njit
def mad(data, ignore_nan=True, zero=False):
    """ Calculate the median absolute deviation of an array."""
    data1d = data.ravel()
    # With median reference point
    if zero==False:
        if ignore_nan:
            ref = np.nanmedian(data)
            result= np.median(np.abs(data-ref))
        else:
            ref = np.median(data)
            result = nanmedian(np.abs(data-ref))
    # Using zero as reference point
    else:
        if ignore_nan:
            result= np.median(np.abs(data))
        else:
            ref = np.median(data)
            result = nanmedian(np.abs(data))
    return result * 1.482602218505602

@njit
def mad2d(data, axis=-1, func=None, ignore_nan=True, zero=False):
    """ Calculate the median absolute deviation of an array."""
    if axis==-1:
        raise Exception('Use mad() if not using axis parameter')
    if data.ndim != 2:
        raise Exception('data must be 2D')
    # With median reference point
    if zero==False:
        if ignore_nan:
            ref = median2d(data,axis=axis,ignore_nan=True)
            newshape = np.array(data.shape)
            newshape[axis] = -1
            newshape = (newshape[0],newshape[1])
            resid = data-ref.reshape(newshape)
            result = median2d(np.abs(resid),axis=axis,ignore_nan=True)
        else:
            ref = median2d(data,axis=axis)
            newshape = np.array(data.shape)
            newshape[axis] = -1
            newshape = (newshape[0],newshape[1])
            resid = data-ref.reshape(newshape)
            result = median2d(np.abs(resid),axis=axis)
    # Using zero as reference point
    else:
        if ignore_nan:
            result = median2d(np.abs(data),axis=axis,ignore_nan=True)
        else:
            result = median2d(np.abs(data),axis=axis)

    return result * 1.482602218505602

@njit
def mad3d(data, axis=-1, func=None, ignore_nan=True, zero=False):
    """ Calculate the median absolute deviation of an array."""
    if axis==-1:
        raise Exception('Use mad() if not using axis parameter')
    if data.ndim != 3:
        raise Exception('data must be 3D')
    # With median reference point
    if zero==False:
        if ignore_nan:
            ref = median3d(data,axis=axis,ignore_nan=True)
            newshape = np.array(data.shape)
            newshape[axis] = -1
            newshape = (newshape[0],newshape[1],newshape[2])
            resid = data-ref.reshape(newshape)
            result = median3d(np.abs(resid),axis=axis,ignore_nan=True)
        else:
            ref = median3d(data,axis=axis)
            newshape = np.array(data.shape)
            newshape[axis] = -1
            newshape = (newshape[0],newshape[1],newshape[2])
            resid = data-ref.reshape(newshape)
            result = median3d(np.abs(resid),axis=axis)
    # Using zero as reference point
    else:
        if ignore_nan:
            result = median3d(np.abs(data),axis=axis,ignore_nan=True)
        else:
            result = median3d(np.abs(data),axis=axis)
    return result * 1.482602218505602

@njit
def quadratic_bisector(x,y):
    """ Calculate the axis of symmetric or bisector of parabola"""
    #https://www.azdhs.gov/documents/preparedness/state-laboratory/lab-licensure-certification/technical-resources/
    #    calibration-training/12-quadratic-least-squares-regression-calib.pdf
    #quadratic regression statistical equation
    n = len(x)
    if n<3:
        return None
    Sxx = np.sum(x**2) - np.sum(x)**2/n
    Sxy = np.sum(x*y) - np.sum(x)*np.sum(y)/n
    Sxx2 = np.sum(x**3) - np.sum(x)*np.sum(x**2)/n
    Sx2y = np.sum(x**2 * y) - np.sum(x**2)*np.sum(y)/n
    Sx2x2 = np.sum(x**4) - np.sum(x**2)**2/n
    #a = ( S(x^2*y)*S(xx)-S(xy)*S(xx^2) ) / ( S(xx)*S(x^2x^2) - S(xx^2)^2 )
    #b = ( S(xy)*S(x^2x^2) - S(x^2y)*S(xx^2) ) / ( S(xx)*S(x^2x^2) - S(xx^2)^2 )
    denom = Sxx*Sx2x2 - Sxx2**2
    if denom==0:
        return np.nan
    a = ( Sx2y*Sxx - Sxy*Sxx2 ) / denom
    b = ( Sxy*Sx2x2 - Sx2y*Sxx2 ) / denom
    if a==0:
        return np.nan
    return -b/(2*a)

@njit
def meshgrid(x,y):
    """ Implementation of numpy's meshgrid function."""
    nx = len(x)
    ny = len(y)
    dtype = np.array(x[0]*y[0]).dtype
    xx = np.zeros((ny,nx),dtype)
    for i in range(ny):
        xx[i,:] = x
    yy = np.zeros((ny,nx),dtype)
    for i in range(nx):
        yy[:,i] = y
    return xx,yy

@njit
def aclip(val,minval,maxval):
    newvals = np.zeros(len(val),float)
    for i in range(len(val)):
        if val[i] < minval:
            nval = minval
        elif val[i] > maxval:
            nval = maxval
        else:
            nval = val[i]
        newvals[i] = nval
    return newvals

@njit
def clip(val,minval,maxval):
    if val < minval:
        nval = minval
    elif val > maxval:
        nval = maxval
    else:
        nval = val
    return nval

@njit
def drop_imag(z):
    EPSILON = 1e-07    
    if abs(z.imag) <= EPSILON:
        z = z.real
    return z

@njit
def gamma(z):
    # Gamma function for a single z value
    # Using the Lanczos approximation
    # https://en.wikipedia.org/wiki/Lanczos_approximation
    g = 7
    n = 9
    p = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    ]
    if z < 0.5:
        y = PI / (np.sin(PI * z) * gamma(1 - z))  # Reflection formula
    else:
        z -= 1
        x = p[0]
        for i in range(1, len(p)):
            x += p[i] / (z + i)
        t = z + g + 0.5
        y = np.sqrt(2 * PI) * t ** (z + 0.5) * np.exp(-t) * x
    return y

@njit
def gammaincinv05(a):
    """ gammaincinv(a,0.5) """
    n = np.array([1.00e-03, 1.12e-01, 2.23e-01, 3.34e-01, 4.45e-01, 5.56e-01,
                  6.67e-01, 7.78e-01, 8.89e-01, 1.00e+00, 2.00e+00, 3.00e+00,
                  4.00e+00, 5.00e+00, 6.00e+00, 7.00e+00, 8.00e+00, 9.00e+00,
                  1.00e+01, 1.10e+01, 1.20e+01])

    y = np.array([5.24420641e-302, 1.25897478e-003, 3.03558724e-002, 9.59815712e-002,
                  1.81209305e-001, 2.76343765e-001, 3.76863377e-001, 4.80541354e-001,
                  5.86193928e-001, 6.93147181e-001, 1.67834699e+000, 2.67406031e+000,
                  3.67206075e+000, 4.67090888e+000, 5.67016119e+000, 6.66963707e+000,
                  7.66924944e+000, 8.66895118e+000, 9.66871461e+000, 1.06685224e+001,
                  1.16683632e+001])
    # Lower edge
    if a < n[0]:
        out = 0.0
    # Upper edge
    elif a > n[-1]:
        # linear approximation to large values
        # coef = np.array([ 0.99984075, -0.32972584])
        out = 0.99984075*a-0.32972584
    # Interpolate values
    else:
        ind = np.searchsorted(n,a)
        # exact match
        if n[ind]==a:
            out = y[ind]
        # At beginning
        elif ind==0:
            slp = (y[1]-y[0])/(n[1]-n[0])
            out = slp*(a-n[0])+y[0]
        else:
            slp = (y[ind]-y[ind-1])/(n[ind]-n[ind-1])
            out = slp*(a-n[ind-1])+y[ind-1]
            
    return out

    
@njit
def linearinterp(data,x,y):
    """
    Linear interpolation.

    Parameters
    ----------
    data : numpy array
      The data to use for interpolation.
    x : float
      The X-value at which to perform the interpolation.
    y : float
      The Y-value at which to perform the interpolation.

    Returns
    -------
    f : float
       The interpolated value.
    
    Examples
    --------

    f = linearinterp(data,x,y)

    """

    
    ny,nx = data.shape

    # Out of bounds
    if x<0 or x>(nx-1) or y<0 or y>(ny-1):
        return 0.0

    x1 = int(x)
    if x1==nx-1:
        x1 -= 1
    x2 = x1+1
    y1 = int(y)
    if y1==ny-1:
        y1 -= 1
    y2 = y1+1

    f11 = data[y1,x1]
    f12 = data[y2,x1]
    f21 = data[y1,x2]
    f22 = data[y2,x2]
    
    # weighted mean
    # denom = (x2-x1)*(y2-y1) = 1
    w11 = (x2-x)*(y2-y)
    w12 = (x2-x)*(y-y1)
    w21 = (x-x1)*(y2-y)
    w22 = (x-x1)*(y-y1)
    f = w11*f11+w12*f12+w21*f21+w22*f22

    return f

@njit
def alinearinterp(data,x,y):
    """
    Linear interpolation.

    Parameters
    ----------
    data : numpy array
      The data to use for interpolation.
    x : numpy array
      Array of X-value at which to perform the interpolation.
    y : numpy array
      Array of Y-value at which to perform the interpolation.

    Returns
    -------
    f : numpy array
       The interpolated values.
    
    Examples
    --------

    f = alinearinterp(data,x,y)

    """

    if x.ndim==2:
        x1d = x.ravel()
        y1d = y.ravel()
    else:
        x1d = x
        y1d = y
    npix = len(x1d)
    f = np.zeros(npix,float)
    for i in range(npix):
        f[i] = linearinterp(data,x1d[i],y1d[i])
    return f


@njit
def inverse(a):
    """ Safely take the inverse of a square 2D matrix."""
    # This checks for zeros on the diagonal and "fixes" them.
    
    # If one of the dimensions is zero in the R matrix [Npars,Npars]
    # then replace it with a "dummy" value.  A large value in R
    # will give a small value in inverse of R.
    #badpar, = np.where(np.abs(np.diag(a))<sys.float_info.min)
    badpar = (np.abs(np.diag(a))<2e-300)
    if np.sum(badpar)>0:
        a[badpar] = 1e10
    ainv = np.linalg.inv(a)
    # What if the inverse fails???
    # can we catch it in numba
    # Fix values
    a[badpar] = 0  # put values back
    ainv[badpar] = 0
    
    return ainv

@njit
def qr_jac_solve(jac,resid,weight=None):
    """ Solve part of a non-linear least squares equation using QR decomposition
        using the Jacobian."""
    # jac: Jacobian matrix, first derivatives, [Npix, Npars]
    # resid: residuals [Npix]
    # weight: weights, ~1/error**2 [Npix]
    
    # QR decomposition
    if weight is None:
        q,r = np.linalg.qr(jac)
        rinv = inverse(r)
        dbeta = rinv @ (q.T @ resid)
    # Weights input, multiply resid and jac by weights        
    else:
        q,r = np.linalg.qr( jac * weight.reshape(-1,1) )
        rinv = inverse(r)
        dbeta = rinv @ (q.T @ (resid*weight))
        
    return dbeta


@njit
def jac_covariance(jac,resid,wt):
    """ Determine the covariance matrix. """
    
    npix,npars = jac.shape
    
    # Weights
    #   If weighted least-squares then
    #   J.T * W * J
    #   where W = I/sig_i**2
    if wt is not None:
        wt2 = wt.reshape(-1,1) + np.zeros(npars)
        hess = jac.T @ (wt2 * jac)
    else:
        hess = jac.T @ jac  # not weighted

    # cov = H-1, covariance matrix is inverse of Hessian matrix
    cov_orig = inverse(hess)

    # Rescale to get an unbiased estimate
    # cov_scaled = cov * (RSS/(m-n)), where m=number of measurements, n=number of parameters
    # RSS = residual sum of squares
    #  using rss gives values consistent with what curve_fit returns
    # Use chi-squared, since we are doing the weighted least-squares and weighted Hessian
    if wt is not None:
        chisq = np.sum(resid**2 * wt)
    else:
        chisq = np.sum(resid**2)
    cov = cov_orig * (chisq/(npix-npars))  # what MPFITFUN suggests, but very small
        
    return cov

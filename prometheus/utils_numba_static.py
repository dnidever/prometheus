# import os
import numpy as np
# import numba
from numba import njit,types,from_dtype,typed
# from numba.typed import Dict,List
from numba.experimental import jitclass
#from numba_kdtree import KDTree

# Fit a PSF model to multiple stars in an image

PI = 3.141592653589793

from numba.pycc import CC
cc = CC('_utils_numba_static')

@cc.export('sum', 'f8(f8[:],b1)')
@cc.export('sumi', 'i8(i8[:],b1)')
def sum(data,ignore_nan=False):
    """Get the sum."""
    if ignore_nan:
        sm = nansum(data)
    else:
        sm = np.sum(data)
    return sm

@njit
@cc.export('nansum', 'f8(f8[:])')
#@cc.export('nansumi', 'i8(i8[:])')
def nansum(data):
    """ Get the sum ignoring nans """
    data = data
    gd, = np.where(np.isfinite(data)==True)
    if len(gd)==0:
        sm = np.nan
    else:
        sm = np.sum(data[gd])
    return sm

@njit
@cc.export('sum2d', 'f8[:](f8[:,:],i4,b1)')
@cc.export('sum2di', 'f8[:](i8[:,:],i4,b1)')
def sum2d(data,axis=-1,ignore_nan=False):
    """Get the sum."""
    if axis==-1:
        raise Exception('Use sum() if not using axis parameter')
    if data.ndim != 2:
        raise Exception('data must be 2D')
    ndim = data.ndim
    shape = data.shape
    loopdims = np.delete(np.arange(ndim),axis)
    loopshape = np.delete(np.array(shape),axis)
    sm = np.zeros(loopshape[0],float)
    for i in range(loopshape[0]):
        if axis==0:
            if ignore_nan:
                sm[i] = nansum(data[:,i])
            else:
                sm[i] = np.sum(data[:,i])
        else:
            if ignore_nan:
                sm[i] = nansum(data[i,:])
            else:
                sm[i] = np.sum(data[i,:])
    return sm

@njit
@cc.export('sum3d', 'f8[:,:](f8[:,:,:],i4,b1)')
@cc.export('sum3di', 'f8[:,:](i8[:,:,:],i4,b1)')
def sum3d(data,axis=-1,ignore_nan=False):
    """Get the sum."""
    if axis==-1:
        raise Exception('Use sum() if not using axis parameter')
    if data.ndim != 3:
        raise Exception('data must be 3D')
    ndim = data.ndim
    shape = data.shape
    loopdims = np.delete(np.arange(ndim),axis)
    loopshape = np.delete(np.array(shape),axis)
    sm = np.zeros((loopshape[0],loopshape[1]),float)
    for i in range(loopshape[0]):
        for j in range(loopshape[1]):
            if axis==0:
                if ignore_nan:
                    sm[i,j] = nansum(data[:,i,j])
                else:
                    sm[i,j] = np.sum(data[:,i,j])
            elif axis==1:
                if ignore_nan:
                    sm[i,j] = nansum(data[i,:,j])
                else:
                    sm[i,j] = np.sum(data[i,:,j])
            else:
                if ignore_nan:
                    sm[i,j] = nansum(data[i,j,:])
                else:
                    sm[i,j] = np.sum(data[i,j,:])
    return sm

@njit
@cc.export('nanmean', 'f8(f8[:])')
@cc.export('nanmeani', 'i8(i8[:])')
def nanmean(data):
    """ Get the mean ignoring nans """
    gd, = np.where(np.isfinite(data)==True)
    if len(gd)==0:
        mn = np.nan
    else:
        mn = np.mean(data[gd])
    return mn

@njit
@cc.export('mean', 'f8(f8[:],b1)')
@cc.export('meani', 'i8(i8[:],b1)')
def mean(data,ignore_nan=False):
    """Get the mean."""
    if ignore_nan:
        mn = nanmean(data)
    else:
        mn = np.mean(data)
    return mn

@njit
@cc.export('mean2d', 'f8[:](f8[:,:],i4,b1)')
@cc.export('mean2di', 'f8[:](i8[:,:],i4,b1)')
def mean2d(data,axis=-1,ignore_nan=False):
    """Get the mean."""
    if axis==-1:
        raise Exception('Use mean() if not using axis parameter')
    if data.ndim != 2:
        raise Exception('data must be 2D')
    ndim = data.ndim
    shape = data.shape
    loopdims = np.delete(np.arange(ndim),axis)
    loopshape = np.delete(np.array(shape),axis)
    mn = np.zeros(loopshape[0],float)
    for i in range(loopshape[0]):
        if axis==0:
            if ignore_nan:
                mn[i] = nanmean(data[:,i])
            else:
                mn[i] = np.mean(data[:,i])
        else:
            if ignore_nan:
                mn[i] = nanmean(data[i,:])
            else:
                mn[i] = np.mean(data[i,:])
    return mn

@njit
@cc.export('mean3d', 'f8[:,:](f8[:,:,:],i4,b1)')
@cc.export('mean3di', 'f8[:,:](i8[:,:,:],i4,b1)')
def mean3d(data,axis=-1,ignore_nan=False):
    """Get the mean."""
    if axis==-1:
        raise Exception('Use mean() if not using axis parameter')
    if data.ndim != 3:
        raise Exception('data must be 3D')
    ndim = data.ndim
    shape = data.shape
    loopdims = np.delete(np.arange(ndim),axis)
    loopshape = np.delete(np.array(shape),axis)
    mn = np.zeros((loopshape[0],loopshape[1]),float)
    for i in range(loopshape[0]):
        for j in range(loopshape[1]):
            if axis==0:
                if ignore_nan:
                    mn[i,j] = nanmean(data[:,i,j])
                else:
                    mn[i,j] = np.mean(data[:,i,j])
            elif axis==1:
                if ignore_nan:
                    mn[i,j] = nanmean(data[i,:,j])
                else:
                    mn[i,j] = np.mean(data[i,:,j])
            else:
                if ignore_nan:
                    mn[i,j] = nanmean(data[i,j,:])
                else:
                    mn[i,j] = np.mean(data[i,j,:])
    return mn

@njit
@cc.export('nanmedian', 'f8(f8[:])')
@cc.export('nanmediani', 'i8(i8[:])')
def nanmedian(data):
    """ Get the median ignoring nans """
    gd, = np.where(np.isfinite(data)==True)
    if len(gd)==0:
        med = np.nan
    else:
        med = np.median(data[gd])
    return med

@njit
@cc.export('median', 'f8(f8[:],b1)')
@cc.export('mediani', 'i8(i8[:],b1)')
def median(data,ignore_nan=False):
    """Get the median."""
    if ignore_nan:
        med = nanmedian(data)
    else:
        med = np.median(data)
    return med

@njit
@cc.export('median2d', 'f8[:](f8[:,:],i4,b1)')
@cc.export('median2di', 'f8[:](i8[:,:],i4,b1)')
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
@cc.export('median3d', 'f8[:,:](f8[:,:,:],i4,b1)')
@cc.export('median3di', 'f8[:,:](i8[:,:,:],i4,b1)')
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
@cc.export('mad', 'f8(f8[:],b1,b1)')
@cc.export('madi', 'i8(i8[:],b1,b1)')
def mad(data, ignore_nan=True, zero=False):
    """ Calculate the median absolute deviation of an array."""
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
@cc.export('mad2d', 'f8[:](f8[:,:],i4,b1,b1)')
@cc.export('mad2di', 'f8[:](i8[:,:],i4,b1,b1)')
def mad2d(data, axis=-1, ignore_nan=True, zero=False):
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
@cc.export('mad3d', 'f8[:,:](f8[:,:,:],i4,b1,b1)')
@cc.export('mad3di', 'f8[:,:](i8[:,:,:],i4,b1,b1)')
def mad3d(data, axis=-1, ignore_nan=True, zero=False):
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
@cc.export('quadratic_bisector', 'f8(f8[:],f8[:])')
@cc.export('quadratic_bisectori', 'f8(i8[:],i8[:])')
def quadratic_bisector(x,y):
    """ Calculate the axis of symmetric or bisector of parabola"""
    #https://www.azdhs.gov/documents/preparedness/state-laboratory/lab-licensure-certification/technical-resources/
    #    calibration-training/12-quadratic-least-squares-regression-calib.pdf
    #quadratic regression statistical equation
    n = len(x)
    if n<3:
        return np.nan
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
@cc.export('meshgrid', '(f8[:],f8[:])')
@cc.export('meshgridi', '(i8[:],i8[:])')
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


# @njit
# @cc.export('meshgridi', '(i8[:],i8[:])')
# def meshgridi(x,y):
#     """ Implementation of numpy's meshgrid function."""
#     nx = len(x)
#     ny = len(y)
#     dtype = np.array(x[0]*y[0]).dtype
#     xx = np.zeros((ny,nx),dtype)
#     for i in range(ny):
#         xx[i,:] = x
#     yy = np.zeros((ny,nx),dtype)
#     for i in range(nx):
#         yy[:,i] = y
#     return xx,yy

@njit
@cc.export('aclip', 'f8[:](f8[:],f8,f8)')
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
@cc.export('clip', 'f8(f8,f8,f8)')
@cc.export('clipi', 'i8(i8,i8,i8)')
def clip(val,minval,maxval):
    if val < minval:
        nval = minval
    elif val > maxval:
        nval = maxval
    else:
        nval = val
    return nval

@njit
@cc.export('drop_imag', 'f8(f8)')
def drop_imag(z):
    EPSILON = 1e-07    
    if np.abs(z.imag) <= EPSILON:
        z = z.real
    return z

@njit('f8(f8)')
@cc.export('gamma', 'f8(f8)')
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
@cc.export('gammaincinv05', 'f8(f8)')
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
@cc.export('linearinterp', 'f8(f8[:,:],f8,f8)')
@cc.export('linearinterpi', 'f8(i8[:,:],i8,i8)')
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
@cc.export('alinearinterp', '(f8[:,:],f8[:],f8[:])')
@cc.export('alinearinterpi', '(f8[:,:],i8[:],i8[:])')
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
    npix = len(x)
    f = np.zeros(npix,float)
    for i in range(npix):
        f[i] = linearinterp(data,x[i],y[i])
    return f

@njit
@cc.export('inverse', 'f8[:,:](f8[:,:])')
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
@cc.export('qr_jac_solve', '(f8[:,:],f8[:],f8[:])')
def qr_jac_solve(jac,resid,weight=None):
    """ Solve part of a non-linear least squares equation using QR decomposition
        using the Jacobian."""
    # jac: Jacobian matrix, first derivatives, [Npix, Npars]
    # resid: residuals [Npix]
    # weight: weights, ~1/error**2 [Npix]

    #if weight is not None and weight.flags['C_CONTIGUOUS']==False:
    #    # Make array contiguous
    if weight is not None:
        weight = np.ascontiguousarray(weight)
    #jac = np.ascontiguousarray(jac)
    #resid = np.ascontiguousarray(resid)
        
    # QR decomposition
    if weight is None:
        q,r = np.linalg.qr(jac)
        rinv = inverse(r)
        dbeta = rinv @ (q.T @ resid)
    # Weights input, multiply resid and jac by weights        
    else:
        weight = np.ascontiguousarray(weight)
        q,r = np.linalg.qr( jac * weight.reshape(-1,1) )
        rinv = inverse(r)
        dbeta = rinv @ (q.T @ (resid*weight))
        
    return dbeta

@njit
@cc.export('jac_covariance', '(f8[:,:],f8[:],f8[:])')
def jac_covariance(jac,resid,wt):
    """ Determine the covariance matrix. """
    
    npix,npars = jac.shape

    # Make array contiguous
    wt = np.ascontiguousarray(wt)
    
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
    dof = npix-npars
    if dof<=0:
        dof = 1
    cov = cov_orig * (chisq/dof)  # what MPFITFUN suggests, but very small
        
    return cov

# @njit
# @cc.export('checkbounds', '(f8[:],f8[:,:])')
# def checkbounds(pars,bounds):
#     """ Check the parameters against the bounds."""
#     # 0 means it's fine
#     # 1 means it's beyond the lower bound
#     # 2 means it's beyond the upper bound
#     npars = len(pars)
#     lbounds,ubounds = bounds[:,0],bounds[:,1]
#     check = np.zeros(npars,np.int32)
#     badlow, = np.where(pars<=lbounds)
#     if len(badlow)>0:
#         check[badlow] = 1
#     badhigh, = np.where(pars>=ubounds)
#     if len(badhigh)>0:
#         check[badhigh] = 2
#     return check

@njit
@cc.export('checkbounds', '(f8[:],UniTuple(f8[:],2))')
def checkbounds(pars,bounds):
    """ Check the parameters against the bounds."""
    # 0 means it's fine
    # 1 means it's beyond the lower bound
    # 2 means it's beyond the upper bound
    npars = len(pars)
    lbounds,ubounds = bounds
    check = np.zeros(npars,np.int32)
    badlo, = np.where(pars<=lbounds)
    if len(badlo)>0:
        check[badlo] = 1
    badhi, = np.where(pars>=ubounds)
    if len(badhi):
        check[badhi] = 2
    return check

@njit
@cc.export('limbounds', '(f8[:],UniTuple(f8[:],2))')
def limbounds(pars,bounds):
    """ Limit the parameters to the boundaries."""
    lbounds,ubounds = bounds
    outpars = np.minimum(np.maximum(pars,lbounds),ubounds)
    return outpars

@njit
@cc.export('limsteps', '(f8[:],f8[:])')
def limsteps(steps,maxsteps):
    """ Limit the parameter steps to maximum step sizes."""
    signs = np.sign(steps)
    outsteps = np.minimum(np.abs(steps),maxsteps)
    outsteps *= signs
    return outsteps

# @njit
# @cc.export('newpars', '(f8[:],f8[:],f8[:,:],f8[:])')
# def newpars(pars,steps,bounds=None,maxsteps=None):
#     """ Return new parameters that fit the constraints."""
#     # Limit the steps to maxsteps
#     if maxsteps is not None:
#         limited_steps = limsteps(steps,maxsteps)
#     else:
#         limited_steps = steps

#     # No bounds input
#     if bounds is None:
#         return pars+limited_steps
        
#     # Make sure that these don't cross the boundaries
#     lbounds,ubounds = bounds[:,0],bounds[:,1]
#     check = checkbounds(pars+limited_steps,bounds)
#     # Reduce step size for any parameters to go beyond the boundaries
#     badpars = (check!=0)
#     # reduce the step sizes until they are within bounds
#     newsteps = limited_steps.copy()
#     count = 0
#     maxiter = 2
#     while (np.sum(badpars)>0 and count<=maxiter):
#         newsteps[badpars] /= 2
#         newcheck = checkbounds(pars+newsteps,bounds)
#         badpars = (newcheck!=0)
#         count += 1
            
#     # Final parameters
#     newpars = pars + newsteps
                
#     # Make sure to limit them to the boundaries
#     check = checkbounds(newpars,bounds)
#     badpars = (check!=0)
#     if np.sum(badpars)>0:
#         # add a tiny offset so it doesn't fit right on the boundary
#         newpars = np.minimum(np.maximum(newpars,lbounds+1e-30),ubounds-1e-30)
#     return newpars

@njit
@cc.export('poly2d', '(f8[:,:],f8[:])')
def poly2d(xdata,pars):
    """ model of 2D linear polynomial."""
    x = xdata[:,0]
    y = xdata[:,1]
    return pars[0]+pars[1]*x+pars[2]*y+pars[3]*x*y

@njit
@cc.export('jacpoly2d', '(f8[:,:],f8[:])')
def jacpoly2d(xdata,pars):
    """ jacobian of 2D linear polynomial."""
    x = xdata[:,0]
    y = xdata[:,1]
    nx = len(x)
    # Model
    m = pars[0]+pars[1]*x+pars[2]*y+pars[3]*x*y
    # Jacobian, partical derivatives wrt the parameters
    jac = np.zeros((nx,4),float)
    jac[:,0] = 1    # constant coefficient
    jac[:,1] = x    # x-coefficient
    jac[:,2] = y    # y-coefficient
    jac[:,3] = x*y  # xy-coefficient
    return m,jac

@njit
@cc.export('poly2dfit', '(f8[:],f8[:],f8[:],f8[:],i4,f8,b1)')
def poly2dfit(x,y,data,error,maxiter=2,minpercdiff=0.5,verbose=False):
    """ Fit a 2D linear function to data robustly."""
    ndata = len(data)
    if ndata<4:
        raise Exception('Need at least 4 data points for poly2dfit')
    gd1, = np.where(np.isfinite(data))
    if len(gd1)<4:
        raise Exception('Need at least 4 good data points for poly2dfit')
    xdata = np.zeros((len(gd1),2),float)
    xdata[:,0] = x[gd1]
    xdata[:,1] = y[gd1]
    initpar = np.zeros(4,float)
    med = np.median(data[gd1])
    sig = mad(data[gd1])
    gd2, = np.where( (np.abs(data-med)<3*sig) & np.isfinite(data))
    if len(gd1)>=4 and len(gd2)<4:
        gd = gd1
    else:
        gd = gd2
    initpar[0] = med
    xdata = np.zeros((len(gd),2),float)
    xdata[:,0] = x[gd]
    xdata[:,1] = y[gd]
    data1 = data[gd]
    error1 = error[gd]

    # Do the fit
    # Iterate
    count = 0
    bestpar = initpar.copy()
    maxpercdiff = 1e10
    # maxsteps = None
    #wt = 1.0/error1.ravel()**2
    wt = np.ones(len(error1),float)  # unweighted
    while (count<maxiter and maxpercdiff>minpercdiff):
        # Use Cholesky, QR or SVD to solve linear system of equations
        m,j = jacpoly2d(xdata,bestpar)
        dy = data1.ravel()-m.ravel()
        # Solve Jacobian
        #if error is not None:
        #dbeta = qr_jac_solve(j,dy,weight=wt)
        dbeta = qr_jac_solve(j,dy,wt)
        #else:
        #    dbeta = qr_jac_solve(j,dy)
        dbeta[~np.isfinite(dbeta)] = 0.0  # deal with NaNs

        # -add "shift cutting" and "line search" in the least squares method
        # basically scale the beta vector to find the best match.
        # check this out
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.line_search.html
        
        # Update parameters
        oldpar = bestpar.copy()
        # limit the steps to the maximum step sizes and boundaries
        #if bounds is not None or maxsteps is not None:
        #    bestpar = newpars(bestpar,dbeta,bounds,maxsteps)
        #else:
        bestpar += dbeta
        # Check differences and changes
        diff = np.abs(bestpar-oldpar)
        denom = np.maximum(np.abs(oldpar.copy()),0.0001)
        percdiff = diff.copy()/denom*100  # percent differences
        maxpercdiff = np.max(percdiff)
                
        if verbose:
            print('N = ',count)
            print('bestpars = ',bestpar)
            print('dbeta = ',dbeta)
                
        count += 1

    # Get covariance and errors
    m,j = jacpoly2d(xdata,bestpar)
    dy = data1.ravel()-m.ravel()
    cov = jac_covariance(j,dy,wt)
    perror = np.sqrt(np.diag(cov))

    return bestpar,perror,cov


spec = [
    ('_inputtype', types.int64),
    ('_data', types.float64[:]),
    ('ndata', types.int32),
    ('_sindex', types.int64[:]),
    ('_sdata', types.float64[:]),
    ('_num', types.int64[:]),
    ('_values', types.float64[:]),
    ('_lo', types.int64[:]),
    ('_hi', types.int64[:]),
    ('_invindex', types.int64[:]),
]
@jitclass(spec)
class Index(object):

    def __init__(self, arr):
        if len(arr)==0:
            raise ValueError('arr has no elements')
        aint32 = np.zeros(10,np.int32)
        aint64 = np.zeros(10,np.int64)
        if arr.dtype is aint32.dtype or arr.dtype is aint64.dtype:
            self._inputtype = 2     # int
        else:
            self._inputtype = 1     # float
        data = np.zeros(len(arr),float)   # make sure they are float
        data[:] = arr
        self._data = data
        self.ndata = len(data)
        self._sindex = np.argsort(self._data)
        self._sdata = data.copy()[self._sindex]
        brklo, = np.where(self._sdata != np.roll(self._sdata,1))
        nbrk = len(brklo)
        if nbrk>0:
            brkhi = np.hstack((brklo[1:nbrk]-1,np.array([self.ndata-1])))
            num = brkhi-brklo+1
            self._num = num
            self._values = self._sdata[brklo]
            self._lo = brklo
            self._hi = brkhi
        else:
            self._num = np.array([1])
            self._values = np.array([data[0]])
            self._lo = np.array([0])
            self._hi = np.array([self.ndata-1])
        self._invindex = np.zeros(len(arr),np.int64)-1
            
    @property
    def data(self):
        return self._data

    @property
    def index(self):
        # unique index
        return self._sindex[self._lo]
    
    @property
    def num(self):
        return self._num

    def __len__(self):
        return len(self._num)
    
    def __getitem__(self,item):
        return self._values[item]
    
    @property
    def values(self):
        return self._values
    
    def get(self,item):
        lo = self._lo[item]
        hi = self._hi[item]
        n = hi-lo+1
        return self._sdata[lo:hi+1]

    def getindex(self,item):
        lo = self._lo[item]
        hi = self._hi[item]
        n = hi-lo+1
        return self._sindex[lo:hi+1]

    @property
    def invindex(self):
        # Construct inverse index
        #  takes you from the original pixels to the unique ones (index.values)
        if self._invindex[0]==-1:
            # Loop over the unique values
            for i in range(len(self)):
                ind = self.getindex(i)   # index array for each unique value
                self._invindex[ind] = i
        return self._invindex

@njit
@cc.export('ravel_multi_index', '(UniTuple(i8[:],2),UniTuple(i8,2))')
@cc.export('ravel_multi_index3', '(UniTuple(i8[:],3),UniTuple(i8,3))')
def ravel_multi_index(multi_index,imshape):
    """ ravel indices"""
    # multi_index: tuple of integer arrays

    n = len(multi_index[0])
    ndim = len(imshape)
    factor = np.zeros(ndim,np.int64)
    factor[-1] = 1
    for i in np.arange(ndim-2,-1,-1):
        factor[i] = factor[i+1]*imshape[i+1]
    
    # Loop over points
    index = np.zeros(n,np.int64)
    for i in range(n):
        for j in range(ndim):
            index[i] += multi_index[j][i]*factor[j]
    
    return index

@njit
@cc.export('unravel_index', '(i8[:],UniTuple(i8,2))')
def unravel_index(indices,imshape):
    """ return multi-dimensional index. """

    n = len(indices)
    ndim = len(imshape)
    factor = np.zeros(ndim,np.int64)
    factor[-1] = 1
    for i in np.arange(ndim-2,-1,-1):
        factor[i] = factor[i+1]*imshape[i+1]
    
    unraveled_coords = np.zeros((n,ndim),np.int64)
    for i in range(n):
        left = indices[i]
        for j in range(ndim):
            unraveled_coords[i,j] = left//factor[j]
            left -= unraveled_coords[i,j]*factor[j]

    return unraveled_coords

# @njit
# @cc.export('unique_indexf', '(f8[:],)')
# @cc.export('unique_indexi', '(i8[:],)')
# def unique_index(array):
#     """ return unique values, index and reverse indez."""
#     index = Index(array)
#     uvals = array[index.index]
#     # Unique index
#     uindex = index.index
#     # Inverse index
#     #   the inverse index list takes you from the original/duplicated pixels
#     #   to the unique ones
#     invindex = index.invindex
#     return uvals,uindex,invindex

@njit
@cc.export('unique_index', '(f8[:],)')
def unique_indexf(array):
    data = np.zeros(len(array),float)   # make sure they are float
    data[:] = array
    ndata = len(data)
    sindex = np.argsort(data)
    sdata = data.copy()[sindex]
    brklo, = np.where(sdata != np.roll(sdata,1))
    nbrk = len(brklo)
    if nbrk>0:
        brkhi = np.hstack((brklo[1:nbrk]-1,np.array([ndata-1])))
        num = brkhi-brklo+1
        values = sdata[brklo]
        lo = brklo
        hi = brkhi
    else:
        num = np.array([1])
        values = np.array([data[0]])
        lo = np.array([0])
        hi = np.array([ndata-1])
    # Construct inverse index
    #  takes you from the original pixels to the unique ones (index.values)
    invindex = np.zeros(len(array),np.int64)-1
    # Loop over the unique values
    for i in range(len(num)):
        #ind = self.getindex(i)   # index array for each unique value
        lo1 = lo[i]
        hi1 = hi[i]
        n = hi1-lo1+1
        ind = sindex[lo1:hi1+1]
        invindex[ind] = i
    # Unique index
    uindex = sindex[lo]
    # Unique values
    uvals = data[uindex]
    return uvals,uindex,invindex


@njit
@cc.export('unique_indexi', '(i8[:],)')
def unique_index(array):
    data = np.zeros(len(array),np.int64)   # make sure they are float
    data[:] = array
    ndata = len(data)
    sindex = np.argsort(data)
    sdata = data.copy()[sindex]
    brklo, = np.where(sdata != np.roll(sdata,1))
    nbrk = len(brklo)
    if nbrk>0:
        brkhi = np.hstack((brklo[1:nbrk]-1,np.array([ndata-1])))
        num = brkhi-brklo+1
        values = sdata[brklo]
        lo = brklo
        hi = brkhi
    else:
        num = np.array([1])
        values = np.array([data[0]])
        lo = np.array([0])
        hi = np.array([ndata-1])
    # Construct inverse index
    #  takes you from the original pixels to the unique ones (index.values)
    invindex = np.zeros(len(array),np.int64)-1
    # Loop over the unique values
    for i in range(len(num)):
        #ind = self.getindex(i)   # index array for each unique value
        lo1 = lo[i]
        hi1 = hi[i]
        n = hi1-lo1+1
        ind = sindex[lo1:hi1+1]
        invindex[ind] = i
    # Unique index
    uindex = sindex[lo]
    # Unique values
    uvals = data[uindex]
    return uvals,uindex,invindex


@njit
@cc.export('smooth2d', '(f8[:,:],i8,i4)')
def smooth2d(im,binsize,med=0):
    """ Do smoothing of a 2D image."""
    # mean by default
    smim = im.copy()
    ny,nx = im.shape
    ny2 = int(np.ceil(ny / binsize))
    nx2 = int(np.ceil(nx / binsize))
    for i in range(nx2):
        for j in range(ny2):
            x1 = i*binsize
            x2 = x1+binsize
            if x2 > nx: x2=nx
            y1 = j*binsize
            y2 = y1+binsize
            if y2 > ny: y2=ny
            if med==1:
                sm = np.median(im[y1:y2,x1:x2])
            else:
                sm = np.mean(im[y1:y2,x1:x2])
            smim[y1:y2,x1:x2] = sm
    return smim

@njit
@cc.export('skygrid', '(f8[:,:],i8,i4,i4)')
def skygrid(im,binsize,tot=0,med=1):
    """ Estimate the background."""
    #binsize = 200
    ny,nx = im.shape
    ny2 = ny // binsize
    nx2 = nx // binsize
    bgim = np.zeros((ny2,nx2),float)
    nsample = np.minimum(1000,binsize*binsize)
    sample = np.random.randint(0,binsize*binsize-1,nsample)
    for i in range(nx2):
        for j in range(ny2):
            x1 = i*binsize
            x2 = x1+binsize
            if x2 > nx: x2=nx
            y1 = j*binsize
            y2 = y1+binsize
            if y2 > ny: y2=ny
            if tot==1:
                bgim[j,i] = np.sum(im[y1:y2,x1:x2].ravel()[sample])
            elif med==1:
                bgim[j,i] = np.median(im[y1:y2,x1:x2].ravel()[sample])
            else:
                bgim[j,i] = np.mean(im[y1:y2,x1:x2].ravel()[sample])
    return bgim

@njit
@cc.export('skyinterp', '(f8[:,:],f8[:,:],i8)')
def skyinterp(binim,fullim,binsize):
    """ linear interpolation"""
    ny,nx = fullim.shape
    ny2 = ny // binsize
    nx2 = nx // binsize
    hbinsize = int(0.5*binsize)

    # Calculate midpoint positions
    xx = np.arange(nx2)*binsize+hbinsize
    yy = np.arange(ny2)*binsize+hbinsize
    
    for i in range(ny2-1):
        for j in range(nx2-1):
            y1 = i*binsize+hbinsize
            y2 = y1+binsize
            x1 = j*binsize+hbinsize
            x2 = x1+binsize
            f11 = binim[i,j]
            f12 = binim[i+1,j]
            f21 = binim[i,j+1]
            f22 = binim[i+1,j+1]
            denom = binsize*binsize
            for k in range(binsize):
                for l in range(binsize):
                    x = x1+k
                    y = y1+l
                    # weighted mean
                    #denom = (x2-x1)*(y2-y1)
                    w11 = (x2-x)*(y2-y)/denom
                    w12 = (x2-x)*(y-y1)/denom
                    w21 = (x-x1)*(y2-y)/denom
                    w22 = (x-x1)*(y-y1)/denom
                    f = w11*f11+w12*f12+w21*f21+w22*f22
                    fullim[y,x] = f

    # Do the edges
    for i in range(hbinsize,nx2*binsize-hbinsize):
        # bottom
        for j in range(hbinsize):
            fullim[j,i] = fullim[hbinsize,i]
        # top
        for j in range(ny2*binsize-hbinsize-1,ny):
            fullim[j,i] = fullim[ny2*binsize-hbinsize-2,i]
    for j in np.arange(hbinsize,ny2*binsize-hbinsize):
        # left
        for i in range(hbinsize):
            fullim[j,i] = fullim[j,hbinsize]
        # right
        for i in range(nx2*binsize-hbinsize-1,nx):
            fullim[j,i] = fullim[j,nx2*binsize-hbinsize-2]
    # Do the corners
    for i in range(hbinsize):
        # bottom-left
        for j in range(hbinsize):
            fullim[j,i] = fullim[hbinsize,hbinsize]
        # top-left
        for j in range(ny2*binsize-hbinsize-1,ny):
            fullim[j,i] = fullim[ny2*binsize-hbinsize-2,hbinsize]
    for i in range(nx2*binsize-hbinsize-1,nx):
        # bottom-right
        for j in range(hbinsize):
            fullim[j,i] = fullim[hbinsize,nx2*binsize-hbinsize-2]
        # top-right
        for j in range(ny2*binsize-hbinsize-1,ny):
            fullim[j,i] = fullim[ny2*binsize-hbinsize-2,nx2*binsize-hbinsize-2]

    return fullim
                    

@njit
@cc.export('sky', '(f8[:,:],i8)')
def sky(im,binsize=0):
    ny,nx = im.shape

    # Figure out best binsize
    if binsize <= 0:
        binsize = np.min(np.array([ny//20,nx//20]))
    binsize = np.maximum(binsize,20)
    ny2 = ny // binsize
    nx2 = nx // binsize

    # Bin in a grid
    bgimbin = skygrid(im,binsize,0,1)

    # Outlier rejection
    # median smoothing
    medbin = 3
    if nx2<medbin or ny2<medbin:
        medbin = np.min(np.array([nx2,ny2]))
    smbgimbin = smooth2d(bgimbin,medbin,1)  # median smoothing
    
    # Linearly interpolate
    bgim = np.zeros(im.shape,np.float64)+np.median(bgimbin)
    bgim = skyinterp(smbgimbin,bgim,binsize)

    return bgim

# @njit
# @cc.export('numba_sky2', '(f8[:],f8[:],f8[:,:],f8[:])')
# def numba_sky2(im,binsize,tot,med):
#     """ Estimate the background."""
#     ny,nx = im.shape
#     ny2 = ny // binsize
#     nx2 = nx // binsize
#     bgim = np.zeros((ny2,nx2),float)
#     #sample = np.random.randint(0,binsize*binsize-1,1000)
#     xsample = np.random.randint(0,binsize-1,1000)
#     ysample = np.random.randint(0,binsize-1,1000)
#     data = np.zeros(1000,float)
#     for i in range(ny2):
#         for j in range(nx2):
#             x1 = i*binsize
#             x2 = x1+binsize
#             if x2 > nx: x2=nx
#             y1 = j*binsize
#             y2 = y1+binsize
#             if y2 > ny: y2=ny
#             for k in range(1000):
#                 data[k] = im[y1+ysample[k],x1+xsample[k]]
#             if tot==1:
#                 bgim[j,i] = np.sum(data)
#             elif med==1:
#                 bgim[j,i] = np.median(data)
#             else:
#                 bgim[j,i] = np.mean(data)
#     #import pdb; pdb.set_trace()

#     return bgim


# @njit
# @cc.export('detection', '(f8[:,:],i4)')
# def detection(im,nsig=10):
#     """  Detect peaks """

#     # just looping over the 9K x 9K array
#     # takes 1.3 sec

#     # bin 2x2 as a crude initial smoothing
#     imbin = dln.rebin(im,binsize=(2,2))
    
#     sig = sigma(imbin)
#     xpeak,ypeak,count = detectpeaks(imbin,sig,nsig)
#     xpeak = xpeak[:count]*2
#     ypeak = ypeak[:count]*2
    
#     return xpeak,ypeak

@njit
@cc.export('detectpeaks', '(f8[:,:],f8,i8)')
def detectpeaks(im,sig,nsig):
    """ Detect peaks"""
    # input sky subtracted image
    ny,nx = im.shape
    nbin = 3  # 5
    nhbin = nbin//2

    mnim = np.zeros(im.shape,float)-100000
    
    count = 0
    xpeak = np.zeros(100000,float)
    ypeak = np.zeros(100000,float)
    for i in np.arange(nhbin+1,nx-nhbin-2):
        for j in range(nhbin+1,ny-nhbin-2):
            if im[j,i]>nsig*sig:
                if mnim[j,i] > -1000:
                    mval = mnim[j,i]
                else:
                    mval = np.mean(im[j-nhbin:j+nhbin+1,i-nhbin:i+nhbin+1])
                    mnim[j,i] = mval
                if mnim[j,i-1] > -1000:
                    lval = mnim[j,i-1]
                else:
                    lval = np.mean(im[j-nhbin:j+nhbin+1,i-nhbin-1:i+nhbin])
                    mnim[j,i-1] = lval
                if mnim[j,i+1] > -1000:
                    rval = mnim[j,i+1]
                else:
                    rval = np.mean(im[j-nhbin:j+nhbin+1,i-nhbin+1:i+nhbin+2])
                    mnim[j,i+1] = rval
                if mnim[j-1,i] > -1000:
                    dval = mnim[j-1,i]
                else:
                    dval = np.mean(im[j-nhbin-1:j+nhbin,i-nhbin:i+nhbin+1])
                    mnim[j-1,i] = dval
                if mnim[j+1,i] > -1000:
                    uval = mnim[j+1,i]
                else:
                    uval = np.mean(im[j-nhbin+1:j+nhbin+2,i-nhbin:i+nhbin+1])
                    mnim[j+1,i] = uval
                # Check that it is a peak
                if (mval>lval and mval>rval and mval>dval and mval>uval):
                    xpeak[count] = i
                    ypeak[count] = j
                    count = count + 1
    return xpeak,ypeak,count

@njit
@cc.export('boundingbox', '(f8[:,:],i8,i8,f8,f8)')
def boundingbox(im,xp,yp,thresh,bmax):
    """ Get bounding box for the source """

    ny,nx = im.shape
    nbin = 3
    nhbin = nbin//2
    
    # Step left until you reach the threshold
    y0 = yp-nhbin
    if y0<0: y0=0
    y1 = yp+nhbin+1
    if y1>ny: y1=ny
    flag = False
    midval = np.mean(im[y0:y1,xp])
    count = 1
    while (flag==False):
        newval = np.mean(im[y0:y1,xp-count])
        if newval < thresh*midval or xp-count==0 or count==bmax:
            flag = True
        lastval = newval
        count += 1
    leftxp = xp-count+1
    # Step right until you reach the threshold
    flag = False
    count = 1
    while (flag==False):
        newval = np.mean(im[y0:y1,xp+count])
        if newval < thresh*midval or xp+count==(nx-1) or count==bmax:
            flag = True
        lastval = newval
        count += 1
    rightxp = xp+count-1
    # Step down until you reach the threshold
    x0 = xp-nhbin
    if x0<0: x0=0
    x1 = xp+nhbin+1
    if x1>nx: x1=nx
    flag = False
    midval = np.mean(im[yp,x0:x1])
    count = 1
    while (flag==False):
        newval = np.mean(im[yp-count,x0:x1])
        if newval < thresh*midval or yp-count==0 or count==bmax:
            flag = True
        lastval = newval
        count += 1
    downyp = yp-count+1
    # Step up until you reach the threshold
    flag = False
    count = 1
    while (flag==False):
        newval = np.mean(im[yp+count,x0:x1])
        if newval < thresh*midval or yp+count==(ny-1) or count==bmax:
            flag = True
        lastval = newval
        count += 1
    upyp = yp+count-1

    return leftxp,rightxp,downyp,upyp

@njit
@cc.export('morpho', '(f8[:,:],i8,i8,i8,i8,i8,i8,f8)')
def morpho(im,xp,yp,x0,x1,y0,y1,thresh):
    """ Measure morphology parameters """
    ny,nx = im.shape

    midval = im[yp,xp]
    hthresh = thresh*midval
    
    nx = x1-x0+1
    ny = y1-y0+1

    # Flux and first moments
    flux = 0.0
    mnx = 0.0
    mny = 0.0
    # X loop    
    for i in range(nx):
        x = i+x0
        # Y loop
        for j in range(ny):
            y = j+y0
            val = im[y,x]
            if val>hthresh:
                # Flux
                flux += val
                # First moments
                mnx += val*x
                mny += val*y
    mnx /= flux
    mny /= flux

    # Second moments
    sigx2 = 0.0
    sigy2 = 0.0
    sigxy = 0.0
    # X loop    
    for i in range(nx):
        x = i+x0
        # Y loop
        for j in range(ny):
            y = j+y0
            val = im[y,x]
            if val>hthresh:
                sigx2 += val*(x-mnx)**2
                sigy2 += val*(y-mny)**2
                sigxy += val*(x-mnx)*(y-mny)
    sigx2 /= flux
    sigy2 /= flux
    sigx = np.sqrt(sigx2)
    sigy = np.sqrt(sigy2)
    sigxy /= flux
    fwhm = (sigx+sigy)*0.5 * 2.35
    
    # Ellipse parameters
    asemi = np.sqrt( 0.5*(sigx2+sigy2) + np.sqrt(((sigx2-sigy2)*0.5)**2 + sigxy**2 ) )
    bsemi = np.sqrt( 0.5*(sigx2+sigy2) - np.sqrt(((sigx2-sigy2)*0.5)**2 + sigxy**2 ) )
    theta = 0.5*np.arctan2(2*sigxy,sigx2-sigy2)  # in radians

    return flux,mnx,mny,sigx,sigy,sigxy,fwhm,asemi,bsemi,theta

@njit
@cc.export('morphology', '(f8[:,:],f8[:],f8[:],f8,f8)')
def morphology(im,xpeak,ypeak,thresh,bmax):
    """ Measure morphology of the peaks."""

    ny,nx = im.shape
    nbin = 3
    nhbin = nbin//2

    mout = np.zeros((len(xpeak),17),float)
    for i in range(len(xpeak)):
        xp = int(xpeak[i])
        yp = int(ypeak[i])
        mout[i,0] = xp
        mout[i,1] = yp
        
        # Get the bounding box
        leftxp,rightxp,downyp,upyp = boundingbox(im,xp,yp,thresh,bmax)
        mout[i,2] = leftxp
        mout[i,3] = rightxp
        mout[i,4] = downyp
        mout[i,5] = upyp
        mout[i,6] = (rightxp-leftxp+1)*(upyp-downyp+1)


        # Measure morphology parameters
        out = morpho(im,xp,yp,leftxp,rightxp,downyp,upyp,thresh)
        #flux,mnx,mny,sigx,sigy,sigxy,fwhm,asemi,bsemi,theta = out
        mout[i,7:] = out

    return mout

# @njit
# def create_table(names, data):
#     """Creates a table-like structure using NumPy structured arrays."""
#     dtype = np.dtype([(name, data[0].dtype) for name in names])
#     table = np.zeros(len(data[0]), dtype=dtype)
#     for i, name in enumerate(names):
#         table[name] = data[i]
#     return table

# @njit
# def convertstringlist(norm_list):
#     numba_list = typed.List.empty_list(types.string)
#     for e in norm_list:
#         numba_list.append(e)
#     return numba_list

# @njit
# @cc.export('asciicode', 'i4(unicode_type)')
# def asciicode(a):
#     """ Get ascii code for a character."""
#     if a==',':
#         return 44
#     elif a=='.':
#         return 46
#     elif a=='0':
#         return 48
#     elif a=='1':
#         return 49
#     elif a=='2':
#         return 50
#     elif a=='3':
#         return 51
#     elif a=='4':
#         return 52
#     elif a=='5':
#         return 53
#     elif a=='6':
#         return 54
#     elif a=='7':
#         return 55
#     elif a=='8':
#         return 56
#     elif a=='9':
#         return 57
#     elif a=='A':
#         return 65
#     elif a=='B':
#         return 66
#     elif a=='C':
#         return 67
#     elif a=='D':
#         return 68
#     elif a=='E':
#         return 69
#     elif a=='F':
#         return 70
#     elif a=='G':
#         return 71
#     elif a=='H':
#         return 72
#     elif a=='I':
#         return 73
#     elif a=='J':
#         return 74
#     elif a=='K':
#         return 75
#     elif a=='L':
#         return 76
#     elif a=='M':
#         return 77
#     elif a=='N':
#         return 78
#     elif a=='O':
#         return 79
#     elif a=='P':
#         return 80
#     elif a=='Q':
#         return 81
#     elif a=='R':
#         return 82
#     elif a=='S':
#         return 83
#     elif a=='T':
#         return 84
#     elif a=='U':
#         return 85
#     elif a=='V':
#         return 86
#     elif a=='W':
#         return 87
#     elif a=='X':
#         return 88
#     elif a=='Y':
#         return 89
#     elif a=='Z':
#         return 90
#     elif a=='_':
#         return 95
#     elif a=='a':
#         return 97
#     elif a=='b':
#         return 98
#     elif a=='c':
#         return 99
#     elif a=='d':
#         return 100
#     elif a=='e':
#         return 101
#     elif a=='f':
#         return 102
#     elif a=='g':
#         return 103
#     elif a=='h':
#         return 104
#     elif a=='i':
#         return 105
#     elif a=='j':
#         return 106
#     elif a=='k':
#         return 107
#     elif a=='l':
#         return 108
#     elif a=='m':
#         return 109
#     elif a=='n':
#         return 110
#     elif a=='o':
#         return 111
#     elif a=='p':
#         return 112
#     elif a=='q':
#         return 113
#     elif a=='r':
#         return 114
#     elif a=='s':
#         return 115
#     elif a=='t':
#         return 116
#     elif a=='u':
#         return 117
#     elif a=='v':
#         return 118
#     elif a=='w':
#         return 119
#     elif a=='x':
#         return 120
#     elif a=='y':
#         return 121
#     elif a=='z':
#         return 122

# @njit
# @cc.export('asciicar', 'unicode_type(i4)')
# def asciichar(code):
#     """ Return ascii character given the code."""
#     vals = ['','','','','','','','','','',   # 0-9
#             '','','','','','','','','','',   # 10-19
#             '','','','','','','','','','',   # 20-29
#             '','','','','','','','','','',   # 30-39
#             '','','','',',','','.','','0','1',   # 40-49
#             '2','3','4','5','6','7','8','9','','',   # 50-59
#             '','','','','','A','B','C','D','E',   # 60-69
#             'F','G','H','I','J','K','L','M','N','O',   # 70-79
#             'P','Q','R','S','T','U','V','W','X','Y',   # 80-89
#             'Z','','','','','_','','a','b','c',   # 90-99
#             'd','e','f','g','h','i','j','k','l','m',   # 100-109
#             'n','o','p','q','r','s','t','u','v','w',   # 110-119
#             'x','y','z','','','','','','','']   # 120-129
#     return vals[code]
    
# @njit
# @cc.export('convertasciitoint', 'i8[:](unicode_type)')
# def convertasciitoint(data):
#     """ Convert ascii string to integer."""
#     n = len(data)
#     out = np.zeros(n,np.int64)
#     for i in range(n):
#         out[i] = asciicode(data[i])
#     return out

# @njit
# @cc.export('convertinttoascii', 'unicode_type(f8[:])')
# def convertinttoascii(data):
#     """ Convert array of ascii code integers to string."""
#     n = len(data)
#     out = ''
#     for i in range(n):
#         out += asciichar(data[i])
#     return out

# @njit
# @cc.export('asciiintvalue', 'i8(i4[:])')
# def asciiintvalue(arr):
#     """ Convert ascii code array to an integer value."""
#     out = 0
#     for i in range(len(arr)):
#         out += arr[i]*1000**i
#     return out

# @njit
# @cc.export('isinteger', 'b1(unicode_type)')
# def isinteger(val):
#     """ Check if this is an integer."""
#     sval = str(val)
#     sval1 = sval[0]
#     if (sval1=='1' or sval1=='2' or sval1=='3' or sval1=='4' or sval1=='5' or
#         sval1=='6' or sval1=='7' or sval1=='8' or sval1=='9'):
#         return True
#     else:
#         return False

# @njit
# @cc.export('formatfloat', 'unicode_type(f8,i4)')
# def formatfloat(val,ndigits):
#     """ Format a floating point number for string output."""
#     if val<0:
#         isnegative = True
#         ndig = ndigits-1
#         val = np.abs(val)
#     else:
#         isnegative = False
#         ndig = ndigits
#     exponent = int(np.log10(val))
#     if np.log10(val)<0:
#         exponent = int(np.log10(val))-1
#     significand = val/10.0**exponent
#     sigvals = str(int(significand*10.0**ndig))
#     osigvals = sigvals[0]+'.'+sigvals[1:]
#     # Whole value, >=1
#     if exponent < ndig and exponent >= 0:
#         sval = str(int(val*10.0**ndig))
#         oval = sval[:exponent+1]+'.'+sval[exponent+1:]
#         out = oval[:ndig]
#     # Whole value, <1
#     elif np.abs(exponent)<ndig and exponent<0:
#         sval = str(int(val*10.0**ndig))
#         aexponent = np.abs(exponent)
#         if aexponent > 1:
#             oval = '0.' + (aexponent-1)*'0' + sval
#         else:
#             oval = '0.'+sval
#         out = oval[:ndig]
#     # Scientific notation
#     else:
#         eout = 'e'+str(exponent)
#         out = osigvals[:ndig-len(eout)]+eout
#     if isnegative:
#         out = '-'+out
#     return out
    
# spec = [
#     #('_names', types.List(types.string)),
#     ('_namesintarray', types.int64[:,:]),
#     ('_namesnchars', types.int64[:]),
#     ('_namesintvalue', types.int64[:]),
#     ('ncols', types.int64),
#     ('nrows', types.int64),
#     ('data', types.float64[:,:]),
# ]
# @jitclass(spec)
# class Table(object):

#     def __init__(self, names, data):
#         n = len(names)
#         namesnchars = np.zeros(n,np.int64)
#         maxlength = 0
#         for i in range(n):
#             namesnchars[i] = len(names[i])
#             maxlength = np.max(np.array([maxlength,len(names[i])]))
#         self._namesnchars = namesnchars
#         namesintarray = np.zeros((n,maxlength),np.int64)-1
#         self._namesintarray = namesintarray
#         namesintvalue = np.zeros(n,np.int64)
#         self._namesintvalue = namesintvalue
#         for i in range(n):
#             nameint = convertasciitoint(names[i])
#             self._namesintarray[i,:len(nameint)] = nameint
#             self._namesintvalue[i] = asciiintvalue(nameint)
#         self.ncols = n
#         ncols,nrows = data.shape
#         self.nrows = nrows
#         self.data = np.zeros(data.shape,np.float64)
#         self.data[:,:] = data.astype(np.float64)

#     def __len__(self):
#         """ Return the number of rows."""
#         return self.nrows

#     def __str__(self):
#         """ string representation."""
#         # Header
#         out = ''
#         for i in range(self.ncols):
#             name1 = self.name(i)
#             n1 = len(name1)
#             if n1<11:
#                 out += name1+(11-n1)*' '
#             else:
#                 out += name1
#         out += '\n'
#         # Rows
#         for i in range(self.nrows):
#             out1 = ''
#             for j in range(self.ncols):
#                 out1 += formatfloat(self.data[j,i],10)+' '
#             out1 += '\n'
#             out += out1
#         return out

#     def name(self,i):
#         """ Return single name."""
#         nchar = self._namesnchars[i]
#         nameint = self._namesintarray[i,:]
#         nameint = nameint[:nchar]
#         colname = convertinttoascii(nameint)
#         return colname

#     @property
#     def names(self):
#         ls = typed.List.empty_list(types.string)
#         for i in range(self.ncols):
#             colname = self.name(i)
#             ls.append(colname)
#         return ls

#     def _getnameindex(self,name):
#         nameint = convertasciitoint(name)
#         nameval =  asciiintvalue(nameint)
#         ind, = np.where(self._namesintvalue==nameval)
#         return ind

#     def getcol(self,item):
#         data1 = np.zeros((self.ncols,1),np.float64)
#         data1[:,0] = self.data[:,item]
#         return Table(self.names,data1)
        
#     def __getitem__(self,item):
#         """ Get a single column."""
#         ind = self._getnameindex(item)
#         if len(ind)==0:
#             print('No column "'+str(item)+'"')
#             return
#         return self.data[ind,:]
        
#     def __setitem__(self,item,data):
#         """ Set the entire column."""
#         ind = self._getnameindex(item)
#         if len(ind)==0:
#             print('No column "'+str(item)+'"')
#             return
#         if len(data)!=self.nrows:
#             print(len(data),'elements input but need',self.nrows,'elements to set entire column')
#             return
#         self.data[ind[0],:] = data

#     def set(self,name,row,data):
#         """ Set a single value."""
#         ind = self._getnameindex(name)
#         if len(ind)==0:
#             print('No column "'+str(name)+'"')
#             return
#         if row>self.nrows-1:
#             print(row,'out of bounds for',self.nrows,'rows')
#         self.data[ind[0],row] = data

#     def setmany(self,name,rows,data):
#         """ Set multiple values in a single row."""
#         # rows and data must be numpy arrays
#         ind = self._getnameindex(name)
#         if len(ind)==0:
#             print('No column "'+str(name)+'"')
#             return
#         if len(rows) != len(data):
#             print('rows and data not the same length')
#             return
#         for i in range(len(rows)):
#             if rows[i]>self.nrows-1:
#                 print(rows[i],'out of bounds for',self.nrows,'rows')
#                 continue
#             self.data[ind[0],rows[i]] = data[i]


@njit
@cc.export('starbbox', 'i8[:](UniTuple(f8,2),UniTuple(i8,2),f8)')
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

    # pixels span +/-0.5 pixels in each direction
    # so pixel x=5 spans x=4.5-5.5
    # Use round() to get the value of the pixel that is covers
    # the coordinate
    # to include pixels at the bottom but not the top we have to
    # subtract a tiny bit before we round
    eta = 1e-5

    # Star coordinates
    xcen,ycen = coords
    ny,nx = imshape   # python images are (Y,X)
    x0 = xcen-radius
    x1 = xcen+radius
    y0 = ycen-radius
    y1 = ycen+radius
    xlo = np.maximum(int(np.round(x0-eta)),0)
    xhi = np.minimum(int(np.round(x1-eta))+1,nx)
    ylo = np.maximum(int(np.round(y0-eta)),0)
    yhi = np.minimum(int(np.round(y1-eta))+1,ny)
    # add 1 at the upper end because those values are EXCLUDED
    # by the standard python convention

    # The old way of doing it
    #xlo = np.maximum(int(np.floor(xcen-radius)),0)
    #xhi = np.minimum(int(np.ceil(xcen+radius+1)),nx)
    #ylo = np.maximum(int(np.floor(ycen-radius)),0)
    #yhi = np.minimum(int(np.ceil(ycen+radius+1)),ny)
    return np.array([xlo,xhi,ylo,yhi])


@njit
@cc.export('getstar', '(UniTuple(i8,2),f8,f8,i8,f8)')
def getstar(imshape,xcen,ycen,hpsfnpix,fitradius):
    """ Return a star's full footprint and fitted pixels data."""
    # always return the same size
    # a distance of 6.2 pixels spans 6 full pixels but you could have
    # 0.1 left on one side and 0.1 left on the other side
    # that's why we have to add 2 pixels
    nfpix = hpsfnpix*2+1
    fbbox = starbbox((xcen,ycen),imshape,hpsfnpix)
    nfx = fbbox[1]-fbbox[0]
    nfy = fbbox[3]-fbbox[2]
    # extra buffer is ALWAYS at the end of each dimension    
    fxdata = np.zeros(nfpix*nfpix,np.int64)-1
    fydata = np.zeros(nfpix*nfpix,np.int64)-1
    fravelindex = np.zeros(nfpix*nfpix,np.int64)-1
    fcount = 0
    # Fitting pixels
    npix = int(np.floor(2*fitradius))+2
    bbox = starbbox((xcen,ycen),imshape,fitradius)
    nx = bbox[1]-bbox[0]
    ny = bbox[3]-bbox[2]
    xdata = np.zeros(npix*npix,np.int64)-1
    ydata = np.zeros(npix*npix,np.int64)-1
    ravelindex = np.zeros(npix*npix,np.int64)-1
    mask = np.zeros(npix*npix,np.int8)
    fcount = 0
    count = 0
    for j in range(nfpix):
        y = j + fbbox[2]
        for i in range(nfpix):
            x = i + fbbox[0]
            r = np.sqrt((x-xcen)**2 + (y-ycen)**2)
            if x>=fbbox[0] and x<=fbbox[1]-1 and y>=fbbox[2] and y<=fbbox[3]-1:
                if r <= 1.0*hpsfnpix:
                    fxdata[fcount] = x
                    fydata[fcount] = y
                    fmulti_index = (np.array([y]),np.array([x]))
                    fravelindex[fcount] = ravel_multi_index(fmulti_index,imshape)[0]
                    fcount += 1
                if r <= fitradius:
                    xdata[count] = x
                    ydata[count] = y
                    multi_index = (np.array([y]),np.array([x]))
                    ravelindex[count] = ravel_multi_index(multi_index,imshape)[0]
                    mask[count] = 1
                    count += 1
    return (fxdata,fydata,fravelindex,fbbox,nfx,nfy,fcount,
            xdata,ydata,ravelindex,bbox,count,mask)

@njit
@cc.export('collatestars', '(UniTuple(i8,2),f8[:],f8[:],i8,f8)')
def collatestars(imshape,starx,stary,hpsfnpix,fitradius):
    """ Get full footprint and fitted pixels data for all stars."""
    nstars = len(starx)
    nfpix = 2*hpsfnpix+1
    npix = int(np.floor(2*fitradius))+2
    # Full footprint arrays
    fxdata = np.zeros((nstars,nfpix*nfpix),np.int64)
    fydata = np.zeros((nstars,nfpix*nfpix),np.int64)
    fravelindex = np.zeros((nstars,nfpix*nfpix),np.int64)
    fbbox = np.zeros((nstars,4),np.int32)
    fshape = np.zeros((nstars,2),np.int32)
    fndata = np.zeros(nstars,np.int32)
    # Fitting pixel arrays
    xdata = np.zeros((nstars,npix*npix),np.int64)
    ydata = np.zeros((nstars,npix*npix),np.int64)
    ravelindex = np.zeros((nstars,npix*npix),np.int64)
    bbox = np.zeros((nstars,4),np.int32)
    shape = np.zeros((nstars,2),np.int32)
    ndata = np.zeros(nstars,np.int32)
    mask = np.zeros((nstars,npix*npix),np.int8)
    for i in range(nstars):
        out = getstar(imshape,starx[i],stary[i],hpsfnpix,fitradius)
        # full footprint information
        fxdata1,fydata1,fravelindex1,fbbox1,fnx1,fny1,fn1 = out[:7]
        fxdata[i,:] = fxdata1
        fydata[i,:] = fydata1
        fravelindex[i,:] = fravelindex1
        fbbox[i,:] = fbbox1
        fshape[i,0] = fny1
        fshape[i,1] = fnx1
        fndata[i] = fn1
        # fitting pixel information
        xdata1,ydata1,ravelindex1,bbox1,n1,mask1 = out[7:]
        xdata[i,:] = xdata1
        ydata[i,:] = ydata1
        ravelindex[i,:] = ravelindex1
        bbox[i,:] = bbox1
        ndata[i] = n1
        mask[i,:] = mask1
    # Trim arrays
    maxfn = np.max(fndata)
    fxdata = fxdata[:,:maxfn]
    fydata = fydata[:,:maxfn]
    fravelindex = fravelindex[:,:maxfn]
    maxn = np.max(ndata)
    xdata = xdata[:,:maxn]
    ydata = ydata[:,:maxn]
    ravelindex = ravelindex[:,:maxn]
    mask = mask[:,:maxn]
    
    return (fxdata,fydata,fravelindex,fbbox,fshape,fndata,
            xdata,ydata,ravelindex,bbox,ndata,mask)


@njit
@cc.export('getfullstar', '(UniTuple(i8,2),f8,f8,i8)')
def getfullstar(imshape,xcen,ycen,hpsfnpix):
    """ Return the entire footprint image/error/x/y arrays for one star."""
    # always return the same size
    # a distance of 6.2 pixels spans 6 full pixels but you could have
    # 0.1 left on one side and 0.1 left on the other side
    # that's why we have to add 2 pixels
    npix = hpsfnpix*2+1
    bbox = starbbox((xcen,ycen),imshape,hpsfnpix)
    nx = bbox[1]-bbox[0]
    ny = bbox[3]-bbox[2]
    # extra buffer is ALWAYS at the end of each dimension    
    xdata = np.zeros(npix*npix,np.int32)-1
    ydata = np.zeros(npix*npix,np.int32)-1
    ravelindex = np.zeros(npix*npix,np.int64)-1
    count = 0
    for j in range(npix):
        y = j + bbox[2]
        for i in range(npix):
            x = i + bbox[0]
            if x>=bbox[0] and x<=bbox[1]-1 and y>=bbox[2] and y<=bbox[3]-1:
                xdata[count] = x
                ydata[count] = y
                multi_index = (np.array([y]),np.array([x]))
                ravelindex[count] = ravel_multi_index(multi_index,imshape)[0]
                count += 1
    return xdata,ydata,ravelindex,bbox,nx,ny,count

@njit
@cc.export('collatefullstars', '(UniTuple(i8,2),f8[:],f8[:],i8)')
def collatefullstars(imshape,starx,stary,hpsfnpix):
    """ Get the entire footprint image/error/x/y for all of the stars."""
    nstars = len(starx)
    npix = 2*hpsfnpix+1
    # Get xdata, ydata, error
    xdata = np.zeros((nstars,npix*npix),np.int32)
    ydata = np.zeros((nstars,npix*npix),np.int32)
    ravelindex = np.zeros((nstars,npix*npix),np.int64)
    bbox = np.zeros((nstars,4),np.int32)
    shape = np.zeros((nstars,2),np.int32)
    ndata = np.zeros(nstars,np.int32)
    for i in range(nstars):
        xdata1,ydata1,ravelindex1,bbox1,nx1,ny1,n1 = getfullstar(imshape,starx[i],stary[i],hpsfnpix)
        xdata[i,:] = xdata1
        ydata[i,:] = ydata1
        ravelindex[i,:] = ravelindex1
        bbox[i,:] = bbox1
        shape[i,0] = ny1
        shape[i,1] = nx1
        ndata[i] = n1
    return xdata,ydata,ravelindex,bbox,shape,ndata

@njit
@cc.export('getfitstar', '(UniTuple(i8,2),f8,f8,f8)')
def getfitstar(imshape,xcen,ycen,fitradius):
    """ Get the fitting pixel information for a single star."""
    npix = int(np.floor(2*fitradius))+2
    bbox = starbbox((xcen,ycen),imshape,fitradius)
    nx = bbox[1]-bbox[0]
    ny = bbox[3]-bbox[2]
    xdata = np.zeros(npix*npix,np.int32)-1
    ydata = np.zeros(npix*npix,np.int32)-1
    ravelindex = np.zeros(npix*npix,np.int64)-1
    mask = np.zeros(npix*npix,np.int32)
    count = 0
    for j in range(ny):
        y = j + bbox[2]
        for i in range(nx):
            x = i + bbox[0]
            r = np.sqrt((x-xcen)**2 + (y-ycen)**2)
            if r <= fitradius:
                xdata[count] = x
                ydata[count] = y
                multi_index = (np.array([y]),np.array([x]))
                ravelindex[count] = ravel_multi_index(multi_index,imshape)[0]
                mask[count] = 1
                count += 1
    return xdata,ydata,ravelindex,count,mask
        
@njit
@cc.export('collatefitstars', '(UniTuple(i8,2),f8[:],f8[:],f8)')
def collatefitstars(imshape,starx,stary,fitradius):
    """ Get the fitting pixel information for all stars."""
    nstars = len(starx)
    npix = int(np.floor(2*fitradius))+2
    # Get xdata, ydata, error
    maxpix = nstars*(npix)**2
    xdata = np.zeros((nstars,npix*npix),np.int32)
    ydata = np.zeros((nstars,npix*npix),np.int32)
    ravelindex = np.zeros((nstars,npix*npix),np.int64)
    mask = np.zeros((nstars,npix*npix),np.int32)
    ndata = np.zeros(nstars,np.int32)
    bbox = np.zeros((nstars,4),np.int32)
    for i in range(nstars):
        xcen = starx[i]
        ycen = stary[i]
        bb = starbbox((xcen,ycen),imshape,fitradius)
        xdata1,ydata1,ravelindex1,n1,mask1 = getfitstar(imshape,xcen,ycen,fitradius)
        xdata[i,:] = xdata1
        ydata[i,:] = ydata1
        ravelindex[i,:] = ravelindex1
        mask[i,:] = mask1
        ndata[i] = n1
        bbox[i,:] = bb
    return xdata,ydata,ravelindex,bbox,ndata,mask

@njit
@cc.export('skyval', 'f8(f8[:],f8[:])')
@cc.export('skyvali', 'f8(i8[:],f8[:])')
def skyval(array,sigma):
    """  Estimate sky value from sky pixels."""
    wt = 1/sigma**2
    med = np.median(array)
    sig = mad(array)
    # reweight outlier pixels using Stetson's method
    resid = array-med
    wt2 = wt/(1+np.abs(resid)**2/np.median(sigma))
    xmn = np.sum(wt2*array)/np.sum(wt2)
    return xmn

@njit
@cc.export('mkbounds', '(f8[:],UniTuple(i8,2),i8)')
def mkbounds(pars,imshape,xoff=10):
    """ Make bounds for a set of input parameters."""
    # is [amp1,xcen1,ycen1,amp2,xcen2,ycen2, ...]
    npars = len(pars)
    nstars = npars // 3
    skyfit = (npars % 3 != 0)
    ny,nx = imshape
    # Make bounds
    lbounds = np.zeros(npars,float)
    ubounds = np.zeros(npars,float)
    lbounds[0:3*nstars:3] = 0
    lbounds[1:3*nstars:3] = np.maximum(pars[1:3*nstars:3]-xoff,0)
    lbounds[2:3*nstars:3] = np.maximum(pars[2:3*nstars:3]-xoff,0)
    if skyfit:
        lbounds[-1] = -np.inf
    ubounds[0:3*nstars:3] = np.inf
    ubounds[1:3*nstars:3] = np.minimum(pars[1:3*nstars:3]+xoff,nx-1)
    ubounds[2:3*nstars:3] = np.minimum(pars[2:3*nstars:3]+xoff,ny-1)
    if skyfit:
        ubounds[-1] = np.inf
    bounds = (lbounds,ubounds)
    return bounds

@njit
@cc.export('steps', 'f8[:](f8[:],f8)')
@cc.export('stepsi', 'f8[:](i8[:],f8)')
def steps(pars,dx=0.5):
    """ Return step sizes to use when fitting the stellar parameters."""
    npars = len(pars)
    nstars = npars // 3
    skyfit = (npars % 3 != 0)
    fsteps = np.zeros(npars,float)
    fsteps[0:3*nstars:3] = np.maximum(np.abs(pars[0:3*nstars:3])*0.5,1)
    fsteps[1:3*nstars:3] = dx        
    fsteps[2:3*nstars:3] = dx
    if skyfit:
        fsteps[-1] = 10
    return fsteps

@njit
@cc.export('newpars', 'f8[:](f8[:],f8[:],UniTuple(f8[:],2),f8[:])')
def newpars(pars,steps,bounds,maxsteps):
    """ Get new parameters given initial parameters, steps and constraints."""
    # Limit the steps to maxsteps
    limited_steps = limsteps(steps,maxsteps)
    # Make sure that these don't cross the boundaries
    lbounds,ubounds = bounds
    check = checkbounds(pars+limited_steps,bounds)
    # Reduce step size for any parameters to go beyond the boundaries
    badpars = (check!=0)
    # reduce the step sizes until they are within bounds
    newsteps = limited_steps.copy()
    count = 0
    maxiter = 2
    while (np.sum(badpars)>0 and count<=maxiter):
        newsteps[badpars] /= 2
        newcheck = checkbounds(pars+newsteps,bounds)
        badpars = (newcheck!=0)
        count += 1
            
    # Final parameters
    newpars = pars + newsteps
            
    # Make sure to limit them to the boundaries
    check = checkbounds(newpars,bounds)
    badpars = (check!=0)
    if np.sum(badpars)>0:
        # add a tiny offset so it doesn't fit right on the boundary
        newpars = np.minimum(np.maximum(newpars,lbounds+1e-30),ubounds-1e-30)
    return newpars

@njit
@cc.export('doPolygonsOverlap', '(f8[:],f8[:],f8[:],f8[:])')
@cc.export('doPolygonsOverlapi', '(i8[:],i8[:],i8[:],i8[:])')
def doPolygonsOverlap(xPolygon1, yPolygon1, xPolygon2, yPolygon2):
    """Returns True if two polygons are overlapping."""

    # How to determine if two polygons overlap.
    # If a vertex of one of the polygons is inside the other polygon
    # then they overlap.
    
    n1 = len(xPolygon1)
    n2 = len(xPolygon2)
    isin = False

    # If ranges don't overlap, then polygons don't overlap
    if rangeoverlap(xPolygon1,xPolygon2)==False or rangeoverlap(yPolygon1,yPolygon2)==False:
        return False
    
    # Loop through all vertices of second polygon
    for i in range(n2):
        # perform iterative boolean OR
        # if any point is inside the polygon then they overlap   
        isin = isin or isPointInPolygon(xPolygon1, yPolygon1, xPolygon2[i], yPolygon2[i])

    # Need to do the reverse as well, not the same
    for i in range(n1):
        isin = isin or isPointInPolygon(xPolygon2, yPolygon2, xPolygon1[i], yPolygon1[i])

    # Two polygons can overlap even if there are no vertices inside each other.
    # Need to check if the line segments overlap
    if isin==False:
        intersect = False
        # Add first vertex to the end
        xp1 = np.append( xPolygon1, xPolygon1[0] )
        yp1 = np.append( yPolygon1, yPolygon1[0] )
        xp2 = np.append( xPolygon2, xPolygon2[0] )
        yp2 = np.append( yPolygon2, yPolygon2[0] )
        for i in range(4):
            for j in range(4):
                intersect = intersect or doLineSegmentsIntersect(xp1[i:i+2],yp1[i:i+2],xp2[j:j+2],yp2[j:j+2])
                if intersect==True:
                    return True
        isin = isin or intersect
        
    return isin

@njit
@cc.export('isPointInPolygon', '(f8[:],f8[:],f8,f8)')
@cc.export('isPointInPolygoni', '(i8[:],i8[:],i8,i8)')
def isPointInPolygon(xPolygon, yPolygon, xPt, yPt):
    """Returns boolean if a point is inside a polygon of vertices."""
    
    # How to tell if a point is inside a polygon:
    # Determine the change in angle made by the point and the vertices
    # of the polygon.  Add up the delta(angle)'s from the first (include
    # the first point again at the end).  If the point is inside the
    # polygon, then the total angle will be +/-360 deg.  If the point is
    # outside, then the total angle will be 0 deg.  Points on the edge will
    # outside.
    # This is called the Winding Algorithm
    # http://geomalgorithms.com/a03-_inclusion.html

    n = len(xPolygon)
    # Array for the angles
    angle = np.zeros(n)

    # add first vertex to the end
    xPolygon1 = np.append( xPolygon, xPolygon[0] )
    yPolygon1 = np.append( yPolygon, yPolygon[0] )

    wn = 0   # winding number counter

    # Loop through the edges of the polygon
    for i in range(n):
        # if edge crosses upward (includes its starting endpoint, and excludes its final endpoint)
        if yPolygon1[i] <= yPt and yPolygon1[i+1] > yPt:
            # if (P is  strictly left of E[i])    // Rule #4
            if isLeft(xPolygon1[i], yPolygon1[i], xPolygon1[i+1], yPolygon1[i+1], xPt, yPt) > 0: 
                 wn += 1   # a valid up intersect right of P.x

        # if edge crosses downward (excludes its starting endpoint, and includes its final endpoint)
        if yPolygon1[i] > yPt and yPolygon1[i+1] <= yPt:
            # if (P is  strictly right of E[i])    // Rule #4
            if isLeft(xPolygon1[i], yPolygon1[i], xPolygon1[i+1], yPolygon1[i+1], xPt, yPt) < 0: 
                 wn -= 1   # a valid up intersect right of P.x

    # wn = 0 only when P is outside the polygon
    if wn == 0:
        return False
    else:
        return True

@njit
@cc.export('isLeft', '(f8,f8,f8,f8,f8,f8)')
@cc.export('isLefti', '(i8,i8,i8,i8,i8,i8)')
def isLeft(x1, y1, x2, y2, x3, y3):
    # isLeft(): test if a point is Left|On|Right of an infinite 2D line.
    #   From http://geomalgorithms.com/a01-_area.html
    # Input:  three points P1, P2, and P3
    # Return: >0 for P3 left of the line through P1 to P2
    # =0 for P3 on the line
    # <0 for P3 right of the line
    return ( (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1) )

@njit
@cc.export('rangeoverlap', '(f8[:],f8[:])')
@cc.export('rangeoverlapi', '(i8[:],i8[:])')
def rangeoverlap(a,b):
    """does the range (start1, end1) overlap with (start2, end2)"""
    return max(a) >= min(b) and min(a) <= max(b)

@njit
@cc.export('doLineSegmentsIntersect', '(f8[:],f8[:],f8[:],f8[:])')
@cc.export('doLineSegmentsIntersecti', '(i8[:],i8[:],i8[:],i8[:])')
def doLineSegmentsIntersect(x1, y1, x2, y2):
    """ Do two line segments intersect."""

    # Check vertical lines

    # Vertical lines, but NOT same X-values
    if x1[0]==x1[1] and x2[0]==x2[1] and x1[0]!=x2[0]:
        return False  # No overlap
    
    # Vertical lines with same X values
    if x1[0]==x1[1] and x2[0]==x2[1] and x1[0]==x2[0]:
        # Check intersection of Y ranges
        I1 = [np.min(y1), np.max(y1)]
        I2 = [np.min(y2), np.max(y2)]
    
        # And we could say that Xa is included into :
        Ia = [max( np.min(y1), np.min(y2) ),
              min( np.max(y1), np.max(y2) )]
    
        # Now, we need to check that this interval Ia exists :
        if rangeoverlap(y1,y2)==False:
            return False  # There is no mutual abcisses        
        else:
            return True   # There is overlap

    # The equation of a line is:
    #
    # f(x) = A*x + b = y
    # For a segment, it is exactly the same, except that x is included on an interval I.
    # 
    # If you have two segments, defined as follow:
    #
    # Segment1 = {(X1, Y1), (X2, Y2)}
    # Segment2 = {(X3, Y3), (X4, Y4)}
    # The abcisse Xa of the potential point of intersection (Xa,Ya) must be contained in both interval I1 and I2, defined as follow:
    I1 = [np.min(x1), np.max(x1)]
    I2 = [np.min(x2), np.max(x2)]
    
    # And we could say that Xa is included into :
    Ia = [max( np.min(x1), np.min(x2) ),
          min( np.max(x1), np.max(x2) )]
    
    # Now, we need to check that this interval Ia exists :
    if rangeoverlap(x1,x2)==False:        
        return False  # There is no mutual abcisses

    # Check that the Y-ranges overlap as well
    if rangeoverlap(y1,y2)==False:        
        return False  # There is no mutual y-value overlap
    
    # So, we have two line formula, and a mutual interval. Your line formulas are:
    # f1(x) = m1*x + b1 = y
    # f2(x) = m2*x + b2 = y
    
    # As we got two points by segment, we are able to determine A1, A2, b1 and b2:
    dx1 = x1[1]-x1[0]
    if dx1==0:
        m1 = np.inf
        b1 = 0
    else:
        m1 = (y1[1]-y1[0])/(x1[1]-x1[0])  # Pay attention to not dividing by zero
        b1 = y1[0]-m1*x1[0]        
    dx2 = x2[1]-x2[0]
    if dx2==0:
        m2 = np.inf
        b2 = 0
    else:
        m2 = (y2[1]-y2[0])/(x2[1]-x2[0])  # Pay attention to not dividing by zero
        b2 = y2[0]-m2*x2[0]

    # If the segments are parallel, then m1 == m2:
    if (m1 == m2) and (b1 != b2):
        return False  # Parallel segments
    
    # If the segments are parallel and on top of each other, the m1==m2 and b1==b2
    # we've already required that the x-ranges (abcissas) overlap
    if (m1 == m2) and (b1 == b2):
        return True   # parallel segments on top of each other
    
    # A point (Xa,Ya) standing on both lines must satisfy both formulas f1 and f2:
    # Ya = m1 * Xa + b1
    # Ya = m2 * Xa + b2
    # A1 * Xa + b1 = m2 * Xa + b2

    # Line segment 1 is vertical line 
    if x1[0]==x1[1]:
        Xa = x1[0]
        Ya = m2*Xa+b2
        if rangeoverlap(x1,[Xa]) and rangeoverlap(x2,[Xa]) and \
           rangeoverlap(y1,[Ya]) and rangeoverlap(y2,[Ya]):
            return True
        else:
            return False
    # Line semgent 2 is vertical line
    elif x2[0]==x2[1]:
        Xa = x2[0]
        Ya = m1*Xa+b1
        if rangeoverlap(x1,[Xa]) and rangeoverlap(x2,[Xa]) and \
           rangeoverlap(y1,[Ya]) and rangeoverlap(y2,[Ya]):        
            return True
        else:
            return False
    # Neither are vertical lines
    else:
        Xa = (b2 - b1) / (m1 - m2)
        
    # The last thing to do is check that Xa is included into Ia:
    if ( (Xa < max( np.min(x1), np.min(x2) )) or
         (Xa > min( np.max(x1), np.max(x2) )) ):
        return False  # intersection is out of bound
    else:
        return True




if __name__ == "__main__":
    cc.compile()

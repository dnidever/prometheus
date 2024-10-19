import os
import numpy as np
import numba
from numba import njit,types,from_dtype
from numba.typed import Dict
from numba.experimental import jitclass
from numba_kdtree import KDTree
from . import models_numba as mnb
from .clock import clock

# Fit a PSF model to multiple stars in an image

PI = 3.141592653589793

@njit
def nansum(data):
    """ Get the sum ignoring nans """
    data1d = data.ravel()
    gd, = np.where(np.isfinite(data1d)==True)
    if len(gd)==0:
        sm = np.nan
    else:
        sm = np.sum(data1d[gd])
    return sm

@njit
def sum(data,ignore_nan=False):
    """Get the sum."""
    if ignore_nan:
        sm = nansum(data.ravel())
    else:
        sm = np.sum(data.ravel())
    return sm

@njit
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
def nanmean(data):
    """ Get the mean ignoring nans """
    data1d = data.ravel()
    gd, = np.where(np.isfinite(data1d)==True)
    if len(gd)==0:
        mn = np.nan
    else:
        mn = np.mean(data1d[gd])
    return mn

@njit
def mean(data,ignore_nan=False):
    """Get the mean."""
    if ignore_nan:
        mn = nanmean(data.ravel())
    else:
        mn = np.mean(data.ravel())
    return mn

@njit
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
def median(data,ignore_nan=False):
    """Get the median."""
    if ignore_nan:
        med = nanmedian(data.ravel())
    else:
        med = np.median(data.ravel())
    return med

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
    dof = npix-npars
    if dof<=0:
        dof = 1
    cov = cov_orig * (chisq/dof)  # what MPFITFUN suggests, but very small
        
    return cov

@njit
def checkbounds(pars,bounds):
    """ Check the parameters against the bounds."""
    # 0 means it's fine
    # 1 means it's beyond the lower bound
    # 2 means it's beyond the upper bound
    npars = len(pars)
    lbounds,ubounds = bounds[:,0],bounds[:,1]
    check = np.zeros(npars,np.int32)
    badlow, = np.where(pars<=lbounds)
    if len(badlow)>0:
        check[badlow] = 1
    badhigh, = np.where(pars>=ubounds)
    if len(badhigh)>0:
        check[badhigh] = 2
    return check

@njit
def limbounds(pars,bounds):
    """ Limit the parameters to the boundaries."""
    lbounds,ubounds = bounds[:,0],bounds[:,1]
    outpars = np.minimum(np.maximum(pars,lbounds),ubounds)
    return outpars

@njit
def limsteps(steps,maxsteps):
    """ Limit the parameter steps to maximum step sizes."""
    signs = np.sign(steps)
    outsteps = np.minimum(np.abs(steps),maxsteps)
    outsteps *= signs
    return outsteps

@njit
def newpars(pars,steps,bounds=None,maxsteps=None):
    """ Return new parameters that fit the constraints."""
    # Limit the steps to maxsteps
    if maxsteps is not None:
        limited_steps = limsteps(steps,maxsteps)
    else:
        limited_steps = steps

    # No bounds input
    if bounds is None:
        return pars+limited_steps
        
    # Make sure that these don't cross the boundaries
    lbounds,ubounds = bounds[:,0],bounds[:,1]
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
def poly2d(xdata,pars):
    """ model of 2D linear polynomial."""
    x = xdata[:,0]
    y = xdata[:,1]
    return pars[0]+pars[1]*x+pars[2]*y+pars[3]*x*y

@njit
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
    wt = 1.0/error1.ravel()**2
    while (count<maxiter and maxpercdiff>minpercdiff):
        # Use Cholesky, QR or SVD to solve linear system of equations
        m,j = jacpoly2d(xdata,bestpar)
        dy = data1.ravel()-m.ravel()
        # Solve Jacobian
        #if error is not None:
        #dbeta = qr_jac_solve(j,dy,weight=wt)
        dbeta = qr_jac_solve(j,dy)
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
            
# from astroML
@njit
def crossmatch(X1, X2, max_distance=np.inf,k=1):
    """Cross-match the values between X1 and X2

    By default, this uses a KD Tree for speed.

    Parameters
    ----------
    X1 : array_like
        first dataset, shape(N1, D)
    X2 : array_like
        second dataset, shape(N2, D)
    max_distance : float (optional)
        maximum radius of search.  If no point is within the given radius,
        then inf will be returned.

    Returns
    -------
    dist, ind: ndarrays
        The distance and index of the closest point in X2 to each point in X1
        Both arrays are length N1.
        Locations with no match are indicated by
        dist[i] = inf, ind[i] = N2
    """
    #X1 = np.asarray(X1, dtype=float)
    #X2 = np.asarray(X2, dtype=float)

    N1, D1 = X1.shape
    N2, D2 = X2.shape

    if D1 != D2:
        raise ValueError('Arrays must have the same second dimension')

    kdt = KDTree(X2)

    dist, ind, neigh = kdt.query(X1, k=k, distance_upper_bound=max_distance)

    return dist, ind
    

# from astroML, modified by D. Nidever
@njit
def xmatch(ra1, dec1, ra2, dec2, dcr=2.0, unique=False, sphere=True):
    """Cross-match angular values between RA1/DEC1 and RA2/DEC2

    Find the closest match in the second list for each element
    in the first list and within the maximum distance.

    By default, this uses a KD Tree for speed.  Because the
    KD Tree only handles cartesian distances, the angles
    are projected onto a 3D sphere.

    This can return duplicate matches if there is an element
    in the second list that is the closest match to two elements
    of the first list.

    Parameters
    ----------
    ra1/dec1 : array_like
        first dataset, arrays of RA and DEC
        both measured in degrees
    ra2/dec2 : array_like
        second dataset, arrays of RA and DEC
        both measured in degrees
    dcr : float (optional)
        maximum radius of search, measured in arcsec.
        This can be an array of the same size as ra1/dec1.
    unique : boolean, optional
        Return unique one-to-one matches.  Default is False and
           allows duplicates.
    sphere : boolean, optional
        The coordinates are spherical in degrees.  Otherwise, the dcr
          is assumed to be in the same units as the input values.
          Default is True.


    Returns
    -------
    ind1, ind2, dist: ndarrays
        The indices for RA1/DEC1 (ind1) and for RA2/DEC2 (ind2) of the
        matches, and the distances (in arcsec).
    """
    n1 = len(ra1)
    n2 = len(ra2)
    X1 = np.zeros((n1,2),float)
    X1[:,0] = ra1
    X1[:,1] = dec1
    X2 = np.zeros((n2,2),float)
    X2[:,0] = ra2
    X2[:,1] = dec2
    
    # Spherical coordinates in degrees
    if sphere:
        X1 = X1 * (np.pi / 180.)
        X2 = X2 * (np.pi / 180.)
        #if utils.size(dcr)>1:
        #    max_distance = (np.max(dcr) / 3600) * (np.pi / 180.)
        #else:
        #    max_distance = (dcr / 3600) * (np.pi / 180.)
        max_distance = (dcr / 3600) * (np.pi / 180.)
        
        # Convert 2D RA/DEC to 3D cartesian coordinates
        Y1 = np.zeros((n1,3),float)
        Y1[:,0] = np.cos(X1[:, 0]) * np.cos(X1[:, 1])
        Y1[:,1] = np.sin(X1[:, 0]) * np.cos(X1[:, 1])
        Y1[:,2] = np.sin(X1[:, 1])
        #Y1 = np.transpose(np.vstack([np.cos(X1[:, 0]) * np.cos(X1[:, 1]),
        #                             np.sin(X1[:, 0]) * np.cos(X1[:, 1]),
        #                             np.sin(X1[:, 1])]))
        Y2 = np.zeros((n2,3),float)
        Y2[:,0] = np.cos(X2[:, 0]) * np.cos(X2[:, 1])
        Y2[:,1] = np.sin(X2[:, 0]) * np.cos(X2[:, 1])
        Y2[:,2] = np.sin(X2[:, 1])
        #Y2 = np.transpose(np.vstack([np.cos(X2[:, 0]) * np.cos(X2[:, 1]),
        #                             np.sin(X2[:, 0]) * np.cos(X2[:, 1]),
        #                             np.sin(X2[:, 1])]))

        # law of cosines to compute 3D distance
        max_y = np.sqrt(2 - 2 * np.cos(max_distance))
        k = 1 if unique is False else 10
        dist, ind = crossmatch(Y1, Y2, max_y, k=k)
        # dist has shape [N1,10] or [N1,1] (if unique)
    
        # convert distances back to angles using the law of tangents
        not_infy,not_infx = np.where(~np.isinf(dist))
        #x = 0.5 * dist[not_infy,not_infx]
        #dist[not_infy,not_infx] = (180. / np.pi * 2 * np.arctan2(x,
        #                           np.sqrt(np.maximum(0, 1 - x ** 2))))
        #dist[not_infy,not_infx] *= 3600.0      # in arcsec
    # Regular coordinates
    else:
        k = 1 if unique is False else 10
        dist, ind = crossmatch(X1, X2, dcr, k=k)
        #dist, ind = crossmatch(X1, X2, np.max(dcr), k=k)
        not_infy,not_infx = np.where(~np.isinf(dist))
        
    # Allow duplicates
    if unique==False:

        # no matches
        if len(not_infx)==0:
            return np.array([-1]), np.array([-1]), np.array([np.inf])
        
        # If DCR is an array then impose the max limits for each element
        #if utils.size(dcr)>1:
        #    bd,nbd = utils.where(dist > dcr)
        #    if nbd>0:
        #        dist[bd] = np.inf
        #        not_inf = ~np.isinf(dist)
        
        # Change to the output that I want
        # dist is [N1,1] if unique==False
        ind1 = np.arange(len(ra1))[not_infy]
        ind2 = ind[not_infy,0]
        mindist = dist[not_infy,0]
        
    # Return unique one-to-one matches
    else:

        # no matches
        if np.sum(~np.isinf(dist[:,0]))==0:
            return np.array([-1]), np.array([-1]), np.array([np.inf])
        
        done = 0
        niter = 1
        # Loop until we converge
        while (done==0):

            # If DCR is an array then impose the max limits for each element
            #if utils.size(dcr)>1:
            #    bd,nbd = utils.where(dist[:,0] > dcr)
            #    if nbd>0:
            #        for i in range(nbd):
            #            dist[bd[i],:] = np.inf

            # no matches
            if np.sum(~np.isinf(dist[:,0]))==0:
                return np.array([-1]), np.array([-1]), np.array([np.inf])

            # closest matches
            not_inf1 = ~np.isinf(dist[:,0])
            not_inf1_ind, = np.where(~np.isinf(dist[:,0]))
            ind1 = np.arange(len(ra1))[not_inf1]  # index into original ra1/dec1 arrays
            ind2 = ind[:,0][not_inf1]             # index into original ra2/dec2 arrays
            mindist = dist[:,0][not_inf1]
            if len(ind2)==0:
                return np.array([-1]), np.array([-1]), np.array([np.inf])
            find2 = np.zeros(len(ind2),float)
            find2[:] = ind2
            index = Index(find2)
            # some duplicates to deal with
            bd, = np.where(index.num>1)
            nbd = len(bd)
            if nbd>0:
                ntorem = 0
                for i in range(nbd):
                    ntorem += index.num[bd[i]]-1
                torem = np.zeros(ntorem,np.int32)  # index into shortened ind1/ind2/mindist
                trcount = 0
                for i in range(nbd):
                    # index into shortened ind1/ind2/mindist
                    indx = index.getindex(bd[i])
                    #indx = index['index'][index['lo'][bd[i]]:index['hi'][bd[i]]+1]
                    # keep the one with the smallest minimum distance
                    si = np.argsort(mindist[indx])
                    if index.num[bd[i]]>2:
                        bad = indx[si[1:]]
                        torem[trcount:trcount+len(bad)] = bad    # add
                        trcount += len(bad)
                    else:
                        torem[trcount:trcount+1] = indx[si[1:]][0]  # add single element
                #ntorem = utils.size(torem)
                torem_orig_index = not_inf1_ind[torem]  # index into original ind/dist arrays
                # For each object that was "removed" and is now unmatched, check the next possible
                # match and move it up in the dist/ind list if it isn't INF
                for i in range(ntorem):
                    # There is a next possible match 
                    if ~np.isinf(dist[torem_orig_index[i],niter-1]):
                        temp = np.zeros(10,np.int64)
                        temp[niter:] = ind[torem_orig_index[i],niter:]  #.squeeze()
                        temp[-niter:] = np.zeros(niter,np.int64)-1
                        ind[torem_orig_index[i],:] = temp
                        temp2 = np.zeros(10,float)
                        temp2[niter:] = dist[torem_orig_index[i],niter:]   #.squeeze()
                        temp2[-niter:] = np.zeros(niter,float)+np.inf
                        dist[torem_orig_index[i],:] = temp2
                        #ind[torem_orig_index[i],:] = np.hstack( (ind[torem_orig_index[i],niter:].squeeze(),
                        #                                         np.repeat(-1,niter)) )
                        #dist[torem_orig_index[i],:] = np.hstack( (dist[torem_orig_index[i],niter:].squeeze(),
                        #                                          np.repeat(np.inf,niter)) )
                    # All INFs
                    else:
                        ind[torem_orig_index[i],:] = -1
                        dist[torem_orig_index[i],:] = np.inf
                        # in the next iteration these will be ignored
            else:
                ntorem = 0

            niter += 1
            # Are we done, no duplicates or hit the maximum 10
            if (ntorem==0) or (niter>=10): done=1
                                
    return ind1, ind2, mindist

@njit
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

@njit
def unique_index(array):
    """ return unique values, index and reverse indez."""
    index = Index(array)
    uvals = array[index.index]
    # Unique index
    uindex = index.index
    # Inverse index
    #   the inverse index list takes you from the original/duplicated pixels
    #   to the unique ones
    invindex = index.invindex
    return uvals,uindex,invindex


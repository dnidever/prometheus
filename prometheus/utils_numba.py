import os
import numpy as np
from numba import njit,types,from_dtype
from numba.experimental import jitclass
from . import models_numba as mnb

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

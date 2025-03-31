# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3

import cython
cimport cython
import numpy as np
cimport numpy as np
from cython.view cimport array as cvarray
#from scipy.special import gamma, gammaincinv, gammainc

from libc.math cimport exp,sqrt,atan2,pi,NAN,floor
from libcpp cimport bool

cdef extern from "math.h":
    double sin(double x)
    double cos(double x)
    #double atan2(double x)

cdef extern from "math.h":
    bint isnan(double x)

# https://stackoverflow.com/questions/8353076/how-do-i-pass-a-pointer-to-a-c-fun$
cdef extern from "stdlib.h":
    ctypedef void const_void "const void"
    void qsort(void *base, int nmemb, int size,
                int(*compar)(const_void *, const_void *)) nogil

cdef int mycmp(const_void * pa, const_void * pb) noexcept:
    cdef double a = (<double *>pa)[0]
    cdef double b = (<double *>pb)[0]
    if a < b:
        return -1
    elif a > b:
        return 1
    else:
        return 0

cdef void myqsort(double * y, ssize_t l) nogil:
    qsort(y, l, sizeof(double), mycmp)


cdef double linearinterp(double[:,:] data, double x, double y):
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
    cdef int nx,ny,x1,x2,y1,y2
    cdef double w11,w12,w21,w22,f

    ny = data.shape[0]
    nx = data.shape[1]

    # Out of bounds
    if x<0 or x>(nx-1) or y<0 or y>(ny-1):
        f = NAN
        return f

    x1 = int(floor(x))
    if x1==nx-1:
        x1 -= 1
    x2 = x1+1
    y1 = int(floor(y))
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

cdef double[:] alinearinterp(double[:,:] data, double[:] x, double[:] y):
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
    cdef Py_ssize_t npix
    #npix = x.size
    npix = len(x)
    #f = np.zeros(npix,float)
    f = cvarray(shape=(100),itemsize=sizeof(double),format="d")
    cdef double[:] mf = f
    for i in range(npix):
       mf[i] = linearinterp(data,x[i],y[i])
    return mf

cpdef double nanmean(double[:] data):
    """ Calculate the mean of an array checking for nans."""
    cdef long n = len(data)
    cdef double result
    cdef long ngood = 0

    result = 0.0
    for i in range(n):
        if isnan(data[i])==False:
            ngood += 1
            result += data[i]
    if ngood > 0:
        result = result / ngood
    else:
        result = NAN
    return result

cpdef tuple nonnanindexes(double[:] data):
    """ Return array of non-nan index values """
    cdef long[:] index
    cdef long n
    cdef long count = 0
    n = len(data)
    index = cvarray(shape=(n,),itemsize=sizeof(long),format="l")
    #index = np.empty(n,int)
    cdef long[:] mindex = index
    count = 0
    for i in range(n):
        if isnan(data[i])==False:
            mindex[count] = i
            count += 1
    return mindex,count

cpdef double[:] nonnanarray(double[:] data):
    """ Return array of non-nan values """
    cdef long[:] index
    cdef long n
    index,n = nonnanindexes(data)
    nndata = cvarray(shape=(n,),itemsize=sizeof(double),format="d")
    cdef double[:] mnndata = nndata
    for i in range(n):
        mnndata[i] =  data[index[i]]
    return mnndata

cpdef double nanmedian(double[:] data):
    """  """
    cdef long n
    cdef double result

    nndata = nonnanarray(data)
    n = len(nndata)

    #cdef double *nndataptr = <double *>nndata.data
    # qsort
    #myqsort(nndataptr,n)
    #void qsort(void *base, int nmemb, int size,
    #            int(*compar)(const_void *, const_void *)) nogil

    # return an array of the non-nan values
    # then use qsort() to sort that array in place
    # can get the median value of that array
    #return result


# cpdef double nanmedian(double[:] data):
#     """ Calculate the median of an array checking for nans."""
#     cdef long n = len(data)
#     cdef double result
#     cdef long ngood = 0

#     result = 0.0
#     for i in range(n):
#         if isnan(data[i])==False:
#             ngood += 1
#             result += data[i]
#     if ngood > 0:
#         result = result / ngood
#     else:
#         result = NAN
#     return result

cpdef double mad(double[:] data, int ignore_nan, int zero):
    """ Calculate the median absolute deviation of an array."""
    cdef int ndata
    cdef double[:] resid
    cdef double ref,result    

    ndata = len(data)
    resid = np.zeros(ndata,float)
    resid = cvarray(shape=(),itemsize=sizeof(double),format="d")
    # With median reference point
    if zero==0:
        if ignore_nan==1:
            ref = np.nanmedian(data)
            for i in range(ndata):
                resid[i] = data[i]-ref
            result = np.median(np.abs(resid))
        else:
            ref = np.median(data)
            for i in range(ndata):
                resid[i] = data[i]-ref
            result = np.nanmedian(np.abs(resid))
    # Using zero as reference point
    else:
        if ignore_nan==1:
            result= np.median(np.abs(data))
        else:
            ref = np.median(data)
            result = np.nanmedian(np.abs(data))
    return result * 1.482602218505602

# cpdef double[:] mad2d(double[:,:] data, int axis, int ignore_nan, int zero):
#     """ Calculate the median absolute deviation of an array."""
#     cdef double[:] result,ref
#     cdef int nx,ny
#     cdef double[:,:] ref2d
#     ny = data.shape[0]
#     nx = data.shape[1]
#     if axis==0:
#         ref = np.zeros(nx,float)
#     elif axis==1:
#         ref = np.zeros(ny,float)
#     ref2d = np.zeros((ny,nx),float)
#     resid = np.zeros((ny,nx),float)

#     # With median reference point
#     if zero==0:
#         if ignore_nan==1:
#             ref = np.asarray(np.nanmedian(data,axis=axis))
#             newshape = np.array(data.shape)
#             newshape[axis] = -1
#             newshape = (newshape[0],newshape[1])
# 	    for i in range():
#                 ref2d[i,:] = ref
#             resid = data-ref.reshape(newshape)
#             result = np.nanmedian(np.abs(resid),axis=axis) * 1.482602218505602
#         else:
#             ref = np.asarray(np.median(data,axis=axis))
#             newshape = np.array(data.shape)
#             newshape[axis] = -1
#             newshape = (newshape[0],newshape[1])
#             resid = data-ref.reshape(newshape)
#             result = np.median(np.abs(resid),axis=axis) * 1.482602218505602
#     # Using zero as reference point
#     else:
#         if ignore_nan==1:
#             result = np.nanmedian(np.abs(data),axis=axis) * 1.482602218505602
#         else:
#             result = np.median(np.abs(data),axis=axis) * 1.482602218505602

#     return result

# cpdef double[:,:] mad3d(double[:,:,:] data, int axis, int ignore_nan, int zero):
#     """ Calculate the median absolute deviation of an array."""
#     cdef double[:] ref
#     cdef double[:,:] result

#     nz = data.shape[0]
#     ny = data.shape[1]
#     nx = data.shape[2]
#     if axis==0:
#         ref = np.zeros((ny,nx),float)
#     elif axis==1:
#         ref = np.zeros((nz,nx),float)
#     elif axis==2:
#         ref = np.zeros((nz,ny),float)
#     resid = np.zeros((nz,ny,nx),float)

#     # With median reference point
#     if zero==0:
#         if ignore_nan==1:
#             ref = np.nanmedian(data,axis=axis)
#             newshape = np.array(data.shape)
#             newshape[axis] = -1
#             newshape = (newshape[0],newshape[1],newshape[2])
#             resid = data-ref.reshape(newshape)
#             result = np.nanmedian(np.abs(resid),axis=axis) * 1.482602218505602
#         else:
#             ref = np.median(data,axis=axis)
#             newshape = np.array(data.shape)
#             newshape[axis] = -1
#             newshape = (newshape[0],newshape[1],newshape[2])
#             resid = data-ref.reshape(newshape)
#             result = np.median(np.abs(resid),axis=axis) * 1.482602218505602
#     # Using zero as reference point
#     else:
#         if ignore_nan==1:
#             result = np.nanmedian(np.abs(data),axis=axis) * 1.482602218505602
#         else:
#             result = np.median(np.abs(data),axis=axis) * 1.482602218505602
#     return result

# cpdef double quadratic_bisector(double[:] x, double[:] y):
#     """ Calculate the axis of symmetric or bisector of parabola"""
#     #https://www.azdhs.gov/documents/preparedness/state-laboratory/lab-licensure-certification/technical-resources/
#     #    calibration-training/12-quadratic-least-squares-regression-calib.pdf
#     #quadratic regression statistical equation
#     cdef long n
#     cdef double Sx,Sy,Sxx,Sxy,Sxx2,Sx2y,Sx2x2,denom,a,b
#     n = len(x)
#     if n<3:
#         return np.nan
#     Sx = np.sum(x)
#     Sy = np.sum(y)
#     Sx2 = 0.0
#     for i in range(n):
#         Sx2 += x[i]**2
#     Sxx = 0.0
#     Sxy = 0.0
#     Sxx2 = 0.0
#     Sx2y = 0.0
#     Sx2x2 = 0.0
#     for i in range(n):
#         Sxx += x[i]**2 - Sx/n
#         Sxy += x[i]*y[i] - Sx*Sy/n
#         Sxx2 += x[i]**3 - Sx*Sx2/n
#         Sx2y += x[i]**2 * y[i] - Sx2*Sy/n
#         Sx2x2 += x[i]**4 - Sx2**2/n
#     #Sxx = np.sum(x**2) - np.sum(x)**2/n
#     #Sxy = np.sum(x*y) - np.sum(x)*np.sum(y)/n
#     #Sxx2 = np.sum(x**3) - np.sum(x)*np.sum(x**2)/n
#     #Sx2y = np.sum(x**2 * y) - np.sum(x**2)*np.sum(y)/n
#     #Sx2x2 = np.sum(x**4) - np.sum(x**2)**2/n
#     #a = ( S(x^2*y)*S(xx)-S(xy)*S(xx^2) ) / ( S(xx)*S(x^2x^2) - S(xx^2)^2 )
#     #b = ( S(xy)*S(x^2x^2) - S(x^2y)*S(xx^2) ) / ( S(xx)*S(x^2x^2) - S(xx^2)^2 )
#     denom = Sxx*Sx2x2 - Sxx2**2
#     if denom==0:
#         return np.nan
#     a = ( Sx2y*Sxx - Sxy*Sxx2 ) / denom
#     b = ( Sxy*Sx2x2 - Sx2y*Sxx2 ) / denom
#     if a==0:
#         return np.nan
#     return -b/(2*a)

# cpdef qr_jac_solve(jac,resid,weight=None):
#     """ Solve part of a non-linear least squares equation using QR decomposition
#         using the Jacobian."""
#     # jac: Jacobian matrix, first derivatives, [Npix, Npars]
#     # resid: residuals [Npix]
#     # weight: weights, ~1/error**2 [Npix]
    
#     # QR decomposition
#     if weight is None:
#         q,r = np.linalg.qr(jac)
#         rinv = inverse(r)
#         dbeta = rinv @ (q.T @ resid)
#     # Weights input, multiply resid and jac by weights        
#     else:
#         q,r = np.linalg.qr( jac * weight.reshape(-1,1) )
#         rinv = inverse(r)
#         dbeta = rinv @ (q.T @ (resid*weight))
        
#     return dbeta

# cpdef jac_covariance(jac,resid,wt):
#     """ Determine the covariance matrix. """
    
#     npix,npars = jac.shape
    
#     # Weights
#     #   If weighted least-squares then
#     #   J.T * W * J
#     #   where W = I/sig_i**2
#     if wt is not None:
#         wt2 = wt.reshape(-1,1) + np.zeros(npars)
#         hess = jac.T @ (wt2 * jac)
#     else:
#         hess = jac.T @ jac  # not weighted

#     # cov = H-1, covariance matrix is inverse of Hessian matrix
#     cov_orig = inverse(hess)
    
#     # Rescale to get an unbiased estimate
#     # cov_scaled = cov * (RSS/(m-n)), where m=number of measurements, n=number of parameters
#     # RSS = residual sum of squares
#     #  using rss gives values consistent with what curve_fit returns
#     # Use chi-squared, since we are doing the weighted least-squares and weighted Hessian
#     if wt is not None:
#         chisq = np.sum(resid**2 * wt)
#     else:
#         chisq = np.sum(resid**2)
#     dof = npix-npars
#     if dof<=0:
#         dof = 1
#     cov = cov_orig * (chisq/dof)  # what MPFITFUN suggests, but very small
        
#     return cov


# cpdef double[:] poly2d(double[:,:] xdata, double[:] pars):
#     """ model of 2D linear polynomial."""
#     cdef double[:] x,y,result
#     cdef long n
#     x = xdata[:,0]
#     y = xdata[:,1]
#     n = len(x)
#     result = np.zeros(n,float)
#     for i in range(n):
#         result[i] = pars[0]+pars[1]*x[i]+pars[2]*y[i]+pars[3]*x[i]*y[i]
#     return result

# cpdef list jacpoly2d(double[:,:] xdata, double[:] pars):
#     """ jacobian of 2D linear polynomial."""
#     cdef double[:] x,y,m
#     cdef double[:,:] jac
#     cdef long n
#     x = xdata[:,0]
#     y = xdata[:,1]
#     n = len(x)
#     # Model
#     m = np.zeros(n,float)
#     for i in range(n):
#         m[i] = pars[0]+pars[1]*x[i]+pars[2]*y[i]+pars[3]*x[i]*y[i]
#     # Jacobian, partical derivatives wrt the parameters
#     jac = np.zeros((n,4),float)
#     jac[:,0] = 1    # constant coefficient
#     jac[:,1] = x    # x-coefficient
#     jac[:,2] = y    # y-coefficient
#     for i in range(n):
#         jac[i,3] = x[i]*y[i]  # xy-coefficient
#     return m,jac

# cpdef poly2dfit(x,y,data,error,maxiter=2,minpercdiff=0.5,verbose=False):
#     """ Fit a 2D linear function to data robustly."""
#     ndata = len(data)
#     if ndata<4:
#         raise Exception('Need at least 4 data points for poly2dfit')
#     gd1, = np.where(np.isfinite(data))
#     if len(gd1)<4:
#         raise Exception('Need at least 4 good data points for poly2dfit')
#     xdata = np.zeros((len(gd1),2),float)
#     xdata[:,0] = x[gd1]
#     xdata[:,1] = y[gd1]
#     initpar = np.zeros(4,float)
#     med = np.median(data[gd1])
#     sig = mad(data[gd1])
#     gd2, = np.where( (np.abs(data-med)<3*sig) & np.isfinite(data))
#     if len(gd1)>=4 and len(gd2)<4:
#         gd = gd1
#     else:
#         gd = gd2
#     initpar[0] = med
#     xdata = np.zeros((len(gd),2),float)
#     xdata[:,0] = x[gd]
#     xdata[:,1] = y[gd]
#     data1 = data[gd]
#     error1 = error[gd]

#     # Do the fit
#     # Iterate
#     count = 0
#     bestpar = initpar.copy()
#     maxpercdiff = 1e10
#     # maxsteps = None
#     wt = 1.0/error1.ravel()**2
#     while (count<maxiter and maxpercdiff>minpercdiff):
#         # Use Cholesky, QR or SVD to solve linear system of equations
#         m,j = jacpoly2d(xdata,bestpar)
#         dy = data1.ravel()-m.ravel()
#         # Solve Jacobian
#         #if error is not None:
#         #dbeta = qr_jac_solve(j,dy,weight=wt)
#         dbeta = qr_jac_solve(j,dy)
#         #else:
#         #    dbeta = qr_jac_solve(j,dy)
#         dbeta[~np.isfinite(dbeta)] = 0.0  # deal with NaNs

#         # -add "shift cutting" and "line search" in the least squares method
#         # basically scale the beta vector to find the best match.
#         # check this out
#         # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.line_search.html
        
        
#         # Update parameters
#         oldpar = bestpar.copy()
#         # limit the steps to the maximum step sizes and boundaries
#         #if bounds is not None or maxsteps is not None:
#         #    bestpar = newpars(bestpar,dbeta,bounds,maxsteps)
#         #else:
#         bestpar += dbeta
#         # Check differences and changes
#         diff = np.abs(bestpar-oldpar)
#         denom = np.maximum(np.abs(oldpar.copy()),0.0001)
#         percdiff = diff.copy()/denom*100  # percent differences
#         maxpercdiff = np.max(percdiff)
                
#         if verbose:
#             print('N = ',count)
#             print('bestpars = ',bestpar)
#             print('dbeta = ',dbeta)
                
#         count += 1

#     # Get covariance and errors
#     m,j = jacpoly2d(xdata,bestpar)
#     dy = data1.ravel()-m.ravel()
#     cov = jac_covariance(j,dy,wt)
#     perror = np.sqrt(np.diag(cov))

#     return bestpar,perror,cov


# cpdef crossmatch(X1, X2, max_distance=np.inf,k=1):
#     """Cross-match the values between X1 and X2

#     By default, this uses a KD Tree for speed.

#     Parameters
#     ----------
#     X1 : array_like
#         first dataset, shape(N1, D)
#     X2 : array_like
#         second dataset, shape(N2, D)
#     max_distance : float (optional)
#         maximum radius of search.  If no point is within the given radius,
#         then inf will be returned.

#     Returns
#     -------
#     dist, ind: ndarrays
#         The distance and index of the closest point in X2 to each point in X1
#         Both arrays are length N1.
#         Locations with no match are indicated by
#         dist[i] = inf, ind[i] = N2
#     """
#     #X1 = np.asarray(X1, dtype=float)
#     #X2 = np.asarray(X2, dtype=float)

#     N1, D1 = X1.shape
#     N2, D2 = X2.shape

#     if D1 != D2:
#         raise ValueError('Arrays must have the same second dimension')

#     kdt = KDTree(X2)

#     dist, ind, neigh = kdt.query(X1, k=k, distance_upper_bound=max_distance)

#     return dist, ind


# # from astroML, modified by D. Nidever
# cpdef xmatch(ra1, dec1, ra2, dec2, dcr=2.0, unique=False, sphere=True):
#     """Cross-match angular values between RA1/DEC1 and RA2/DEC2

#     Find the closest match in the second list for each element
#     in the first list and within the maximum distance.

#     By default, this uses a KD Tree for speed.  Because the
#     KD Tree only handles cartesian distances, the angles
#     are projected onto a 3D sphere.

#     This can return duplicate matches if there is an element
#     in the second list that is the closest match to two elements
#     of the first list.

#     Parameters
#     ----------
#     ra1/dec1 : array_like
#         first dataset, arrays of RA and DEC
#         both measured in degrees
#     ra2/dec2 : array_like
#         second dataset, arrays of RA and DEC
#         both measured in degrees
#     dcr : float (optional)
#         maximum radius of search, measured in arcsec.
#         This can be an array of the same size as ra1/dec1.
#     unique : boolean, optional
#         Return unique one-to-one matches.  Default is False and
#            allows duplicates.
#     sphere : boolean, optional
#         The coordinates are spherical in degrees.  Otherwise, the dcr
#           is assumed to be in the same units as the input values.
#           Default is True.


#     Returns
#     -------
#     ind1, ind2, dist: ndarrays
#         The indices for RA1/DEC1 (ind1) and for RA2/DEC2 (ind2) of the
#         matches, and the distances (in arcsec).
#     """
#     n1 = len(ra1)
#     n2 = len(ra2)
#     X1 = np.zeros((n1,2),float)
#     X1[:,0] = ra1
#     X1[:,1] = dec1
#     X2 = np.zeros((n2,2),float)
#     X2[:,0] = ra2
#     X2[:,1] = dec2
    
#     # Spherical coordinates in degrees
#     if sphere:
#         X1 = X1 * (np.pi / 180.)
#         X2 = X2 * (np.pi / 180.)
#         #if utils.size(dcr)>1:
#         #    max_distance = (np.max(dcr) / 3600) * (np.pi / 180.)
#         #else:
#         #    max_distance = (dcr / 3600) * (np.pi / 180.)
#         max_distance = (dcr / 3600) * (np.pi / 180.)
        
#         # Convert 2D RA/DEC to 3D cartesian coordinates
#         Y1 = np.zeros((n1,3),float)
#         Y1[:,0] = np.cos(X1[:, 0]) * np.cos(X1[:, 1])
#         Y1[:,1] = np.sin(X1[:, 0]) * np.cos(X1[:, 1])
#         Y1[:,2] = np.sin(X1[:, 1])
#         #Y1 = np.transpose(np.vstack([np.cos(X1[:, 0]) * np.cos(X1[:, 1]),
#         #                             np.sin(X1[:, 0]) * np.cos(X1[:, 1]),
#         #                             np.sin(X1[:, 1])]))
#         Y2 = np.zeros((n2,3),float)
#         Y2[:,0] = np.cos(X2[:, 0]) * np.cos(X2[:, 1])
#         Y2[:,1] = np.sin(X2[:, 0]) * np.cos(X2[:, 1])
#         Y2[:,2] = np.sin(X2[:, 1])
#         #Y2 = np.transpose(np.vstack([np.cos(X2[:, 0]) * np.cos(X2[:, 1]),
#         #                             np.sin(X2[:, 0]) * np.cos(X2[:, 1]),
#         #                             np.sin(X2[:, 1])]))

#         # law of cosines to compute 3D distance
#         max_y = np.sqrt(2 - 2 * np.cos(max_distance))
#         k = 1 if unique is False else 10
#         dist, ind = crossmatch(Y1, Y2, max_y, k=k)
#         # dist has shape [N1,10] or [N1,1] (if unique)
    
#         # convert distances back to angles using the law of tangents
#         not_infy,not_infx = np.where(~np.isinf(dist))
#         #x = 0.5 * dist[not_infy,not_infx]
#         #dist[not_infy,not_infx] = (180. / np.pi * 2 * np.arctan2(x,
#         #                           np.sqrt(np.maximum(0, 1 - x ** 2))))
#         #dist[not_infy,not_infx] *= 3600.0      # in arcsec
#     # Regular coordinates
#     else:
#         k = 1 if unique is False else 10
#         dist, ind = crossmatch(X1, X2, dcr, k=k)
#         #dist, ind = crossmatch(X1, X2, np.max(dcr), k=k)
#         not_infy,not_infx = np.where(~np.isinf(dist))
        
#     # Allow duplicates
#     if unique==False:

#         # no matches
#         if len(not_infx)==0:
#             return np.array([-1]), np.array([-1]), np.array([np.inf])
        
#         # If DCR is an array then impose the max limits for each element
#         #if utils.size(dcr)>1:
#         #    bd,nbd = utils.where(dist > dcr)
#         #    if nbd>0:
#         #        dist[bd] = np.inf
#         #        not_inf = ~np.isinf(dist)
        
#         # Change to the output that I want
#         # dist is [N1,1] if unique==False
#         ind1 = np.arange(len(ra1))[not_infy]
#         ind2 = ind[not_infy,0]
#         mindist = dist[not_infy,0]
        
#     # Return unique one-to-one matches
#     else:

#         # no matches
#         if np.sum(~np.isinf(dist[:,0]))==0:
#             return np.array([-1]), np.array([-1]), np.array([np.inf])
        
#         done = 0
#         niter = 1
#         # Loop until we converge
#         while (done==0):

#             # If DCR is an array then impose the max limits for each element
#             #if utils.size(dcr)>1:
#             #    bd,nbd = utils.where(dist[:,0] > dcr)
#             #    if nbd>0:
#             #        for i in range(nbd):
#             #            dist[bd[i],:] = np.inf

#             # no matches
#             if np.sum(~np.isinf(dist[:,0]))==0:
#                 return np.array([-1]), np.array([-1]), np.array([np.inf])

#             # closest matches
#             not_inf1 = ~np.isinf(dist[:,0])
#             not_inf1_ind, = np.where(~np.isinf(dist[:,0]))
#             ind1 = np.arange(len(ra1))[not_inf1]  # index into original ra1/dec1 arrays
#             ind2 = ind[:,0][not_inf1]             # index into original ra2/dec2 arrays
#             mindist = dist[:,0][not_inf1]
#             if len(ind2)==0:
#                 return np.array([-1]), np.array([-1]), np.array([np.inf])
#             find2 = np.zeros(len(ind2),float)
#             find2[:] = ind2
#             index = Index(find2)
#             # some duplicates to deal with
#             bd, = np.where(index.num>1)
#             nbd = len(bd)
#             if nbd>0:
#                 ntorem = 0
#                 for i in range(nbd):
#                     ntorem += index.num[bd[i]]-1
#                 torem = np.zeros(ntorem,np.int32)  # index into shortened ind1/ind2/mindist
#                 trcount = 0
#                 for i in range(nbd):
#                     # index into shortened ind1/ind2/mindist
#                     indx = index.getindex(bd[i])
#                     #indx = index['index'][index['lo'][bd[i]]:index['hi'][bd[i]]+1]
#                     # keep the one with the smallest minimum distance
#                     si = np.argsort(mindist[indx])
#                     if index.num[bd[i]]>2:
#                         bad = indx[si[1:]]
#                         torem[trcount:trcount+len(bad)] = bad    # add
#                         trcount += len(bad)
#                     else:
#                         torem[trcount:trcount+1] = indx[si[1:]][0]  # add single element
#                 #ntorem = utils.size(torem)
#                 torem_orig_index = not_inf1_ind[torem]  # index into original ind/dist arrays
#                 # For each object that was "removed" and is now unmatched, check the next possible
#                 # match and move it up in the dist/ind list if it isn't INF
#                 for i in range(ntorem):
#                     # There is a next possible match 
#                     if ~np.isinf(dist[torem_orig_index[i],niter-1]):
#                         temp = np.zeros(10,np.int64)
#                         temp[niter:] = ind[torem_orig_index[i],niter:]  #.squeeze()
#                         temp[-niter:] = np.zeros(niter,np.int64)-1
#                         ind[torem_orig_index[i],:] = temp
#                         temp2 = np.zeros(10,float)
#                         temp2[niter:] = dist[torem_orig_index[i],niter:]   #.squeeze()
#                         temp2[-niter:] = np.zeros(niter,float)+np.inf
#                         dist[torem_orig_index[i],:] = temp2
#                         #ind[torem_orig_index[i],:] = np.hstack( (ind[torem_orig_index[i],niter:].squeeze(),
#                         #                                         np.repeat(-1,niter)) )
#                         #dist[torem_orig_index[i],:] = np.hstack( (dist[torem_orig_index[i],niter:].squeeze(),
#                         #                                          np.repeat(np.inf,niter)) )
#                     # All INFs
#                     else:
#                         ind[torem_orig_index[i],:] = -1
#                         dist[torem_orig_index[i],:] = np.inf
#                         # in the next iteration these will be ignored
#             else:
#                 ntorem = 0

#             niter += 1
#             # Are we done, no duplicates or hit the maximum 10
#             if (ntorem==0) or (niter>=10): done=1
                                
#     return ind1, ind2, mindist


# cpdef skygrid(im,binsize,tot=0,med=1):
#     """ Estimate the background."""
#     #binsize = 200
#     ny,nx = im.shape
#     ny2 = ny // binsize
#     nx2 = nx // binsize
#     bgim = np.zeros((ny2,nx2),float)
#     nsample = np.minimum(1000,binsize*binsize)
#     sample = np.random.randint(0,binsize*binsize-1,nsample)
#     for i in range(nx2):
#         for j in range(ny2):
#             x1 = i*binsize
#             x2 = x1+binsize
#             if x2 > nx: x2=nx
#             y1 = j*binsize
#             y2 = y1+binsize
#             if y2 > ny: y2=ny
#             if tot==1:
#                 bgim[j,i] = np.sum(im[y1:y2,x1:x2].ravel()[sample])
#             elif med==1:
#                 bgim[j,i] = np.median(im[y1:y2,x1:x2].ravel()[sample])
#             else:
#                 bgim[j,i] = np.mean(im[y1:y2,x1:x2].ravel()[sample])
#     return bgim

# cpdef skyinterp(binim,fullim,binsize):
#     """ linear interpolation"""
#     ny,nx = fullim.shape
#     ny2 = ny // binsize
#     nx2 = nx // binsize
#     hbinsize = int(0.5*binsize)

#     # Calculate midpoint positions
#     xx = np.arange(nx2)*binsize+hbinsize
#     yy = np.arange(ny2)*binsize+hbinsize
    
#     for i in range(ny2-1):
#         for j in range(nx2-1):
#             y1 = i*binsize+hbinsize
#             y2 = y1+binsize
#             x1 = j*binsize+hbinsize
#             x2 = x1+binsize
#             f11 = binim[i,j]
#             f12 = binim[i+1,j]
#             f21 = binim[i,j+1]
#             f22 = binim[i+1,j+1]
#             denom = binsize*binsize
#             for k in range(binsize):
#                 for l in range(binsize):
#                     x = x1+k
#                     y = y1+l
#                     # weighted mean
#                     #denom = (x2-x1)*(y2-y1)
#                     w11 = (x2-x)*(y2-y)/denom
#                     w12 = (x2-x)*(y-y1)/denom
#                     w21 = (x-x1)*(y2-y)/denom
#                     w22 = (x-x1)*(y-y1)/denom
#                     f = w11*f11+w12*f12+w21*f21+w22*f22
#                     fullim[y,x] = f

#     # Do the edges
#     for i in range(hbinsize,nx2*binsize-hbinsize):
#         # bottom
#         for j in range(hbinsize):
#             fullim[j,i] = fullim[hbinsize,i]
#         # top
#         for j in range(ny2*binsize-hbinsize-1,ny):
#             fullim[j,i] = fullim[ny2*binsize-hbinsize-2,i]
#     for j in np.arange(hbinsize,ny2*binsize-hbinsize):
#         # left
#         for i in range(hbinsize):
#             fullim[j,i] = fullim[j,hbinsize]
#         # right
#         for i in range(nx2*binsize-hbinsize-1,nx):
#             fullim[j,i] = fullim[j,nx2*binsize-hbinsize-2]
#     # Do the corners
#     for i in range(hbinsize):
#         # bottom-left
#         for j in range(hbinsize):
#             fullim[j,i] = fullim[hbinsize,hbinsize]
#         # top-left
#         for j in range(ny2*binsize-hbinsize-1,ny):
#             fullim[j,i] = fullim[ny2*binsize-hbinsize-2,hbinsize]
#     for i in range(nx2*binsize-hbinsize-1,nx):
#         # bottom-right
#         for j in range(hbinsize):
#             fullim[j,i] = fullim[hbinsize,nx2*binsize-hbinsize-2]
#         # top-right
#         for j in range(ny2*binsize-hbinsize-1,ny):
#             fullim[j,i] = fullim[ny2*binsize-hbinsize-2,nx2*binsize-hbinsize-2]

#     return fullim


# cpdef sky(im,binsize=0):
#     ny,nx = im.shape

#     # Figure out best binsize
#     if binsize <= 0:
#         binsize = np.min(np.array([ny//20,nx//20]))
#     binsize = np.maximum(binsize,20)
#     ny2 = ny // binsize
#     nx2 = nx // binsize

#     # Bin in a grid
#     bgimbin = skygrid(im,binsize,0,1)

#     # Outlier rejection
#     # median smoothing
#     medbin = 3
#     if nx2<medbin or ny2<medbin:
#         medbin = np.min(np.array([nx2,ny2]))
#     smbgimbin = smooth2d(bgimbin,medbin,1)  # median smoothing
    
#     # Linearly interpolate
#     bgim = np.zeros(im.shape,np.float64)+np.median(bgimbin)
#     bgim = skyinterp(smbgimbin,bgim,binsize)

#     return bgim

# cpdef detection(im,nsig=10):
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

# cpdef detectpeaks(im,sig,nsig):
#     """ Detect peaks"""
#     # input sky subtracted image
#     ny,nx = im.shape
#     nbin = 3  # 5
#     nhbin = nbin//2

#     mnim = np.zeros(im.shape,float)-100000
    
#     count = 0
#     xpeak = np.zeros(100000,float)
#     ypeak = np.zeros(100000,float)
#     for i in np.arange(nhbin+1,nx-nhbin-2):
#         for j in range(nhbin+1,ny-nhbin-2):
#             if im[j,i]>nsig*sig:
#                 if mnim[j,i] > -1000:
#                     mval = mnim[j,i]
#                 else:
#                     mval = np.mean(im[j-nhbin:j+nhbin+1,i-nhbin:i+nhbin+1])
#                     mnim[j,i] = mval
#                 if mnim[j,i-1] > -1000:
#                     lval = mnim[j,i-1]
#                 else:
#                     lval = np.mean(im[j-nhbin:j+nhbin+1,i-nhbin-1:i+nhbin])
#                     mnim[j,i-1] = lval
#                 if mnim[j,i+1] > -1000:
#                     rval = mnim[j,i+1]
#                 else:
#                     rval = np.mean(im[j-nhbin:j+nhbin+1,i-nhbin+1:i+nhbin+2])
#                     mnim[j,i+1] = rval
#                 if mnim[j-1,i] > -1000:
#                     dval = mnim[j-1,i]
#                 else:
#                     dval = np.mean(im[j-nhbin-1:j+nhbin,i-nhbin:i+nhbin+1])
#                     mnim[j-1,i] = dval
#                 if mnim[j+1,i] > -1000:
#                     uval = mnim[j+1,i]
#                 else:
#                     uval = np.mean(im[j-nhbin+1:j+nhbin+2,i-nhbin:i+nhbin+1])
#                     mnim[j+1,i] = uval
#                 # Check that it is a peak
#                 if (mval>lval and mval>rval and mval>dval and mval>uval):
#                     xpeak[count] = i
#                     ypeak[count] = j
#                     count = count + 1
#     return xpeak,ypeak,count

# cpdef boundingbox(im,xp,yp,thresh,bmax):
#     """ Get bounding box for the source """

#     ny,nx = im.shape
#     nbin = 3
#     nhbin = nbin//2
    
#     # Step left until you reach the threshold
#     y0 = yp-nhbin
#     if y0<0: y0=0
#     y1 = yp+nhbin+1
#     if y1>ny: y1=ny
#     flag = False
#     midval = np.mean(im[y0:y1,xp])
#     count = 1
#     while (flag==False):
#         newval = np.mean(im[y0:y1,xp-count])
#         if newval < thresh*midval or xp-count==0 or count==bmax:
#             flag = True
#         lastval = newval
#         count += 1
#     leftxp = xp-count+1
#     # Step right until you reach the threshold
#     flag = False
#     count = 1
#     while (flag==False):
#         newval = np.mean(im[y0:y1,xp+count])
#         if newval < thresh*midval or xp+count==(nx-1) or count==bmax:
#             flag = True
#         lastval = newval
#         count += 1
#     rightxp = xp+count-1
#     # Step down until you reach the threshold
#     x0 = xp-nhbin
#     if x0<0: x0=0
#     x1 = xp+nhbin+1
#     if x1>nx: x1=nx
#     flag = False
#     midval = np.mean(im[yp,x0:x1])
#     count = 1
#     while (flag==False):
#         newval = np.mean(im[yp-count,x0:x1])
#         if newval < thresh*midval or yp-count==0 or count==bmax:
#             flag = True
#         lastval = newval
#         count += 1
#     downyp = yp-count+1
#     # Step up until you reach the threshold
#     flag = False
#     count = 1
#     while (flag==False):
#         newval = np.mean(im[yp+count,x0:x1])
#         if newval < thresh*midval or yp+count==(ny-1) or count==bmax:
#             flag = True
#         lastval = newval
#         count += 1
#     upyp = yp+count-1

#     return leftxp,rightxp,downyp,upyp


# cpdef morpho(im,xp,yp,x0,x1,y0,y1,thresh):
#     """ Measure morphology parameters """
#     ny,nx = im.shape

#     midval = im[yp,xp]
#     hthresh = thresh*midval
    
#     nx = x1-x0+1
#     ny = y1-y0+1

#     # Flux and first moments
#     flux = 0.0
#     mnx = 0.0
#     mny = 0.0
#     # X loop    
#     for i in range(nx):
#         x = i+x0
#         # Y loop
#         for j in range(ny):
#             y = j+y0
#             val = im[y,x]
#             if val>hthresh:
#                 # Flux
#                 flux += val
#                 # First moments
#                 mnx += val*x
#                 mny += val*y
#     mnx /= flux
#     mny /= flux

#     # Second moments
#     sigx2 = 0.0
#     sigy2 = 0.0
#     sigxy = 0.0
#     # X loop    
#     for i in range(nx):
#         x = i+x0
#         # Y loop
#         for j in range(ny):
#             y = j+y0
#             val = im[y,x]
#             if val>hthresh:
#                 sigx2 += val*(x-mnx)**2
#                 sigy2 += val*(y-mny)**2
#                 sigxy += val*(x-mnx)*(y-mny)
#     sigx2 /= flux
#     sigy2 /= flux
#     sigx = np.sqrt(sigx2)
#     sigy = np.sqrt(sigy2)
#     sigxy /= flux
#     fwhm = (sigx+sigy)*0.5 * 2.35

#     # Ellipse parameters
#     asemi = np.sqrt( 0.5*(sigx2+sigy2) + np.sqrt(((sigx2-sigy2)*0.5)**2 + sigxy**2 ) )
#     bsemi = np.sqrt( 0.5*(sigx2+sigy2) - np.sqrt(((sigx2-sigy2)*0.5)**2 + sigxy**2 ) )
#     theta = 0.5*np.arctan2(2*sigxy,sigx2-sigy2)  # in radians

#     return flux,mnx,mny,sigx,sigy,sigxy,fwhm,asemi,bsemi,theta

# cpdef morphology(im,xpeak,ypeak,thresh,bmax):
#     """ Measure morphology of the peaks."""

#     ny,nx = im.shape
#     nbin = 3
#     nhbin = nbin//2

#     mout = np.zeros((len(xpeak),17),float)
#     for i in range(len(xpeak)):
#         xp = int(xpeak[i])
#         yp = int(ypeak[i])
#         mout[i,0] = xp
#         mout[i,1] = yp
        
#         # Get the bounding box
#         leftxp,rightxp,downyp,upyp = boundingbox(im,xp,yp,thresh,bmax)
#         mout[i,2] = leftxp
#         mout[i,3] = rightxp
#         mout[i,4] = downyp
#         mout[i,5] = upyp
#         mout[i,6] = (rightxp-leftxp+1)*(upyp-downyp+1)

#         # Measure morphology parameters
#         out = morpho(im,xp,yp,leftxp,rightxp,downyp,upyp,thresh)
#         #flux,mnx,mny,sigx,sigy,sigxy,fwhm,asemi,bsemi,theta = out
#         mout[i,7:] = out

#     return mout

# cpdef starbbox(coords,imshape,radius):
#     """                                                                                        
#      Return the boundary box for a star given radius and image size.                            
                                                                                                
#     Parameters                                                                                 
#     ----------                                                                                 
#     coords: list or tuple                                                                      
#        Central coordinates (xcen,ycen) of star (*absolute* values).                            
#     imshape: list or tuple                                                                     
#        Image shape (ny,nx) values.  Python images are (Y,X).                                   
#     radius: float                                                                              
#        Radius in pixels.  
 
#     Returns                                                                                     
#     -------                                                                                     
#     bbox : BoundingBox object                                                                   
#        Bounding box of the x/y ranges.                                                          
#        Upper values are EXCLUSIVE following the python convention.                              
                                                                                                
#     """

#     # pixels span +/-0.5 pixels in each direction
#     # so pixel x=5 spans x=4.5-5.5
#     # Use round() to get the value of the pixel that is covers
#     # the coordinate
#     # to include pixels at the bottom but not the top we have to
#     # subtract a tiny bit before we round
#     eta = 1e-5

#     # Star coordinates
#     xcen,ycen = coords
#     ny,nx = imshape   # python images are (Y,X)
#     x0 = xcen-radius
#     x1 = xcen+radius
#     y0 = ycen-radius
#     y1 = ycen+radius
#     xlo = np.maximum(int(np.round(x0-eta)),0)
#     xhi = np.minimum(int(np.round(x1-eta))+1,nx)
#     ylo = np.maximum(int(np.round(y0-eta)),0)
#     yhi = np.minimum(int(np.round(y1-eta))+1,ny)
#     # add 1 at the upper end because those values are EXCLUDED
#     # by the standard python convention

#     # The old way of doing it
#     #xlo = np.maximum(int(np.floor(xcen-radius)),0)
#     #xhi = np.minimum(int(np.ceil(xcen+radius+1)),nx)
#     #ylo = np.maximum(int(np.floor(ycen-radius)),0)
#     #yhi = np.minimum(int(np.ceil(ycen+radius+1)),ny)
#     return np.array([xlo,xhi,ylo,yhi])


# cpdef getstar(imshape,xcen,ycen,hpsfnpix,fitradius):
#     """ Return a star's full footprint and fitted pixels data."""
#     # always return the same size
#     # a distance of 6.2 pixels spans 6 full pixels but you could have
#     # 0.1 left on one side and 0.1 left on the other side
#     # that's why we have to add 2 pixels
#     nfpix = hpsfnpix*2+1
#     fbbox = starbbox((xcen,ycen),imshape,hpsfnpix)
#     nfx = fbbox[1]-fbbox[0]
#     nfy = fbbox[3]-fbbox[2]
#     # extra buffer is ALWAYS at the end of each dimension    
#     fxdata = np.zeros(nfpix*nfpix,np.int64)-1
#     fydata = np.zeros(nfpix*nfpix,np.int64)-1
#     fravelindex = np.zeros(nfpix*nfpix,np.int64)-1
#     fcount = 0
#     # Fitting pixels
#     npix = int(np.floor(2*fitradius))+2
#     bbox = starbbox((xcen,ycen),imshape,fitradius)
#     nx = bbox[1]-bbox[0]
#     ny = bbox[3]-bbox[2]
#     xdata = np.zeros(npix*npix,np.int64)-1
#     ydata = np.zeros(npix*npix,np.int64)-1
#     ravelindex = np.zeros(npix*npix,np.int64)-1
#     mask = np.zeros(npix*npix,np.int8)
#     fcount = 0
#     count = 0
#     for j in range(nfpix):
#         y = j + fbbox[2]
#         for i in range(nfpix):
#             x = i + fbbox[0]
#             r = np.sqrt((x-xcen)**2 + (y-ycen)**2)
#             if x>=fbbox[0] and x<=fbbox[1]-1 and y>=fbbox[2] and y<=fbbox[3]-1:
#                 if r <= 1.0*hpsfnpix:
#                     fxdata[fcount] = x
#                     fydata[fcount] = y
#                     fmulti_index = (np.array([y]),np.array([x]))
#                     fravelindex[fcount] = ravel_multi_index(fmulti_index,imshape)[0]
#                     fcount += 1
#                 if r <= fitradius:
#                     xdata[count] = x
#                     ydata[count] = y
#                     multi_index = (np.array([y]),np.array([x]))
#                     ravelindex[count] = ravel_multi_index(multi_index,imshape)[0]
#                     mask[count] = 1
#                     count += 1
#     return (fxdata,fydata,fravelindex,fbbox,nfx,nfy,fcount,
#             xdata,ydata,ravelindex,bbox,count,mask)


# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3

import cython
cimport cython
import numpy as np
cimport numpy as np
from cython.view cimport array as cvarray
#from scipy.special import gamma, gammaincinv, gammainc

from libc.math cimport exp,sqrt,atan2,pi,NAN,floor,pow
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

# cdef int mycmp(const_void * pa, const_void * pb) noexcept:
#     cdef double a = (<double *>pa)[0]
#     cdef double b = (<double *>pb)[0]
#     if a < b:
#         return -1
#     elif a > b:
#         return 1
#     else:
#         return 0

# cdef void myqsort(double * y, ssize_t l) nogil:
#     qsort(y, l, sizeof(double), mycmp)

cpdef double[:,:] transpose(double[:,:] A):
    """ Transpose a 2D array """
    cdef Py_ssize_t nx,ny
    cdef double[:,:] B
    ny = A.shape[0]
    nx = A.shape[1]
    B = np.zeros((nx,ny),np.float64)
    for i in range(ny):
        for j in range(nx):
            B[j,i] = A[i,j]
    return B
    
cpdef double[:,:] matmult(double[:,:] A, double[:,:] B):
    """
    Multiply two matrices A and B into C:  C = A @ B
    A: m x k
    B: k x n
    C: m x n (output, must be pre-allocated)
    """
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t m = A.shape[0]
    cdef Py_ssize_t n = B.shape[1]
    cdef Py_ssize_t K = A.shape[1]
    cdef double s
    cdef double[:,:] C
    C = np.zeros((m,n),np.float64)
    
    for i in range(m):
        for j in range(n):
            s = 0.0
            for k in range(K):
                s += A[i,k] * B[k,j]
            C[i,j] = s
    return C

cpdef double[:] sub1d(double[:] arr1,double[:] arr2):
    """ Subtract two 1D arrays"""
    cdef Py_ssize_t n,i
    cdef double[:] out
    n = len(arr1)
    out = np.zeros(n,np.float64)
    for i in range(n):
        out[i] = arr1[i]-arr2[i]
    return out

cpdef double[:] add1d(double[:] arr1,double[:] arr2):
    """ Add two 1D arrays"""
    cdef Py_ssize_t n,i
    cdef double[:] out
    n = len(arr1)
    out = np.zeros(n,np.float64)
    for i in range(n):
        out[i] = arr1[i]+arr2[i]
    return out

cpdef double[:] mult1d(double[:] arr1,double[:] arr2):
    """ Multiply two 1D arrays"""
    cdef Py_ssize_t n,i
    cdef double[:] out
    n = len(arr1)
    out = np.zeros(n,np.float64)
    for i in range(n):
        out[i] = arr1[i]*arr2[i]
    return out

cpdef double[:] pow1d(double[:] arr,double power):
    """ Take 1D array to a power """
    cdef Py_ssize_t n,i
    cdef double[:] out
    n = len(arr)
    out = np.zeros(n,np.float64)
    for i in range(n):
        out[i] = pow(arr[i],power)
    return out

cpdef double[:] fact1d(double[:] arr,double factor):
    """ Multiply 1D array by a factor """
    cdef Py_ssize_t n,i
    cdef double[:] out
    n = len(arr)
    out = np.zeros(n,np.float64)
    for i in range(n):
        out[i] = arr[i] * factor
    return out

cpdef double[:,:] sub2d(double[:,:] arr1,double[:,:] arr2):
    """ Subtract two 2D arrays"""
    cdef Py_ssize_t nx,ny,i,j
    cdef double[:,:] out
    ny = arr1.shape[0]
    nx = arr1.shape[1]
    out = np.zeros((ny,nx),np.float64)
    for i in range(ny):
        for j in range(nx):
            out[i,j] = arr1[i,j]-arr2[i,j]
    return out

cpdef double[:,:] add2d(double[:,:] arr1,double[:,:] arr2):
    """ Add two 2D arrays"""
    cdef Py_ssize_t nx,ny,i,j
    cdef double[:,:] out
    ny = arr1.shape[0]
    nx = arr1.shape[1]
    out = np.zeros((ny,nx),np.float64)
    for i in range(ny):
        for j in range(nx):
            out[i,j] = arr1[i,j]+arr2[i,j]
    return out

cpdef double[:,:] mult2d(double[:,:] arr1,double[:,:] arr2):
    """ Multiply two 2D arrays"""
    cdef Py_ssize_t nx,ny,i,j
    cdef double[:,:] out
    ny = arr1.shape[0]
    nx = arr1.shape[1]
    out = np.zeros((ny,nx),np.float64)
    for i in range(ny):
        for j in range(nx):
            out[i,j] = arr1[i,j]*arr2[i,j]
    return out

cpdef double[:,:] pow2d(double[:,:] arr,double power):
    """ Take a 2D array to a power """
    cdef Py_ssize_t nx,ny,i,j
    cdef double[:,:] out
    ny = arr.shape[0]
    nx = arr.shape[1]
    out = np.zeros((ny,nx),np.float64)
    for i in range(ny):
        for j in range(nx):
            out[i,j] = pow(arr[i,j],power)
    return out

cpdef double[:,:] fact2d(double[:,:] arr,double factor):
    """ Multiply 2D array by a factor """
    cdef Py_ssize_t nx,ny,i,j
    cdef double[:,:] out
    ny = arr.shape[0]
    nx = arr.shape[1]
    out = np.zeros((ny,nx),np.float64)
    for i in range(ny):
        for j in range(nx):
            out[i,j] = arr[i,j] * factor
    return out

cpdef double[:,:,:] sub3d(double[:,:,:] arr1,double[:,:,:] arr2):
    """ Subtract two 3D arrays"""
    cdef Py_ssize_t nx,ny,nz,i,j,k
    cdef double[:,:,:] out
    nz = arr1.shape[0]
    ny = arr1.shape[1]
    nx = arr1.shape[2]
    out = np.zeros((nz,ny,nx),np.float64)
    for i in range(nz):
        for j in range(ny):
            for k in range(nx):
                out[i,j,k] = arr1[i,j,k]-arr2[i,j,k]
    return out

cpdef double[:,:,:] add3d(double[:,:,:] arr1,double[:,:,:] arr2):
    """ Add two 3D arrays"""
    cdef Py_ssize_t nx,ny,nz,i,j,k
    cdef double[:,:,:] out
    nz = arr1.shape[0]
    ny = arr1.shape[1]
    nx = arr1.shape[2]
    out = np.zeros((nz,ny,nx),np.float64)
    for i in range(nz):
        for j in range(ny):
            for k in range(nx):
                out[i,j,k] = arr1[i,j,k]+arr2[i,j,k]
    return out

cpdef double[:,:,:] mult3d(double[:,:,:] arr1,double[:,:,:] arr2):
    """ Multiply two 3D arrays"""
    cdef Py_ssize_t nx,ny,nz,i,j,k
    cdef double[:,:,:] out
    nz = arr1.shape[0]
    ny = arr1.shape[1]
    nx = arr1.shape[2]
    out = np.zeros((nz,ny,nx),np.float64)
    for i in range(nz):
        for j in range(ny):
            for k in range(nx):
                out[i,j,k] = arr1[i,j,k]*arr2[i,j,k]
    return out

cpdef double[:,:,:] pow3d(double[:,:,:] arr,double power):
    """ Take a 3D array to a power """
    cdef Py_ssize_t nx,ny,nz,i,j,k
    cdef double[:,:,:] out
    nz = arr.shape[0]
    ny = arr.shape[1]
    nx = arr.shape[2]
    out = np.zeros((nz,ny,nx),np.float64)
    for i in range(nz):
        for j in range(ny):
            for k in range(nx):
                out[i,j,k] = pow(arr[i,j,k],power)
    return out

cpdef double[:,:,:] fact3d(double[:,:,:] arr,double factor):
    """ Multiply 3D array by a factor """
    cdef Py_ssize_t nx,ny,nz,i,j,k
    cdef double[:,:,:] out
    nz = arr.shape[0]
    ny = arr.shape[1]
    nx = arr.shape[2]
    out = np.zeros((nz,ny,nx),np.float64)
    for i in range(nz):
        for j in range(ny):
            for k in range(nx):
                out[i,j,k] = arr[i,j,k] * factor
    return out

cpdef double[:] ravel2d(double[:,:] datain):
    """ Ravel or flatten 2D data to 1D """
    # Use C-type indexing
    # with the last axis index changing fastest, back to the first
    # axis index changing slowest.
    cdef int i,j,idx
    cdef Py_ssize_t nx,ny
    ny = datain.shape[0]
    nx = datain.shape[1]
    npix = nx*ny
    dataout = np.zeros(npix,np.float64)
    idx = 0
    for i in range(ny):
        for j in range(nx):
            dataout[idx] = datain[i,j]
            idx += 1
    return dataout

cpdef double[:] ravel3d(double[:,:,:] datain):
    """ Ravel or flatten 2D data to 1D """
    # Use C-type indexing
    # with the last axis index changing fastest, back to the first
    # axis index changing slowest.
    cdef int i,j,k,idx
    cdef Py_ssize_t nx,ny,nz
    nz = datain.shape[0]
    ny = datain.shape[1]
    nx = datain.shape[2]
    npix = nx*ny*nz
    dataout = np.zeros(npix,np.float64)
    idx = 0
    for i in range(nz):
        for j in range(ny):
            for k in range(nx):
                dataout[idx] = datain[i,j,k]
                idx += 1
    return dataout

    
cpdef double[:,:] reshape2d(double[:,:] datain, Py_ssize_t[:] shapeout):
    """ Reshape a 2D double array """
    # 2D -> 2D
    cdef Py_ssize_t nxout,nyout
    cdef long i,j,idx
    cdef double[:] datain1d
    cdef double[:,:] dataout

    # Reshape reading the index order using C-like index order
    # with the last axis index changing fastest, back to the first
    # axis index changing slowest.

    # You can think of reshaping as first raveling the array (using the given
    # index order), then inserting the elements from the raveled array into the
    # new array using the same kind of index ordering as was used for the
    # raveling.

    # Step 1: ravel the input 2D array to 1D
    datain1d = ravel2d(datain)

    # Step 2: insert the elements into the new array using
    #           the same index ordering
    nyout = shapeout[0]
    nxout = shapeout[1]
    dataout = np.zeros((nyout,nxout),np.float64)
    idx = 0
    for i in range(nyout):
        for j in range(nxout):
            dataout[i,j] = datain1d[idx]
            idx += 1
    return dataout

cpdef double[:,:,:] reshape3d(double[:,:,:] datain, Py_ssize_t[:] shapeout):
    """ Reshape a 3D double array """
    # 3D -> 3D
    cdef Py_ssize_t nxout,nyout,nzout
    cdef long i,j,k,idx
    cdef double[:] datain1d
    cdef double[:,:,:] dataout

    # Reshape reading the index order using C-like index order
    # with the last axis index changing fastest, back to the first
    # axis index changing slowest.

    # You can think of reshaping as first raveling the array (using the given
    # index order), then inserting the elements from the raveled array into the
    # new array using the same kind of index ordering as was used for the
    # raveling.

    # Step 1: ravel the input 2D array to 1D
    datain1d = ravel3d(datain)

    # Step 2: insert the elements into the new array using
    #           the same index ordering
    nzout = shapeout[0]
    nyout = shapeout[1]
    nxout = shapeout[2]
    dataout = np.zeros((nzout,nyout,nxout),np.float64)
    idx = 0
    for i in range(nzout):
        for j in range(nyout):
            for k in range(nxout):
                dataout[i,j,k] = datain1d[idx]
                idx += 1
    return dataout

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
    #f = np.zeros(npix,np.float64)
    f = cvarray(shape=(100,),itemsize=sizeof(double),format="d")
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
    resid = np.zeros(ndata,np.float64)
    resid = cvarray(shape=(ndata,),itemsize=sizeof(double),format="d")
    # With median reference point
    if zero==0:
        if ignore_nan==1:
            ref = np.nanmedian(data)
            for i in range(ndata):
                resid[i] = data[i]-ref
            result = np.nanmedian(np.abs(resid))
        else:
            ref = np.median(data)
            for i in range(ndata):
                resid[i] = data[i]-ref
            result = np.median(np.abs(resid))
    # Using zero as reference point
    else:
        if ignore_nan==1:
            result = np.nanmedian(np.abs(data))
        else:
            ref = np.median(data)
            result = np.median(np.abs(data))
    return result * 1.482602218505602

cpdef double[:] mad2d(double[:,:] data, int axis, int ignore_nan, int zero):
    """ Calculate the median absolute deviation of an array."""
    cdef Py_ssize_t nx,ny
    cdef int i
    cdef double[:] result,ref
    cdef double[:,:] ref2d
    ny = data.shape[0]
    nx = data.shape[1]
    #if axis==0:
    #    ref = np.zeros(nx,np.float64)
    #elif axis==1:
    #    ref = np.zeros(ny,np.float64)
    ref2d = np.zeros((ny,nx),np.float64)
    resid = np.zeros((ny,nx),np.float64)

    # With median reference point
    if zero==0:
        if ignore_nan==1:
            ref = np.asarray(np.nanmedian(data,axis=axis))
            if axis==0:
                for i in range(ny):
                    ref2d[i,:] = ref
            else:
                for i in range(nx):
                    ref2d[:,i] = ref
            resid = sub2d(data,ref2d)
            result = np.nanmedian(np.abs(resid),axis=axis) * 1.482602218505602
        else:
            ref = np.asarray(np.median(data,axis=axis))
            if axis==0:
                for i in range(ny):
                    ref2d[i,:] = ref
            else:
                for i in range(nx):
                    ref2d[:,i] = ref
            resid = sub2d(data,ref2d)
            result = np.median(np.abs(resid),axis=axis) * 1.482602218505602
    # Using zero as reference point
    else:
        if ignore_nan==1:
            result = np.nanmedian(np.abs(data),axis=axis) * 1.482602218505602
        else:
            result = np.median(np.abs(data),axis=axis) * 1.482602218505602

    return result

cpdef double[:,:] mad3d(double[:,:,:] data, int axis, int ignore_nan, int zero):
    """ Calculate the median absolute deviation of an array."""
    cdef double[:,:] ref
    cdef double[:,:] result

    nz = data.shape[0]
    ny = data.shape[1]
    nx = data.shape[2]
    #if axis==0:
    #    ref = np.zeros((ny,nx),np.float64)
    #elif axis==1:
    #    ref = np.zeros((nz,nx),np.float64)
    #elif axis==2:
    #    ref = np.zeros((nz,ny),np.float64)
    ref3d = np.zeros((nz,ny,nx),np.float64)
    resid = np.zeros((nz,ny,nx),np.float64)

    # With median reference point
    if zero==0:
        if ignore_nan==1:
            ref = np.nanmedian(data,axis=axis)
            if axis==0:
                for i in range(nz):
                    ref3d[i,:,:] = ref
            elif axis==1:
                for i in range(ny):
                    ref3d[:,i,:] = ref
            else:
                for i in range(nx):
                    ref3d[:,:,i] = ref
            resid = sub3d(data,ref3d)
            result = np.nanmedian(np.abs(resid),axis=axis) * 1.482602218505602
        else:
            ref = np.median(data,axis=axis)
            if axis==0:
                for i in range(nz):
                    ref3d[i,:,:] = ref
            elif axis==1:
                for i in range(ny):
                    ref3d[:,i,:] = ref
            else:
                for i in range(nx):
                    ref3d[:,:,i] = ref
            resid = sub3d(data,ref3d)
            result = np.median(np.abs(resid),axis=axis) * 1.482602218505602
    # Using zero as reference point
    else:
        if ignore_nan==1:
            result = np.nanmedian(np.abs(data),axis=axis) * 1.482602218505602
        else:
            result = np.median(np.abs(data),axis=axis) * 1.482602218505602
    return result

cpdef double quadratic_bisector(double[:] x, double[:] y):
    """ Calculate the axis of symmetric or bisector of parabola"""
    #https://www.azdhs.gov/documents/preparedness/state-laboratory/lab-licensure-certification/technical-resources/
    #    calibration-training/12-quadratic-least-squares-regression-calib.pdf
    #quadratic regression statistical equation
    cdef long n
    cdef double Sx,Sy,Sxx,Sxy,Sxx2,Sx2y,Sx2x2,denom,a,b
    n = len(x)
    if n<3:
        return np.nan
    Sx = np.sum(x)
    Sy = np.sum(y)
    Sx2 = 0.0
    for i in range(n):
        Sx2 += x[i]**2
    Sxx = 0.0
    Sxy = 0.0
    Sxx2 = 0.0
    Sx2y = 0.0
    Sx2x2 = 0.0
    for i in range(n):
        Sxx += x[i]**2 - Sx/n
        Sxy += x[i]*y[i] - Sx*Sy/n
        Sxx2 += x[i]**3 - Sx*Sx2/n
        Sx2y += x[i]**2 * y[i] - Sx2*Sy/n
        Sx2x2 += x[i]**4 - Sx2**2/n
    #Sxx = np.sum(x**2) - np.sum(x)**2/n
    #Sxy = np.sum(x*y) - np.sum(x)*np.sum(y)/n
    #Sxx2 = np.sum(x**3) - np.sum(x)*np.sum(x**2)/n
    #Sx2y = np.sum(x**2 * y) - np.sum(x**2)*np.sum(y)/n
    #Sx2x2 = np.sum(x**4) - np.sum(x**2)**2/n
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


cpdef long[:] arange(int start, int stop, int step):
    """ Make an array from start to stop in steps """
    cdef int num,i
    cdef long[:] array
    num = (stop-start) // step
    array = np.zeros(num,int)
    for i in range(num):
        array[i] = start + i*step
    return array

cpdef double[:] linspace(double start, double stop, int num):
    """ Make an array from start to stop with num elements """
    # the endpoint is always included
    cdef int i
    cdef double step
    cdef double[:] array
    step = (stop-start)/(num-1)
    array = np.zeros(num,np.float64)
    for i in range(num):
        array[i] = start + i*step
    return array

cdef list intmeshgrid(long[:] x, long[:] y):
    """ Implementation of numpy's meshgrid function."""
    cdef int nx,ny
    cdef long[:,:] xx,yy
    nx = len(x)
    ny = len(y)
    xx = np.zeros((ny,nx),int)
    for i in range(ny):
        xx[i,:] = x
    yy = np.zeros((ny,nx),int)
    for i in range(nx):
        yy[:,i] = y
    return xx,yy

cdef list meshgrid(double[:] x, double[:] y):
    """ Implementation of numpy's meshgrid function."""
    cdef int nx,ny
    cdef double[:,:] xx,yy
    nx = len(x)
    ny = len(y)
    xx = np.zeros((ny,nx),np.float64)
    for i in range(ny):
        xx[i,:] = x
    yy = np.zeros((ny,nx),np.float64)
    for i in range(nx):
        yy[:,i] = y
    return xx,yy

cdef double[:] aclip(double[:] val,double minval,double maxval):
    cdef double[:] newals
    newvals = np.zeros(len(val),np.float64)
    for i in range(len(val)):
        if val[i] < minval:
            nval = minval
        elif val[i] > maxval:
            nval = maxval
        else:
            nval = val[i]
        newvals[i] = nval
    return newvals

cdef double clip(double val,double minval,double maxval):
    if val < minval:
        nval = minval
    elif val > maxval:
        nval = maxval
    else:
        nval = val
    return nval

cdef double gamma(double z):
    # Gamma function for a single z value
    # Using the Lanczos approximation
    # https://en.wikipedia.org/wiki/Lanczos_approximation
    cdef int g,n,i
    cdef double y
    cdef double[:] p
    cdef double PI = 3.141592653589793
    g = 7
    n = 9
    p = np.array([
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    ])
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

cdef double gammaincinv05(double a):
    """ gammaincinv(a,0.5) """
    cdef int ind
    cdef double out,slp
    cdef double[:] n,y
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

cpdef double[:,:] inverse(double[:,:] a):
    """ Safely take the inverse of a square 2D matrix."""
    # This checks for zeros on the diagonal and "fixes" them.
    cdef int i
    cdef Py_ssize_t[:] badpar
    cdef double[:,:] ainv
    
    # If one of the dimensions is zero in the R matrix [Npars,Npars]
    # then replace it with a "dummy" value.  A large value in R
    # will give a small value in inverse of R.
    #badpar, = np.where(np.abs(np.diag(a))<sys.float_info.min)
    badpar = np.where(np.abs(np.diag(a))<2e-300)[0]
    if len(badpar)>0:
        for i in range(len(badpar)):
            a[badpar[i],badpar[i]] = 1e10
    ainv = np.linalg.inv(a)
    # What if the inverse fails???
    # can we catch it
    # Fix values
    if len(badpar)>0:
        for i in range(len(badpar)):
            #a[badpar] = 0  # put values back
            ainv[badpar[i],badpar[i]] = 0
    
    return ainv

cpdef double[:] qr_jac_solve(double[:,:] jac,double[:] resid,double[:] weight):
    """ Solve part of a non-linear least squares equation using QR decomposition
        using the Jacobian."""
    # jac: Jacobian matrix, first derivatives, [Npix, Npars]
    # resid: residuals [Npix]
    # weight: weights, ~1/error**2 [Npix]
    # dbeta: update array [Npix]
    cdef Py_ssize_t npix,npars
    cdef double wsum
    cdef double[:] dbeta,sqrtweight
    cdef double[:,:] sqrtweight2d

    npix = jac.shape[0]
    npars = jac.shape[1]

    # Check if the weights are all the same, unweighted
    wsum = 0.0
    for i in range(npix):
        wsum += np.abs(weight[i]-weight[0])
    
    # QR decomposition
    if wsum <= 1e-10:
        q,r = np.linalg.qr(jac)
        rinv = inverse(r)
        #rinv = np.linalg.pinv(r)  # use pseudo-inverse 
        dbeta = rinv @ (q.T @ resid)
    # Weights input, multiply resid and jac by weights        
    else:
        # We need to multiply Jac and resid by the sqrt(weight)
        sqrtweight = pow1d(weight,0.5)
        sqrtweight2d = np.zeros((npix,npars),np.float64)
        for i in range(npars):
            sqrtweight2d[:,i] = sqrtweight
        q,r = np.linalg.qr( mult2d(jac,sqrtweight2d) )
        rinv = inverse(r)
        #rinv = np.linalg.pinv(r)
        dbeta = rinv @ (q.T @ mult1d(resid,sqrtweight))
        
    return dbeta

cpdef double[:,:] jac_covariance(double[:,:] jac,double[:] resid, double[:] weight):
    """ Determine the covariance matrix. """
    cdef Py_ssize_t npix,npars
    cdef double dof,wsum
    cdef double[:,:] cov,cov_orig,hess,jacT

    npix = jac.shape[0]
    npars = jac.shape[1]

    jacT = transpose(jac)
    
    # Check if the weights are all the same, unweighted
    wsum = 0.0
    for i in range(npix):
        wsum += np.abs(weight[i]-weight[0])
    
    # Weights
    #   If weighted least-squares then
    #   J.T * W * J
    #   where W = I/sig_i**2
    if wsum > 1e-10:
        weight2d = np.zeros((npix,npars),np.float64)
        for i in range(npars):
            weight2d[:,i] = weight
        hess = matmult(jacT, (mult2d(weight2d,jac)))
    else:
        hess = matmult(jacT, jac)  # not weighted
        
    # cov = H-1, covariance matrix is inverse of Hessian matrix
    cov_orig = inverse(hess)
    
    # Rescale to get an unbiased estimate
    # cov_scaled = cov * (RSS/(m-n)), where m=number of measurements, n=number of parameters
    # RSS = residual sum of squares
    #  using rss gives values consistent with what curve_fit returns
    # Use chi-squared, since we are doing the weighted least-squares and weighted Hessian
    if wsum > 1e-10:
        chisq = np.sum(mult1d(pow1d(resid,2),weight))
    else:
        chisq = np.sum(pow1d(resid,2))
    dof = npix-npars
    if dof<=0:
        dof = 1
    cov = fact2d(cov_orig,(chisq/dof))  # what MPFITFUN suggests, but very small
        
    return cov


# checkbounds: return integer array
cpdef int[:] checkbounds(double[:] pars, double[:,:] bounds):
    """
    Check the parameters against the bounds.
    0 = ok, 1 = below lower bound, 2 = above upper bound
    """
    cdef Py_ssize_t npars = pars.shape[0]
    cdef double[:] lbounds = bounds[:,0]
    cdef double[:] ubounds = bounds[:,1]
    cdef int[:] check = np.zeros(npars, dtype=np.int32)
    cdef Py_ssize_t i

    for i in range(npars):
        if pars[i] <= lbounds[i]:
            check[i] = 1
        elif pars[i] >= ubounds[i]:
            check[i] = 2
        else:
            check[i] = 0
    return check


# limbounds: clip parameters to boundaries
cpdef double[:] limbounds(double[:] pars, double[:,:] bounds):
    """
    Limit the parameters to the boundaries.
    """
    cdef Py_ssize_t npars = pars.shape[0]
    cdef double[:] lbounds = bounds[:,0]
    cdef double[:] ubounds = bounds[:,1]
    cdef double[:] outpars = np.zeros(npars, dtype=np.float64)
    cdef Py_ssize_t i

    for i in range(npars):
        if pars[i] < lbounds[i]:
            outpars[i] = lbounds[i]
        elif pars[i] > ubounds[i]:
            outpars[i] = ubounds[i]
        else:
            outpars[i] = pars[i]

    return outpars


# limsteps: limit the magnitude of parameter steps
cpdef double[:] limsteps(double[:] steps, double maxsteps):
    """
    Limit the parameter steps to maximum step sizes.
    """
    cdef Py_ssize_t npars = steps.shape[0]
    cdef double[:] outsteps = np.zeros(npars, dtype=np.float64)
    cdef double s
    cdef Py_ssize_t i

    for i in range(npars):
        s = steps[i]
        if s >= 0:
            outsteps[i] = min(s, maxsteps)
        else:
            outsteps[i] = max(s, -maxsteps)

    return outsteps


cpdef double[:] newpars(double[:] pars,double[:] steps,
                        double[:,:] bounds,int maxsteps):
    """
    Return new parameters that fit the constraints.
    pars, steps: double[:]
    bounds: double[:,2] or None
    maxsteps: int, optional. If -1, no limit
    """
    cdef Py_ssize_t n = pars.shape[0]
    cdef double[:] limited_steps = np.zeros(n, dtype=np.float64)
    cdef double[:] newsteps = np.zeros(n, dtype=np.float64)
    cdef double[:] newpars_arr = np.zeros(n, dtype=np.float64)
    cdef double[:] lbounds
    cdef double[:] ubounds
    cdef double[:] check
    cdef bint[:] badpars
    cdef int count, maxiter = 2
    cdef Py_ssize_t i
    
    # Limit the steps
    if maxsteps > 0:
        limited_steps = limsteps(steps, maxsteps)
    else:
        for i in range(n):
            limited_steps[i] = steps[i]

    # Extract lower and upper bounds
    lbounds = bounds[:,0]
    ubounds = bounds[:,1]

    # Initialize
    check = checkbounds(add1d(pars,limited_steps), bounds)
    badpars = np.zeros(n, dtype=np.int8)
    for i in range(n):
        badpars[i] = check[i] != 0

    # Reduce step sizes if outside bounds
    newsteps[:] = limited_steps
    count = 0
    while np.sum(badpars) > 0 and count <= maxiter:
        for i in range(n):
            if badpars[i]:
                newsteps[i] /= 2
        check = checkbounds(add1d(pars,newsteps), bounds)
        for i in range(n):
            badpars[i] = check[i] != 0
        count += 1

    # Final parameters
    for i in range(n):
        newpars_arr[i] = pars[i] + newsteps[i]

    # Make sure final parameters stay inside bounds
    check = checkbounds(newpars_arr, bounds)
    for i in range(n):
        badpars[i] = check[i] != 0

    if np.sum(badpars) > 0:
        for i in range(n):
            newpars_arr[i] = min(max(newpars_arr[i], lbounds[i] + 1e-30),
                                 ubounds[i] - 1e-30)

    return newpars_arr

cpdef double[:] poly2d(double[:,:] xdata, double[:] pars):
    """ model of 2D linear polynomial."""
    cdef double[:] x,y,result
    cdef long n
    x = xdata[:,0]
    y = xdata[:,1]
    n = len(x)
    result = np.zeros(n,np.float64)
    for i in range(n):
        result[i] = pars[0]+pars[1]*x[i]+pars[2]*y[i]+pars[3]*x[i]*y[i]
    return result

cpdef list jacpoly2d(double[:,:] xdata, double[:] pars):
    """ jacobian of 2D linear polynomial."""
    cdef double[:] x,y,m
    cdef double[:,:] jac
    cdef long n
    x = xdata[:,0]
    y = xdata[:,1]
    n = len(x)
    # Model
    m = np.zeros(n,np.float64)
    for i in range(n):
        m[i] = pars[0]+pars[1]*x[i]+pars[2]*y[i]+pars[3]*x[i]*y[i]
    # Jacobian, partical derivatives wrt the parameters
    jac = np.zeros((n,4),np.float64)
    jac[:,0] = 1    # constant coefficient
    jac[:,1] = x    # x-coefficient
    jac[:,2] = y    # y-coefficient
    for i in range(n):
        jac[i,3] = x[i]*y[i]  # xy-coefficient
    return m,jac

cpdef list poly2dfit(double[:] x,double[:] y,double[:] data,double[:] error,int maxiter,double minpercdiff):
    """ Fit a 2D linear function to data robustly."""
    # maxiter=2
    # minpercdiff=0.5
    cdef int ndata,count
    cdef Py_ssize_t[:] gd1,gd2,gd,bad
    cdef double med,sig
    cdef double[:] initpar,bestpar,wt,wt1
    cdef double[:] data1,error1,resid
    cdef double[:,:] xdata
    
    ndata = len(data)
    #if ndata<4:
    #    raise Exception('Need at least 4 data points for poly2dfit')
    gd1, = np.where(np.isfinite(data))
    #if len(gd1)<4:
    #    raise Exception('Need at least 4 good data points for poly2dfit')
    xdata = np.zeros((len(gd1),2),np.float64)
    xdata[:,0] = np.array(x)[gd1]
    xdata[:,1] = np.array(y)[gd1]
    initpar = np.zeros(4,np.float64)
    med = np.median(np.array(data)[gd1])
    sig = mad(np.array(data)[gd1],0,0)
    resid = np.zeros(ndata,np.float64)
    for i in range(ndata):
        resid[i] = data[i]-med
    gd2, = np.where( (np.abs(resid)<3*sig) & np.isfinite(data))
    if len(gd1)>=4 and len(gd2)<4:
        gd = gd1
    else:
        gd = gd2
    initpar[0] = med
    xdata = np.zeros((len(gd),2),np.float64)
    xdata[:,0] = np.array(x)[gd]
    xdata[:,1] = np.array(y)[gd]
    data1 = np.array(data)[gd]
    error1 = np.array(error)[gd]

    # Do the fit
    # Iterate
    count = 0
    bestpar = initpar.copy()
    maxpercdiff = 1e10
    # maxsteps = None
    wt = np.zeros(ndata,np.float64)
    for i in range(ndata):
        wt[i] = 1.0/error1[i]**2
    wt1 = np.ones(ndata,np.float64)    
    while (count<maxiter and maxpercdiff>minpercdiff):
        # Use Cholesky, QR or SVD to solve linear system of equations
        m,j = jacpoly2d(xdata,bestpar)
        dy = data1-m
        # Solve Jacobian
        #if error is not None:
        #dbeta = qr_jac_solve(j,dy,weight=wt)
        dbeta = qr_jac_solve(j,dy,wt1)
        #else:
        #    dbeta = qr_jac_solve(j,dy)
        bad = np.where(~np.isfinite(dbeta))[0]
        if len(bad)>0:
            for i in range(len(bad)):
                dbeta[bad[i]] = 0.0    # deal with NaNs

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
        bestpar = add1d(bestpar,dbeta)
        # Check differences and changes
        diff = np.abs(sub1d(bestpar,oldpar))
        denom = np.maximum(np.abs(oldpar.copy()),0.0001)
        percdiff = diff.copy()/denom*100  # percent differences
        maxpercdiff = np.max(percdiff)
                
        #if verbose:
        #    print('N = ',count)
        #    print('bestpars = ',bestpar)
        #    print('dbeta = ',dbeta)
                
        count += 1

    # Get covariance and errors
    m,j = jacpoly2d(xdata,bestpar)
    dy = data1.ravel()-m.ravel()
    cov = jac_covariance(j,dy,wt)
    perror = np.sqrt(np.diag(cov))

    return bestpar,perror,cov


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
#     X1 = np.zeros((n1,2),np.float64)
#     X1[:,0] = ra1
#     X1[:,1] = dec1
#     X2 = np.zeros((n2,2),np.float64)
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
#         Y1 = np.zeros((n1,3),np.float64)
#         Y1[:,0] = np.cos(X1[:, 0]) * np.cos(X1[:, 1])
#         Y1[:,1] = np.sin(X1[:, 0]) * np.cos(X1[:, 1])
#         Y1[:,2] = np.sin(X1[:, 1])
#         #Y1 = np.transpose(np.vstack([np.cos(X1[:, 0]) * np.cos(X1[:, 1]),
#         #                             np.sin(X1[:, 0]) * np.cos(X1[:, 1]),
#         #                             np.sin(X1[:, 1])]))
#         Y2 = np.zeros((n2,3),np.float64)
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
#             find2 = np.zeros(len(ind2),np.float64)
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
#                         temp2 = np.zeros(10,np.float64)
#                         temp2[niter:] = dist[torem_orig_index[i],niter:]   #.squeeze()
#                         temp2[-niter:] = np.zeros(niter,np.float64)+np.inf
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
#     bgim = np.zeros((ny2,nx2),np.float64)
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

#     mnim = np.zeros(im.shape,np.float64)-100000
    
#     count = 0
#     xpeak = np.zeros(100000,np.float64)
#     ypeak = np.zeros(100000,np.float64)
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

#     mout = np.zeros((len(xpeak),17),np.float64)
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


cpdef Py_ssize_t[:] starbbox(double[:] coords,Py_ssize_t[:] imshape,double radius):
    """                                                                                        
     Return the boundary box for a star given radius and image size.                            
                                                                                                
    Parameters                                                                                 
    ----------                                                                                 
    coords: array                                                                  
       Central coordinates (xcen,ycen) of star (*absolute* values).                            
    imshape: array
       Image shape (ny,nx) values.  Python images are (Y,X).                                   
    radius: float                                                                              
       Radius in pixels.  
 
    Returns                                                                                     
    -------                                                                                     
    bbox : BoundingBox object                                                                   
       Bounding box of the x/y ranges.                                                          
       Upper values are EXCLUSIVE following the python convention.                              
                                                                                                
    """
    cdef Py_ssize_t nx,ny,xlo,xhi,ylo,yhi
    cdef double eta,xcen,ycen,x0,x1,y0,y1
    
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
    bbox = np.zeros(4,int)
    bbox[0] = xlo
    bbox[1] = xhi
    bbox[2] = ylo
    bbox[3] = yhi
    return bbox

cpdef double[:,:] slice2d(double[:,:] array, Py_ssize_t[:] bbox):
    """ Return a slice of a 2D array """
    # bbox is [xlo,xhi,ylo,yhi]
    cdef int i,j
    cdef Py_ssize_t xlo,xhi,ylo,yhi,nx,ny
    cdef double[:,:] subarray
    xlo,xhi,ylo,yhi = bbox
    nx = xhi-xlo
    ny = yhi-ylo
    # end points are EXCLUDED by the standard python convention
    subarray = np.zeros((ny,nx),np.float64)
    for i in range(ny):
        for j in range(nx):
            subarray[i,j] = array[i+ylo,j+xlo]
    return subarray

cpdef list relcoord(double[:] x, double[:] y, Py_ssize_t[:] shape):
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
    cdef int[:] midpt
    cdef double[:] relx,rely
    midpt = np.array(shape)
    midpt[0] = shape[0]//2
    midpt[1] = shape[1]//2
    nx = len(x)
    relx = np.zeros(nx,np.float64)
    rely = np.zeros(nx,np.float64)
    for i in range(nx):
        relx[i] = (x[i]-midpt[1])/shape[1]*2
        rely[i] = (y[i]-midpt[0])/shape[0]*2
    return [relx,rely]


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


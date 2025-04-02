# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
# cython: cdivision=True
# cython: binding=False
# cython: inter_types=True

import cython
cimport cython
import numpy as np
cimport numpy as np
from cython.view cimport array as cvarray
from cpython cimport array
from scipy.special import gamma, gammaincinv, gammainc

from libc.math cimport exp,sqrt,atan2,pi,NAN
from libcpp cimport bool
from libc.stdlib cimport malloc, free
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

cdef extern from "math.h":
    double sin(double x)
    double cos(double x)
    #double atan2(double x)


cdef (double,double,double) gauss_abt2cxy(double asemi,double bsemi,double theta):
    """ Convert asemi/bsemi/theta to cxx/cyy/cxy. """
    cdef float sintheta,costheta,sintheta2,costheta2
    cdef float asemi2,bsemi2,cxx,cyy,cxy
    # theta in radians
    sintheta = sin(theta)
    costheta = cos(theta)
    sintheta2 = sintheta**2
    costheta2 = costheta**2
    asemi2 = asemi**2
    bsemi2 = bsemi**2
    if asemi2 != 0.0 and bsemi2 != 0.0:
        cxx = costheta2/asemi2 + sintheta2/bsemi2
        cyy = sintheta2/asemi2 + costheta2/bsemi2
        cxy = 2*costheta*sintheta*(1/asemi2-1/bsemi2)
    else:
        cxx = NAN
        cyy = NAN
        cxy = NAN
    return cxx,cyy,cxy

cdef (double,double,double) gauss_cxy2abt(double cxx, double cyy, double cxy):
    """ Convert asemi/bsemi/theta to cxx/cyy/cxy. """
    cdef double xstd,ystd,theta,sin2t
    # a+c = 1/xstd2 + 1/ystd2
    # b = sin2t * (1/xstd2 + 1/ystd2)
    # tan 2*theta = b/(a-c)
    if cxx==cyy or cxy==0:
        theta = 0.0
    else:
        theta = atan2(cxy,cxx-cyy)/2.0

    if theta==0:
        # a = 1 / xstd2
        # b = 0        
        # c = 1 / ystd2
        xstd = 1/sqrt(cxx)
        ystd = 1/sqrt(cyy)
    else:
        sin2t = sin(2.0*theta)
        # b/sin2t + (a+c) = 2/xstd2
        # xstd2 = 2.0/(b/sin2t + (a+c))
        xstd = sqrt( 2.0/(cxy/sin2t + (cxx+cyy)) )

        # a+c = 1/xstd2 + 1/ystd2
        ystd = sqrt( 1/(cxx+cyy-1/xstd**2) )

        # theta in radians

    return xstd,ystd,theta


#cpdef list agaussian2d(double[:] x, double[:] y, double[:] pars, int nderiv):
cdef double* agaussian2d(double[:] x, double[:] y, double[:] pars, int nderiv):
#cpdef double[:,:] agaussian2d(double[:] x, double[:] y, double[:] pars, int nderiv):
    """
    Two dimensional Gaussian model function with x/y array inputs.
    
    Parameters
    ----------
    x : numpy array
      Array of X-values of points for which to compute the Gaussian model.
    y : numpy array
      Array of Y-values of points for which to compute the Gaussian model.
    pars : numpy array
       Parameter list. pars = [amplitude, x0, y0, xsigma, ysigma, theta].
         Or can include cxx, cyy, cxy at the end so they don't have to be
         computed.
    nderiv : int
       The number of derivatives to return.

    Returns
    -------
    g : numpy array
      The Gaussian model for the input x/y values and parameters.  Always
        returned as 1D raveled() array.
    derivative : numpy array
      List of derivatives of g relative to the input parameters.
        Always 2D [Npix,Nderiv] with the 1st dimension being the x/y arrays
        raveled() to 1D.

    Example
    -------

    g,derivative = agaussian2d(x,y,pars,3)

    """
    #cdef double[:] g = np.zeros(len(x))
    #cdef double[:,:] deriv = np.zeros((len(x),nderiv))
    #cdef double[:] d1 = np.zeros(nderiv)
    cdef double amp,xc,yc,asemi,bsemi,theta,cxx,cyy,cxy
    cdef long i,j

    amp = pars[0]
    xc = pars[1]
    yc = pars[2]
    asemi = pars[3]
    bsemi = pars[4]
    theta = pars[5]
    if len(pars)==6:
        cxx,cyy,cxy = gauss_abt2cxy(asemi,bsemi,theta)
    else:
        cxx = pars[6]
        cyy = pars[7]
        cxy = pars[8]
    #cdef array.array allpars = array.array('d',[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    #allpars = cvarray(shape=(9,),itemsize=sizeof(double),format="d")
    cdef double allpars[9]
    allpars[0] = amp
    allpars[1] = xc
    allpars[2] = yc
    allpars[3] = asemi
    allpars[4] = bsemi
    allpars[5] = theta
    allpars[6] = cxx
    allpars[7] = cyy
    allpars[8] = cxy

    # Unravel 2D arrays
    npix = len(x)
    # Initialize output
    #g = np.zeros(npix,float)
    #if nderiv>0:
    #    deriv = np.zeros((npix,nderiv))
    #else:
    #    deriv = np.zeros((1,1))

    #cdef double *g = <double*>malloc(npix * sizeof(double))
    #cdef double *deriv = <double*>malloc(6*npix * sizeof(double))
    cdef double *out =  <double*>malloc(7*npix * sizeof(double))
    #out = cvarray(shape=(npix,7),itemsize=sizeof(double),format="d")
    #cdef double[:,:] mout = out

    #g = np.zeros(len(x),float)
    #g = cvarray(shape=(npix,),itemsize=sizeof(double),format="d")
    #cdef double[:] mg = g
    #deriv = np.zeros((len(x),nderiv),float)
    #deriv = cvarray(shape=(npix,nderiv),itemsize=sizeof(double),format="d")
    #cdef double[:,:] mderiv = deriv
    # Loop over the points
    for i in range(npix):
        #g1,deriv1 = gaussian2d(x[i],y[i],allpars,nderiv)
        #mg[i] = g1
        out1 = gaussian2d(x[i],y[i],allpars,nderiv)
        for j in range(nderiv+1):
            #mout[i,j] = out1[j]
            out[i*7+j] = out1[j]
        #if nderiv>0:
        #    #deriv[i,:] = deriv1
        #    for j in range(nderiv):
        #        #mderiv[i,j] = deriv1[j]
        #        mderiv[i,j] = out1[j+1]
        free(out1)
    return out
    #return mout
    #return [mg,mderiv]
    #return [np.asarray(g),np.asarray(deriv)]


cpdef void testgaussian2d(double[:] x, double[:] y, double[:] pars, int nderiv):
    out = agaussian2d(x,y,pars,nderiv)
    free(out)


cpdef double[:] npgaussian2d(double[:] x, double[:] y, double[:] pars, int nderiv):
    cdef double[:] xdiff,ydiff
    cdef double amp,xsig,ysig,theta,cost2,sint2,sin2t,xsig2,ysig2
    cdef long npix

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

    npix = len(x)
    xdiff = np.zeros(npix,float)
    ydiff = np.zeros(npix,float)
    g = np.zeros(npix,float)
    for i in range(npix):
        xdiff[i] = x[i] - pars[1]
        ydiff[i] = y[i] - pars[2]
    #g = amp * np.exp(-0.5*((a * xdiff**2) + (b * xdiff * ydiff) + (c * ydiff**2)))

    # cdef double[:,:] out = np.zeros((npix,nderiv+1),float)

    # # Compute derivative as well
    # if nderiv>0:
    #     if nderiv>=1:
    #         dg_dA = g / amp
    #         out[:,1] = dg_dA
    #         #derivative.append(dg_dA)
    #     if nderiv>=2:        
    #         dg_dx_mean = g * 0.5*((2 * a * xdiff) + (b * ydiff))
    #         out[:,2] = dg_dx_mean
    #         #derivative.append(dg_dx_mean)
    #     if nderiv>=3:
    #         dg_dy_mean = g * 0.5*((2 * c * ydiff) + (b * xdiff))
    #         out[:,3] = dg_dy_mean
    #         #derivative.append(dg_dy_mean)
    #     if nderiv>=4:
    #         xdiff2 = xdiff ** 2
    #         ydiff2 = ydiff ** 2
    #         xsig3 = xsig ** 3
    #         da_dxsig = -cost2 / xsig3
    #         db_dxsig = -sin2t / xsig3            
    #         dc_dxsig = -sint2 / xsig3            
    #         dg_dxsig = g * (-(da_dxsig * xdiff2 +
    #                           db_dxsig * xdiff * ydiff +
    #                           dc_dxsig * ydiff2))
    #         out[:,4] = dg_dxsig
    #         #derivative.append(dg_dxsig)
    #     if nderiv>=5:
    #         ysig3 = ysig ** 3
    #         da_dysig = -sint2 / ysig3
    #         db_dysig = sin2t / ysig3            
    #         dc_dysig = -cost2 / ysig3            
    #         dg_dysig = g * (-(da_dysig * xdiff2 +
    #                           db_dysig * xdiff * ydiff +
    #                           dc_dysig * ydiff2))
    #         out[:,5] = dg_dysig
    #         #derivative.append(dg_dysig)
    #     if nderiv>=6:
    #         sint = np.sin(theta)
    #         cost = np.cos(theta)
    #         cos2t = np.cos(2.0*theta)
    #         da_dtheta = (sint * cost * ((1. / ysig2) - (1. / xsig2)))
    #         db_dtheta = (cos2t / xsig2) - (cos2t / ysig2)            
    #         dc_dtheta = -da_dtheta            
    #         dg_dtheta = g * (-(da_dtheta * xdiff2 +
    #                            db_dtheta * xdiff * ydiff +
    #                            dc_dtheta * ydiff2))
    #         #derivative.append(dg_dtheta)
    #         out[:,6] = dg_dtheta

    # return out


#cdef (double,double[6]) gaussian2d(double x, double y, double[:] pars, int nderiv):
#cdef list gaussian2d(double x, double y, double[:] pars, int nderiv):
cdef double* gaussian2d(double x, double y, double[:] pars, int nderiv):
#cdef double[:] gaussian2d(double x, double y, double[:] pars, int nderiv):
    """
    Two dimensional Gaussian model function.
    
    Parameters
    ----------
    x : float
      Single X-value for which to compute the Gaussian model.
    y : float
      Single Y-value of points for which to compute the Gaussian model.
    pars : numpy array
       Parameter list. pars = [amplitude, x0, y0, xsigma, ysigma, theta]
    nderiv : int
       The number of derivatives to return.

    Returns
    -------
    g : float
      The Gaussian model for the input x/y values and parameters (same
        shape as x/y).
    derivative : numpy array
      Array of derivatives of g relative to the input parameters.

    Example
    -------

    g,derivative = gaussian2d(x,y,pars,nderiv)

    """
    cdef double g
    #cdef double[6] deriv = [0.0,0.0,0.0,0.0,0.0,0.0]
    #cdef double[7] out
    cdef double *out = <double*>malloc(7 * sizeof(double))
    #cdef double[7] out

    amp = pars[0]
    xc = pars[1]
    yc = pars[2]
    asemi = pars[3]
    bsemi = pars[4]
    theta = pars[5]
    if len(pars)==6:
        cxx,cyy,cxy = gauss_abt2cxy(asemi,bsemi,theta)
    else:
        cxx = pars[6]
        cyy = pars[7]
        cxy = pars[8]

    u = (x-xc)
    u2 = u**2
    v = (y-yc)
    v2 = v**2
    # amp = 1/(asemi*bsemi*2*np.pi)
    g = amp * exp(-0.5*(cxx*u**2 + cyy*v**2 + cxy*u*v))

    out[0] = g

    #return g

    #  pars = [amplitude, x0, y0, xsigma, ysigma, theta]
    #deriv = np.zeros(nderiv,float)
    #deriv = array.array(6)
    #deriv[0] = 0.0
    if nderiv>0:
        # amplitude
        dg_dA = g / amp
        #deriv[0] = dg_dA
        out[1] = dg_dA
        # x0
        dg_dx_mean = g * 0.5*((2. * cxx * u) + (cxy * v))
        #deriv[1] = dg_dx_mean
        out[2] = dg_dx_mean
        # y0
        dg_dy_mean = g * 0.5*((cxy * u) + (2. * cyy * v))
        #deriv[2] = dg_dy_mean
        out[3] = dg_dy_mean
        if nderiv>3:
            sint = sin(theta)        
            cost = cos(theta)        
            sint2 = sint ** 2
            cost2 = cost ** 2
            sin2t = sin(2. * theta)
            # xsig
            asemi2 = asemi ** 2
            asemi3 = asemi ** 3
            da_dxsig = -cost2 / asemi3
            db_dxsig = -sin2t / asemi3
            dc_dxsig = -sint2 / asemi3
            dg_dxsig = g * (-(da_dxsig * u2 +
                              db_dxsig * u * v +
                              dc_dxsig * v2))
            #deriv[3] = dg_dxsig
            out[4] = dg_dxsig
            # ysig
            bsemi2 = bsemi ** 2
            bsemi3 = bsemi ** 3
            da_dysig = -sint2 / bsemi3
            db_dysig = sin2t / bsemi3
            dc_dysig = -cost2 / bsemi3
            dg_dysig = g * (-(da_dysig * u2 +
                              db_dysig * u * v +
                              dc_dysig * v2))
            #deriv[4] = dg_dysig
            out[5] = dg_dysig
            # dtheta
            if asemi != bsemi:
                cos2t = cos(2.0*theta)
                da_dtheta = (sint * cost * ((1. / bsemi2) - (1. / asemi2)))
                db_dtheta = (cos2t / asemi2) - (cos2t / bsemi2)
                dc_dtheta = -da_dtheta
                dg_dtheta = g * (-(da_dtheta * u2 +
                                   db_dtheta * u * v +
                                   dc_dtheta * v2))
                #deriv[5] = dg_dtheta
                out[6] = dg_dtheta

    return out

#cpdef void testgauss2d(double[:] x, double[:] y, double[:] pars, int nderiv):
cpdef double[:,:] testgauss2d(double[:] x, double[:] y, double[:] pars, int nderiv):
    out = gauss2d(x,y,pars,nderiv)

    cdef long npix = len(x)
    cdef long i,j
    cdef double[:,:] npout = np.zeros((npix,7),float)
    for i in range(npix):
        #index = i*7
        for j in range(7):
            npout[i,j] = out[i*7+j]
    free(out)

    return npout    


#cpdef list gauss2d(double[:] x, double[:] y, double[:] pars, int nderiv):
cdef double* gauss2d(double[:] x, double[:] y, double[:] pars, int nderiv):
#cdef double[:,:] gauss2d(double[:] x, double[:] y, double[:] pars, int nderiv):
    """
    Two dimensional Gaussian model function with x/y array inputs.
    
    Parameters
    ----------
    x : numpy array
      Array of X-values of points for which to compute the Gaussian model.
    y : numpy array
      Array of Y-values of points for which to compute the Gaussian model.
    pars : numpy array
       Parameter list. pars = [amplitude, x0, y0, xsigma, ysigma, theta].
         Or can include cxx, cyy, cxy at the end so they don't have to be
         computed.
    nderiv : int
       The number of derivatives to return.

    Returns
    -------
    g : numpy array
      The Gaussian model for the input x/y values and parameters.  Always
        returned as 1D raveled() array.
    derivative : numpy array
      List of derivatives of g relative to the input parameters.
        Always 2D [Npix,Nderiv] with the 1st dimension being the x/y arrays
        raveled() to 1D.

    Example
    -------

    g,derivative = agaussian2d(x,y,pars,3)

    """
    #cdef double[:] g = np.zeros(len(x))
    #cdef double[:,:] deriv = np.zeros((len(x),nderiv))
    #cdef double[:] d1 = np.zeros(nderiv)
    cdef double amp,xc,yc,asemi,bsemi,theta,cxx,cyy,cxy
    cdef double u,u2,v,v2
    cdef long i

    amp = pars[0]
    xc = pars[1]
    yc = pars[2]
    asemi = pars[3]
    bsemi = pars[4]
    theta = pars[5]
    if len(pars)==6:
        cxx,cyy,cxy = gauss_abt2cxy(asemi,bsemi,theta)
    else:
        cxx = pars[6]
        cyy = pars[7]
        cxy = pars[8]
    #cdef array.array allpars = array.array('d',[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    #allpars = cvarray(shape=(9,),itemsize=sizeof(double),format="d")
    #cdef double allpars[9]
    #allpars[0] = amp
    #allpars[1] = xc
    #allpars[2] = yc
    #allpars[3] = asemi
    #allpars[4] = bsemi
    #allpars[5] = theta
    #allpars[6] = cxx
    #allpars[7] = cyy
    #allpars[8] = cxy

    sint = sin(theta)
    cost = cos(theta)        
    sint2 = sint ** 2
    cost2 = cost ** 2
    sin2t = sin(2. * theta)
    asemi2 = asemi ** 2
    asemi3 = asemi ** 3
    bsemi2 = bsemi ** 2
    bsemi3 = bsemi ** 3
    cos2t = cos(2.0*theta)

    npix = len(x)

    #cdef double *g = <double*>malloc(npix * sizeof(double))
    #cdef double *deriv = <double*>malloc(6*npix * sizeof(double))
    cdef double *out =  <double*>malloc(7*npix * sizeof(double))
    #out = cvarray(shape=(npix,7),itemsize=sizeof(double),format="d")
    #cdef double[:,:] mout = out

    #g = np.zeros(len(x),float)
    #g = cvarray(shape=(npix,),itemsize=sizeof(double),format="d")

    #cdef double[:] mg = g
    #deriv = np.zeros((len(x),nderiv),float)
    #deriv = cvarray(shape=(npix,nderiv),itemsize=sizeof(double),format="d")
    #cdef double[:,:] mderiv = deriv
    cdef long index = 0
    #cdef long ncols = npix
    cdef long ncols = 7
    # Loop over the points
    for i in range(npix):
        #index = i * cols + j;
        index = i * ncols
        u = (x[i]-xc)
        u2 = u**2
        v = (y[i]-yc)
        v2 = v**2
        # amp = 1/(asemi*bsemi*2*np.pi)
        g = amp * exp(-0.5*(cxx*u**2 + cyy*v**2 + cxy*u*v))
        #out[i,0] = g
        out[index] = g

        #  pars = [amplitude, x0, y0, xsigma, ysigma, theta]
        #deriv = np.zeros(nderiv,float)
        #deriv = array.array(6)
        #deriv[0] = 0.0
        if nderiv>0:
            # amplitude
            dg_dA = g / amp
            #deriv[0] = dg_dA
            #out[i,1] = dg_dA
            out[index+1] = dg_dA
            # x0
            dg_dx_mean = g * 0.5*((2. * cxx * u) + (cxy * v))
            #deriv[1] = dg_dx_mean
            #out[i,2] = dg_dx_mean
            out[index+2] = dg_dx_mean
            # y0
            dg_dy_mean = g * 0.5*((cxy * u) + (2. * cyy * v))
            #deriv[2] = dg_dy_mean
            #out[i,3] = dg_dy_mean
            out[index+3] = dg_dy_mean
            if nderiv>3:
                # xsig
                da_dxsig = -cost2 / asemi3
                db_dxsig = -sin2t / asemi3
                dc_dxsig = -sint2 / asemi3
                dg_dxsig = g * (-(da_dxsig * u2 +
                                  db_dxsig * u * v +
                                  dc_dxsig * v2))
                #deriv[3] = dg_dxsig
                #out[i,4] = dg_dxsig
                out[index+4] = dg_dxsig		
                # ysig
                da_dysig = -sint2 / bsemi3
                db_dysig = sin2t / bsemi3
                dc_dysig = -cost2 / bsemi3
                dg_dysig = g * (-(da_dysig * u2 +
                                  db_dysig * u * v +
                                  dc_dysig * v2))
                #deriv[4] = dg_dysig
                #out[i,5] = dg_dysig
                out[index+5] = dg_dysig
                # dtheta
                if asemi != bsemi:
                    da_dtheta = (sint * cost * ((1. / bsemi2) - (1. / asemi2)))
                    db_dtheta = (cos2t / asemi2) - (cos2t / bsemi2)
                    dc_dtheta = -da_dtheta
                    dg_dtheta = g * (-(da_dtheta * u2 +
                                       db_dtheta * u * v +
                                       dc_dtheta * v2))
                    #deriv[5] = dg_dtheta
                    #out[i,6] = dg_dtheta
                    out[index+6] = dg_dtheta

    return out



#cpdef list gauss2d2(double[:] x, double[:] y, double[:] pars, int nderiv):
#cdef double* gauss2d2(double[:] x, double[:] y, double[:] pars, int nderiv):
cpdef double[:,:] gauss2d2(double[:] x, double[:] y, double[:] pars, int nderiv):
#cpdef double[:] gauss2d2(double[:] x, double[:] y, double[:] pars, int nderiv):
    """
    Two dimensional Gaussian model function with x/y array inputs.
    
    Parameters
    ----------
    x : numpy array
      Array of X-values of points for which to compute the Gaussian model.
    y : numpy array
      Array of Y-values of points for which to compute the Gaussian model.
    pars : numpy array
       Parameter list. pars = [amplitude, x0, y0, xsigma, ysigma, theta].
         Or can include cxx, cyy, cxy at the end so they don't have to be
         computed.
    nderiv : int
       The number of derivatives to return.

    Returns
    -------
    g : numpy array
      The Gaussian model for the input x/y values and parameters.  Always
        returned as 1D raveled() array.
    derivative : numpy array
      List of derivatives of g relative to the input parameters.
        Always 2D [Npix,Nderiv] with the 1st dimension being the x/y arrays
        raveled() to 1D.

    Example
    -------

    g,derivative = agaussian2d(x,y,pars,3)

    """
    #cdef double[:] g = np.zeros(len(x))
    #cdef double[:,:] deriv = np.zeros((len(x),nderiv))
    #cdef double[:] d1 = np.zeros(nderiv)
    cdef double amp,xc,yc,asemi,bsemi,theta,cxx,cyy,cxy
    cdef double u,u2,v,v2
    cdef long i

    amp = pars[0]
    xc = pars[1]
    yc = pars[2]
    asemi = pars[3]
    bsemi = pars[4]
    theta = pars[5]
    if len(pars)==6:
        cxx,cyy,cxy = gauss_abt2cxy(asemi,bsemi,theta)
    else:
        cxx = pars[6]
        cyy = pars[7]
        cxy = pars[8]
    #cdef array.array allpars = array.array('d',[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    #allpars = cvarray(shape=(9,),itemsize=sizeof(double),format="d")
    #cdef double allpars[9]
    #allpars[0] = amp
    #allpars[1] = xc
    #allpars[2] = yc
    #allpars[3] = asemi
    #allpars[4] = bsemi
    #allpars[5] = theta
    #allpars[6] = cxx
    #allpars[7] = cyy
    #allpars[8] = cxy

    sint = sin(theta)
    cost = cos(theta)        
    sint2 = sint ** 2
    cost2 = cost ** 2
    sin2t = sin(2. * theta)
    asemi2 = asemi ** 2
    asemi3 = asemi ** 3
    bsemi2 = bsemi ** 2
    bsemi3 = bsemi ** 3
    cos2t = cos(2.0*theta)

    npix = len(x)

    #cdef double *g = <double*>malloc(npix * sizeof(double))
    #cdef double *deriv = <double*>malloc(6*npix * sizeof(double))
    #cdef double *out =  <double*>malloc(7*npix * sizeof(double))
    # 1D arrays
    #out = cvarray(shape=(npix*7,),itemsize=sizeof(double),format="d")
    #cdef double[:] mout = out
    # 2D arrays
    out = cvarray(shape=(npix,7),itemsize=sizeof(double),format="d")
    cdef double[:,:] mout = out


    #cdef double[:] mg = g
    #deriv = np.zeros((len(x),nderiv),float)
    #deriv = cvarray(shape=(npix,nderiv),itemsize=sizeof(double),format="d")
    #cdef double[:,:] mderiv = deriv
    cdef long index = 0
    #cdef long ncols = npix
    cdef long ncols = 7
    # Loop over the points
    for i in range(npix):
        #index = i * cols + j;
        index = i * ncols
        u = (x[i]-xc)
        u2 = u**2
        v = (y[i]-yc)
        v2 = v**2
        # amp = 1/(asemi*bsemi*2*np.pi)
        g = amp * exp(-0.5*(cxx*u**2 + cyy*v**2 + cxy*u*v))
        #out[i,0] = g
        #mout[index] = g
        mout[i,0] = g

        #  pars = [amplitude, x0, y0, xsigma, ysigma, theta]
        #deriv = np.zeros(nderiv,float)
        #deriv = array.array(6)
        #deriv[0] = 0.0
        if nderiv>0:
            # amplitude
            dg_dA = g / amp
            #deriv[0] = dg_dA
            #out[i,1] = dg_dA
            #mout[index+1] = dg_dA
            mout[i,1] = dg_dA
            # x0
            dg_dx_mean = g * 0.5*((2. * cxx * u) + (cxy * v))
            #deriv[1] = dg_dx_mean
            #out[i,2] = dg_dx_mean
            #mout[index+2] = dg_dx_mean
            mout[i,2] = dg_dx_mean
            # y0
            dg_dy_mean = g * 0.5*((cxy * u) + (2. * cyy * v))
            #deriv[2] = dg_dy_mean
            #out[i,3] = dg_dy_mean
            #mout[index+3] = dg_dy_mean
            mout[i,3] = dg_dy_mean
            if nderiv>3:
                # xsig
                da_dxsig = -cost2 / asemi3
                db_dxsig = -sin2t / asemi3
                dc_dxsig = -sint2 / asemi3
                dg_dxsig = g * (-(da_dxsig * u2 +
                                  db_dxsig * u * v +
                                  dc_dxsig * v2))
                #deriv[3] = dg_dxsig
                #out[i,4] = dg_dxsig
                #mout[index+4] = dg_dxsig
                mout[i,4] = dg_dxsig
                # ysig
                da_dysig = -sint2 / bsemi3
                db_dysig = sin2t / bsemi3
                dc_dysig = -cost2 / bsemi3
                dg_dysig = g * (-(da_dysig * u2 +
                                  db_dysig * u * v +
                                  dc_dysig * v2))
                #deriv[4] = dg_dysig
                #out[i,5] = dg_dysig
                #mout[index+5] = dg_dysig
                mout[i,5] = dg_dysig
                # dtheta
                if asemi != bsemi:
                    da_dtheta = (sint * cost * ((1. / bsemi2) - (1. / asemi2)))
                    db_dtheta = (cos2t / asemi2) - (cos2t / bsemi2)
                    dc_dtheta = -da_dtheta
                    dg_dtheta = g * (-(da_dtheta * u2 +
                                       db_dtheta * u * v +
                                       dc_dtheta * v2))
                    #deriv[5] = dg_dtheta
                    #out[i,6] = dg_dtheta
                    #mout[index+6] = dg_dtheta
                    mout[i,6] = dg_dtheta

    #pout = np.zeros((npix,7),float)
    #pout = cvarray(shape=(npix,7),itemsize=sizeof(double),format="d")
    #for i in range(npix):
    #    for j in range(7):
    #        pout[i,j] = out[i*ncols+j]

    return mout


#cdef (double,double[6]) gauss1d(double x, double y, double[:] pars, int nderiv):
#cdef list gauss1d(double x, double y, double[:] pars, int nderiv):
#cdef double* gauss1d(double x, double y, double[:] pars, int nderiv):
#cdef double[:] gauss1d(double x, double y, double[:] pars, int nderiv):
#cdef void gauss1d(double x, double y, double[:] pars, int nderiv):
#cdef void gauss1d(double x, double y, double amp, double xc, double yc,
#                  double asemi, double bsemi, double theta, double cxx,
#		  double cyy, double cxy, int nderiv):
#cdef double gauss1d(double x, double y, double amp, double xc, double yc,
#                    double asemi, double bsemi, double theta, double cxx,
#                    double cyy, double cxy, int nderiv):
#cdef (double,double,double,double,double,double,double) gauss1d(double x, double y,
#                    double amp, double xc, double yc,
#                    double asemi, double bsemi, double theta, double cxx,
#                    double cyy, double cxy, int nderiv):
#cdef array.array gauss1d(double x, double y, double amp, double xc, double yc,
#cdef double[:] gauss1d(double x, double y, double amp, double xc, double yc,
#cdef void gauss1d(double x, double y, double amp, double xc, double yc,
#                  double asemi, double bsemi, double theta, double cxx,
#                  double cyy, double cxy, int nderiv, double* out):
cdef double* gauss1d(double x, double y, double amp, double xc, double yc,
                     double asemi, double bsemi, double theta, double cxx,
                     double cyy, double cxy, int nderiv):
    cdef double g
    #cdef double[6] deriv = [0.0,0.0,0.0,0.0,0.0,0.0]
    #cdef double[7] out
    #cdef double *out = <double*>malloc(7 * sizeof(double))
    #cdef double[7] out
    #cdef double amp,xc,yc,asemi,bsemi,theta,cxx,cyy,cxy
    cdef double u,u2,v,v2
    #cdef double[7] out = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    cdef double dg_dA,dg_dx_mean,dg_dy_mean,dg_dxsig,dg_dysig,dg_dtheta

    #amp = pars[0]
    #xc = pars[1]
    #yc = pars[2]
    #asemi = pars[3]
    #bsemi = pars[4]
    #theta = pars[5]
    #cxx = pars[6]
    #cyy = pars[7]
    #cxy = pars[8]

    u = (x-xc)
    u2 = u**2
    v = (y-yc)
    v2 = v**2
    # amp = 1/(asemi*bsemi*2*np.pi)
    g = amp * exp(-0.5*(cxx*u**2 + cyy*v**2 + cxy*u*v))

    #out = cvarray(shape=(7,),itemsize=sizeof(double),format="d")
    #cdef double[:] mout = out
    #cdef array.array out
    #out = array.array('d',[0,0,0,0,0,0,0])
    #cdef double out[7]
    #cdef double[:] mout = out
    #mout[0] = 0
    #mout[1] = 0
    #mout[2] = 0
    #mout[3] = 0
    #mout[4] = 0
    #mout[5] = 0
    #mout[6] = 0

    #out[0] = g

    cdef double[7] out = [0,0,0,0,0,0,0]

    dg_dA = 0.0
    dg_dx_mean = 0.0
    dg_dy_mean = 0.0
    dg_dxsig = 0.0
    dg_dysig = 0.0
    dg_dtheta = 0.0

    if nderiv>0:
        # amplitude
        dg_dA = g / amp
        #deriv[0] = dg_dA
        #out[1] = dg_dA
        # x0
        dg_dx_mean = g * 0.5*((2. * cxx * u) + (cxy * v))
        #deriv[1] = dg_dx_mean
        #out[2] = dg_dx_mean
        # y0
        dg_dy_mean = g * 0.5*((cxy * u) + (2. * cyy * v))
        #deriv[2] = dg_dy_mean
        #out[3] = dg_dy_mean
        if nderiv>3:
            sint = sin(theta)        
            cost = cos(theta)        
            sint2 = sint ** 2
            cost2 = cost ** 2
            sin2t = sin(2. * theta)
            # xsig
            asemi2 = asemi ** 2
            asemi3 = asemi ** 3
            da_dxsig = -cost2 / asemi3
            db_dxsig = -sin2t / asemi3
            dc_dxsig = -sint2 / asemi3
            dg_dxsig = g * (-(da_dxsig * u2 +
                              db_dxsig * u * v +
                              dc_dxsig * v2))
            #deriv[3] = dg_dxsig
            #out[4] = dg_dxsig
            # ysig
            bsemi2 = bsemi ** 2
            bsemi3 = bsemi ** 3
            da_dysig = -sint2 / bsemi3
            db_dysig = sin2t / bsemi3
            dc_dysig = -cost2 / bsemi3
            dg_dysig = g * (-(da_dysig * u2 +
                              db_dysig * u * v +
                              dc_dysig * v2))
            #deriv[4] = dg_dysig
            #out[5] = dg_dysig
            # dtheta
            if asemi != bsemi:
                cos2t = cos(2.0*theta)
                da_dtheta = (sint * cost * ((1. / bsemi2) - (1. / asemi2)))
                db_dtheta = (cos2t / asemi2) - (cos2t / bsemi2)
                dc_dtheta = -da_dtheta
                dg_dtheta = g * (-(da_dtheta * u2 +
                                   db_dtheta * u * v +
                                   dc_dtheta * v2))
                #deriv[5] = dg_dtheta
                #out[6] = dg_dtheta

    out[0] = g
    out[1] = dg_dA
    out[2] = dg_dx_mean
    out[3] = dg_dy_mean
    out[4] = dg_dxsig
    out[5] = dg_dysig
    out[6] = dg_dtheta

    #print(out)

    return out
    #return mout
    #return g
    #return g,dg_dA,dg_dx_mean,dg_dy_mean,dg_dxsig,dg_dysig,dg_dtheta

#cpdef list gauss2d3(double[:] x, double[:] y, double[:] pars, int nderiv):
cdef double* gauss2d3(double[:] x, double[:] y, double[:] pars, int nderiv):
#cpdef double[:,:] gauss2d3(double[:] x, double[:] y, double[:] pars, int nderiv):
#cpdef double[:] gauss2d3(double[:] x, double[:] y, double[:] pars, int nderiv):
#cdef void gauss2d3(double[:] x, double[:] y, double[:] pars, int nderiv):

    #cdef double[:] g = np.zeros(len(x))
    #cdef double[:,:] deriv = np.zeros((len(x),nderiv))
    #cdef double[:] d1 = np.zeros(nderiv)
    #cdef double amp,xc,yc,asemi,bsemi,theta,cxx,cyy,cxy
    #cdef double u,u2,v,v2
    cdef double x1,y1
    cdef long i,j,npix,index

    amp = pars[0]
    xc = pars[1]
    yc = pars[2]
    asemi = pars[3]
    bsemi = pars[4]
    theta = pars[5]
    if len(pars)==6:
        cxx,cyy,cxy = gauss_abt2cxy(asemi,bsemi,theta)
    else:
        cxx = pars[6]
        cyy = pars[7]
        cxy = pars[8]
    #cdef array.array allpars = array.array('d',[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    cdef double allpars[9]
    #allpars = cvarray(shape=(9,),itemsize=sizeof(double),format="d")
    allpars[0] = amp
    allpars[1] = xc
    allpars[2] = yc
    allpars[3] = asemi
    allpars[4] = bsemi
    allpars[5] = theta
    allpars[6] = cxx
    allpars[7] = cyy
    allpars[8] = cxy

    npix = len(x)

    #cdef double *g = <double*>malloc(npix * sizeof(double))
    #cdef double *deriv = <double*>malloc(6*npix * sizeof(double))
    cdef double *out =  <double*>malloc(7*npix * sizeof(double))
    #cdef double *out =  <double*>PyMem_Malloc(7*npix * sizeof(double))
    # 1D arrays
    #out = cvarray(shape=(npix*7,),itemsize=sizeof(double),format="d")
    #cdef double[:] mout = out
    # 2D arrays
    #out = cvarray(shape=(npix,7),itemsize=sizeof(double),format="d")
    #cdef double[:,:] mout = out

    #cdef double out1 *
    #cdef double[7] out1 = [0,0,0,0,0,0,0]
    #cdef double o1,o2,o3,o4,o5,o6,o7
    #cdef double[:] out1
    #out1[0] = 0.0
    #out1[1] = 0.0
    #out1[2] = 0.0
    #out1[3] = 0.0
    #out1[4] = 0.0
    #out1[5] = 0.0
    #out1[6] = 0.0

    # Loop over the points
    for i in range(npix):
        x1 = x[i]
        y1 = y[i]
        #gauss1d(1.0,1.0,allpars,6)
        #gauss1d(1.0,1.0,amp,xc,yc,asemi,bsemi,theta,cxx,cyy,cxy,nderiv)
        out1 = gauss1d(x1,y1,amp,xc,yc,asemi,bsemi,theta,cxx,cyy,cxy,nderiv)
        #print(out1)
        #gauss1d(x1,y1,amp,xc,yc,asemi,bsemi,theta,cxx,cyy,cxy,nderiv,out1)
        #o1,o2,o3,o4,o5,o6,o7 = gauss1d(x1,y1,amp,xc,yc,asemi,bsemi,theta,cxx,cyy,cxy,nderiv)
        #out1 = gauss1d(x1,y1,allpars,nderiv)
        #for j in range(nderiv+1):
        #    mout[i,j] = out1[j]
        #mout[i,0] = o1
        #mout[i,1] = o2
        #mout[i,2] = o3
        #mout[i,3] = o4
        #mout[i,4] = o5
        #mout[i,5] = o6
        #mout[i,6] = o7
        index = i*7
        #out[index] = o1
        #out[index+1] = o2
        #out[index+2] = o3
        #out[index+3] = o4
        #out[index+4] = o5
        #out[index+5] = o6
        #out[index+6] = o7
        out[index] = out1[0]
        out[index+1] = out1[1]
        out[index+2] = out1[2]
        out[index+3] = out1[3]
        out[index+4] = out1[4]
        out[index+5] = out1[5]
        out[index+6] = out1[6]
	
    #free(out)

    return out
    #return mout

#cpdef void testgauss2d3(double[:] x, double[:] y, double[:] pars, int nderiv):
cpdef double[:,:] testgauss2d3(double[:] x, double[:] y, double[:] pars, int nderiv):
#cpdef void testgauss2d3(double[:] x, double[:] y, double[:] pars, int nderiv):
    out = gauss2d3(x,y,pars,nderiv)

    cdef long npix = len(x)
    cdef long i,j
    cdef double[:,:] npout = np.zeros((npix,7),float)
    for i in range(npix):
        #index = i*7
        for j in range(7):
            npout[i,j] = out[i*7+j]
            print(npout[i,j])
    free(out)
    #PyMem_Free(out)

    return npout

cpdef double[:] gaussian2d_integrate(double x, double y, double[:] pars, int nderiv, int osamp):
    cdef double theta,cost2,sint2
    cdef double xsig2,ysig2,a,b,c,g,xdiff,ydiff,xdiff2,ydiff2,x0,y0,dd,dx,dy
    cdef int nx,ny,col,row,hosamp,i
    #cdef double[:] x2,y2,xdiff,ydiff,g
    cdef double dg_dA,dg_dx_mean,dg_dy_mean,dg_dxsig,dg_dysig,dg_dtheta
    cdef double cost,sint,xsig3,ysig3,da_dxsig,db_dxsig,dc_dxsig
    cdef double da_dysig,db_dysig,dc_dysig
    cdef double da_dtheta,db_dtheta,dc_dtheta


    #osamp2 = float(osamp)**2
    #nx = x.size
    #dx = (np.arange(osamp).astype(float)+1)/osamp-(1/(2*osamp))-0.5
    #dx2 = np.tile(dx,(osamp,1))
    #x2 = np.tile(x,(osamp,osamp,1)) + np.tile(dx2.T,(nx,1,1)).T
    #y2 = np.tile(y,(osamp,osamp,1)) + np.tile(dx2,(nx,1,1)).T    
    
    # pars = [amplitude, x0, y0, xsigma, ysigma, theta]
    x0 = pars[1]
    y0 = pars[2]
    theta = pars[5]
    cost2 = cos(theta) ** 2
    sint2 = sin(theta) ** 2
    sin2t = sin(2. * theta)
    xsig = pars[3]
    ysig = pars[4]
    xsig2 = pars[3] ** 2
    ysig2 = pars[4] ** 2
    #xdiff = x2 - pars[1]
    #ydiff = y2 - pars[2]
    a = 0.5 * ((cost2 / xsig2) + (sint2 / ysig2))
    b = 0.5 * ((sin2t / xsig2) - (sin2t / ysig2))
    c = 0.5 * ((sint2 / xsig2) + (cost2 / ysig2))

    nsamp = osamp*osamp
    cdef double[16] dxarr = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    cdef double[16] dyarr = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    if osamp==1:
        dd = 0.0
        dd0 = 0.0
        dxarr[0] = 0.0
        dyarr[0] = 0.0
    elif osamp==2:
        dd = 0.50
        dd0 = -0.25
        dxarr[0:4] = [-0.25,-0.25,
                       0.25,0.25]
        dyarr[0:4] = [-0.25,0.25,
                      -0.25,0.25]
    elif osamp==3:
        dd = 0.3333
        dd0 = -0.3333
        dxarr[0:9] = [-0.3333,-0.3333,-0.3333,
                      0.0,0.0,0.0,
                      0.3333,0.3333,0.3333]
        dyarr[0:9] = [-0.3333,0.0,0.3333,
                      -0.3333,0.0,0.3333,
                      -0.3333,0.0,0.3333]
    elif osamp==4:
        dd = 0.25
        dd0 = -0.375
        dxarr[0:16] = [-0.375,-0.375,-0.375,-0.375,
	               -0.125,-0.125,-0.125,-0.125,
                       0.125,0.125,0.125,0.125,
                       0.375,0.375,0.375,0.375]
        dyarr[0:16] = [-0.375,-0.125,0.125,0.375,
                       -0.375,-0.125,0.125,0.375,
                       -0.375,-0.125,0.125,0.375,
                       -0.375,-0.125,0.125,0.375]

    #cdef double g = 0.0
    g = 0.0
    #cdef double[7] out = [0,0,0,0,0,0,0]
    #cdef double[6] derivative = [0,0,0,0,0,0]
    out = cvarray(shape=(7,),itemsize=sizeof(double),format="d")
    cdef double[:] mout = out
    for i in range(7):
        mout[i] = 0.0
    hosamp = osamp//2
    dg_dA = 0.0
    dg_dx_mean = 0.0
    dg_dy_mean = 0.0
    dg_dxsig = 0.0
    dg_dysig = 0.0
    dg_dtheta = 0.0
    for i in range(nsamp):
        col = i // osamp
        row = i % osamp
        dx = col*dd+dd0
        dy = row*dd+dd0
        #xdiff = (x+dx)-x0
        #ydiff = (y+dy)-y0
        #dx = (col-hosamp)*dd
        #dy = (row-hosamp)*dd
        xdiff = (x+dxarr[i])-x0
        ydiff = (y+dyarr[i])-y0
        xdiff2 = xdiff*xdiff
        ydiff2 = ydiff*ydiff
        g1 = pars[0] * np.exp(-((a * xdiff ** 2) + (b * xdiff * ydiff) +
                                (c * ydiff ** 2)))
        #print(i,g,col,row,dx,dy,dxarr[i],dyarr[i])
        mout[0] += g1

        # Compute derivative as well
        if nderiv>=1:
            dg_dA = g1 / pars[0]
            mout[1] += dg_dA
        if nderiv>=2:        
            dg_dx_mean = g1 * ((2. * a * xdiff) + (b * ydiff))
            mout[2] += dg_dx_mean
        if nderiv>=3:
            dg_dy_mean = g1 * ((b * xdiff) + (2. * c * ydiff))
            mout[3] += dg_dy_mean
        if nderiv>=4:
            cost = cos(theta)
            sint = sin(theta)
            xsig3 = xsig ** 3
            da_dxsig = -cost2 / xsig3
            db_dxsig = -sin2t / xsig3
            dc_dxsig = -sint2 / xsig3        
            dg_dxsig = g1 * (-(da_dxsig * xdiff2 +
                                   db_dxsig * xdiff * ydiff +
                                   dc_dxsig * ydiff2))
            mout[4] += dg_dxsig
        if nderiv>=5:
            ysig3 = ysig ** 3            
            da_dysig = -sint2 / ysig3
            db_dysig = sin2t / ysig3
            dc_dysig = -cost2 / ysig3        
            dg_dysig = g1 * (-(da_dysig * xdiff2 +
                                   db_dysig * xdiff * ydiff +
                                   dc_dysig * ydiff2))
            mout[5] += dg_dysig
        if nderiv>=6:
            cos2t = cos(2. * theta)            
            da_dtheta = (sint * cost * ((1. / ysig2) - (1. / xsig2)))
            db_dtheta = (cos2t / xsig2) - (cos2t / ysig2)
            dc_dtheta = -da_dtheta        
            dg_dtheta = g1 * (-(da_dtheta * xdiff2 +
                                db_dtheta * xdiff * ydiff +
                                dc_dtheta * ydiff2))
            mout[6] += dg_dtheta

    for i in range(7):
        mout[i] /= nsamp   # take average

    return mout

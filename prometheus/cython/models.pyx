# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
# cython: cdivision=True
# cython: binding=False
# cython: inter_types=True
# cython: cpow=True

import cython
cimport cython
import numpy as np
cimport numpy as np
from cython.view cimport array as cvarray
from cpython cimport array
from scipy.special import gamma, gammaincinv, gammainc

from libc.math cimport exp,sqrt,atan2,pi,NAN,log,log10,abs,pow
from libcpp cimport bool
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

#cdef extern from "complex.h":
#    complex double cpow(complex double base, complex double exponent)

#include <time.h>

cdef extern from "math.h":
    double sin(double x)
    double cos(double x)
    #double atan2(double x)

#cimport utils

from libc.time cimport time, time_t, clock, clock_t, CLOCKS_PER_SEC
from cpython.datetime cimport datetime

cpdef double get_current_time():
    #cdef time_t rawtime
    #cdef datetime dt
    #std::time_t time_value = std::time(nullptr); // Get current time
    #double double_time = static_cast<double>(time_value); // Convert to double

    #time(&rawtime)
    #print(rawtime)

    #cdef double double_time = static_cast<double>(rawtime)

    #dt = datetime.fromtimestamp(rawtime)
    #return double_time

    #clock_t start_t, end_t;
    #double total_t;
    #int i;
    #start_t = clock();
    #end_t = clock();
    #total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;

    cdef clock_t start_t, end_t
    cdef double total_t
    start_t = clock()
    end_t = clock()
    total_t = <double>(end_t - start_t) / CLOCKS_PER_SEC
    return total_t


#cpdef void gettime():
#  #cdef time_t rawtime
#  #cdef struct tm * timeinfo
#
#  cdef time_t now = time(0)
#
#  #time ( &rawtime )
#  #timeinfo = localtime ( &rawtime )
#  #printf( "Current local time and date: %s", asctime (timeinfo) )


cpdef int sum1i(int[:] array):
    cdef int sm
    n = len(array)
    sm = 0
    for i in range(n):
        sm = sm + array[i]
    return sm

cpdef long sum1l(long[:] array):
    cdef long sm
    n = len(array)
    sm = 0
    for i in range(n):
        sm = sm + array[i]
    return sm

cpdef float sum1f(float[:] array):
    cdef float sm
    n = len(array)
    sm = 0
    for i in range(n):
        sm += array[i]
    return sm

cpdef double sum1d(double[:] array):
    cdef double sm
    n = len(array)
    sm = 0
    for i in range(n):
        sm += array[i]
    return sm

cpdef double sum1dpy(double[:] array):
    cdef double sm
    sm = np.sum(array)
    return sm

cpdef double gaussfwhm(double[:,:] im):
    """
    Use the Gaussian equation Area
    Volume = A*2*pi*sigx*sigy
    to estimate the FWHM.
    """
    cdef double volume,ht,sigma,fwhm
    volume = np.sum(im)
    ht = np.max(im)
    sigma = sqrt(volume/(ht*2*pi))
    fwhm = 2.35*sigma
    return fwhm


# cpdef contourfwhm(im):
#     """                                                                                         
#     Measure the FWHM of a PSF or star image using contours.                                     
                                                                                                
#     Parameters                                                                                  
#     ----------                                                                                  
#     im : numpy array                                                                            
#      The 2D image of a star.                                                                    
                                                                                                
#     Returns                                                                                     
#     -------                                                                                     
#     fwhm : float                                                                                
#        The full-width at half maximum.                                                          
                                                                                                
#     Example                                                                                     
#     -------                                                                                     
                                                                                                
#     fwhm = contourfwhm(im)                                                                      
                                                                                                
#     """
#     # get contour at half max and then get average radius                                       
#     ny,nx = im.shape
#     xcen = nx//2
#     ycen = ny//2
#     xx,yy = utils.meshgrid(np.arange(nx)-nx//2,np.arange(ny)-ny//2)
#     rr = np.sqrt(xx**2+yy**2)

#     # Get half-flux radius                                                                      
#     hfrad = hfluxrad(im)
#     # mask the image to only 2*half-flux radius                                                 
#     mask = (rr<2*hfrad)

#     # Find contours at a constant value of 0.5                                                  
#     contours = measure.find_contours(im*mask, 0.5*np.max(im))
#     # If there are multiple contours, find the one that                                         
#     #   encloses the center                                                                     
#     if len(contours)>1:
#         for i in range(len(contours)):
#             x1 = contours[i][:,0]
#             y1 = contours[i][:,1]
#             inside = coords.isPointInPolygon(x1,y1,xcen,ycen)
#             if inside:
#                 contours = contours[i]
#                 break
#     else:
#         contours = contours[0]   # first level                                                  
#     xc = contours[:,0]
#     yc = contours[:,1]
#     r = np.sqrt((xc-nx//2)**2+(yc-ny//2)**2)
#     fwhm = np.mean(r)*2
#     return fwhm

# cpdef imfwhm(im):
#     """                                                                                         
#     Measure the FWHM of a PSF or star image.                                                    
                                                                                                
#     Parameters                                                                                  
#     ----------                                                                                  
#     im : numpy array                                                                            
#       The image of a star.                                                                      
                                                                                                
#     Returns                                                                                     
#     -------                                                                                     
#     fwhm : float                                                                                
#       The full-width at half maximum of the star.                                               
                                                                                                
#     Example                                                                                     
#     -------                                                                                     
                                                                                                
#     fwhm = imfwhm(im)                                                                           
                                                                                                
#     """
#     ny,nx = im.shape
#     xx,yy = utils.meshgrid(np.arange(nx)-nx//2,np.arange(ny)-ny//2)
#     rr = np.sqrt(xx**2+yy**2)
#     centerf = im[ny//2,nx//2]
#     si = np.argsort(rr.ravel())
#     rsi = rr.ravel()[si]
#     fsi = im.ravel()[si]
#     ind, = np.where(fsi<0.5*centerf)
#     bestr = np.min(rsi[ind])
#     bestind = ind[np.argmin(rsi[ind])]
#     # fit a robust line to the neighboring points                                               
#     gd, = np.where(np.abs(rsi-bestr) < 1.0)
#     coef,absdev = ladfit.ladfit(rsi[gd],fsi[gd])
#     # where does the line cross y=0.5                                                           
#     bestr2 = (0.5-coef[0])/coef[1]
#     fwhm = 2*bestr2
#     return fwhm



# cpdef numba_linearinterp(binim,fullim,binsize):
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
                    
#     # do the edges
#     #for i in range(binsize):
#     #    fullim[i,:] = fullim[binsize,:]
#     #    fullim[:,i] = fullim[:,binsize]
#     #    fullim[-i,:] = fullim[-binsize,:]
#     #    fullim[:,-i] = fullim[:,-binsize]
#     # do the corners
#     #fullim[:binsize,:binsize] = binsize[0,0]
#     #fullim[-binsize:,:binsize] = binsize[-1,0]
#     #fullim[:binsize,-binsize:] = binsize[0,-1]
#     #fullim[-binsize:,-binsize:] = binsize[-1,-1]

#     return fullim


cpdef int[:] checkbounds(double[:] pars, double[:,:] bounds):
    """ Check the parameters against the bounds."""
    # 0 means it's fine
    # 1 means it's beyond the lower bound
    # 2 means it's beyond the upper bound
    cdef int npars
    cdef int[:] check
    npars = len(pars)
    check = np.zeros(npars,np.int32)
    for i in range(npars):
        if pars[i] <= bounds[i,0]:
            check[i] = 1
        if pars[i] >= bounds[i,1]:
            check[i] = 2
    return check


cpdef double[:] limbounds(double[:] pars, double[:,:] bounds):
    """ Limit the parameters to the boundaries."""
    cdef double[:] outpars
    npars = len(pars)
    outpars = np.zeros(len(pars),float)
    for i in range(npars):
        p = max(p[i],bounds[i,0])
        p = min(p,bounds[i,1])
        outpars[i] = p
    return outpars


cpdef double[:] limsteps(double[:] steps, double[:] maxsteps):
    """ Limit the parameter steps to maximum step sizes."""
    cdef double[:] outsteps
    cdef double ostep
    npars = len(steps)
    outsteps = np.zeros(npars,float)
    for i in range(npars):
        sgn = np.sign(steps[i])
        ostep = min(np.abs(steps[i]),maxsteps[i])
        ostep *= sgn
        outsteps[i] = ostep
    return outsteps


# cpdef double[:] newlsqpars(double[:] pars, double[:] steps, double[:,:] bounds, double[:] maxsteps):
#     """ Return new parameters that fit the constraints."""
#     # Limit the steps to maxsteps
#     cdef double[:] limited_steps,lbounds,ubounds,newsteps,newparams
#     cdef int[:] check, newcheck
#     limited_steps = limsteps(steps,maxsteps)
        
#     # Make sure that these don't cross the boundaries
#     lbounds = bounds[:,0]
#     ubounds = bounds[:,1]
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
#     newparams = pars + newsteps
            
#     # Make sure to limit them to the boundaries
#     check = checkbounds(newparams,bounds)
#     badpars = (check!=0)
#     if np.sum(badpars)>0:
#         # add a tiny offset so it doesn't fit right on the boundary
#         newparams = np.minimum(np.maximum(newparams,lbounds+1e-30),ubounds-1e-30)
#     return newparams


# cpdef newbestpars(bestpars,dbeta):
#     """ Get new pars from offsets."""
#     newpars = np.zeros(3,float)
#     maxchange = 0.5
#     # Amplitude
#     ampmin = bestpars[0]-maxchange*np.abs(bestpars[0])
#     ampmin = np.maximum(ampmin,0)
#     ampmax = bestpars[0]+np.abs(maxchange*bestpars[0])
#     newamp = utils.clip(bestpars[0]+dbeta[0],ampmin,ampmax)
#     newpars[0] = newamp
#     # Xc, maxchange in pixels
#     xmin = bestpars[1]-maxchange
#     xmax = bestpars[1]+maxchange
#     newx = utils.clip(bestpars[1]+dbeta[1],xmin,xmax)
#     newpars[1] = newx
#     # Yc
#     ymin = bestpars[2]-maxchange
#     ymax = bestpars[2]+maxchange
#     newy = utils.clip(bestpars[2]+dbeta[2],ymin,ymax)
#     newpars[2] = newy
#     return newpars


# cpdef starbbox(coords,imshape,radius):
#     """                                                                                         
#     Return the boundary box for a star given radius and image size.                             
                                                                                                
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

#     # Star coordinates                                                                          
#     xcen,ycen = coords
#     ny,nx = imshape   # python images are (Y,X)                                                 
#     xlo = np.maximum(int(np.floor(xcen-radius)),0)
#     xhi = np.minimum(int(np.ceil(xcen+radius+1)),nx)
#     ylo = np.maximum(int(np.floor(ycen-radius)),0)
#     yhi = np.minimum(int(np.ceil(ycen+radius+1)),ny)

#     return BoundingBox(xlo,xhi,ylo,yhi)

# cpdef bbox2xy(bbox):
#     """                                                                                         
#     Convenience method to convert boundary box of X/Y limits to 2-D X and Y arrays.  The upper limits
#     are EXCLUSIVE following the python convention.                                              
                                                                                                
#     Parameters                                                                                  
#     ----------                                                                                  
#     bbox : BoundingBox object                                                                   
#       A BoundingBox object cpdefining a rectangular region of an image.                           
                                                                                                
#     Returns                                                                                     
#     -------                                                                                     
#     x : numpy array                                                                             
#       The 2D array of X-values of the bounding box region.                                      
#     y : numpy array                                                                             
#       The 2D array of Y-values of the bounding box region.                                      
                                                                                                
#     Example                                                                                     
#     -------                                                                                     
                                                                                                
#     x,y = bbox2xy(bbox)                                                                         
                                                                                                
#     """
#     if isinstance(bbox,BoundingBox):
#         x0,x1 = bbox.xrange
#         y0,y1 = bbox.yrange
#     else:
#         x0,x1 = bbox[0]
#         y0,y1 = bbox[1]
#     dx = np.arange(x0,x1)
#     nxpix = len(dx)
#     dy = np.arange(y0,y1)
#     nypix = len(dy)
#     # Python images are (Y,X)                                                                   
#     x = dx.reshape(1,-1)+np.zeros(nypix,int).reshape(-1,1)   # broadcasting is faster           
#     y = dy.reshape(-1,1)+np.zeros(nxpix,int)
#     return x,y


# ###################################################################
# # Analytical PSF models


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


# ####### GAUSSIAN ########


cpdef double gaussian2d_flux(double[:] pars):
    """
    Return the total flux (or volume) of a 2D Gaussian.

    Parameters
    ----------
    pars : numpy array
       Parameter list. pars = [amplitude, x0, y0, xsigma, ysigma, theta]

    Returns
    -------
    flux : float
       Total flux or volumne of the 2D Gaussian.
    
    Example
    -------

    flux = gaussian2d_flux(pars)

    """
    cdef double amp,xsig,ysig,volume
    # Volume is 2*pi*A*sigx*sigy
    amp = pars[0]
    xsig = pars[3]
    ysig = pars[4]    
    volume = 2*pi*amp*xsig*ysig
    return volume


cpdef double gaussian2d_fwhm(double[:] pars):
    """
    Return the FWHM of a 2D Gaussian.

    Parameters
    ----------
    pars : numpy array
       Parameter list. pars = [amplitude, x0, y0, xsigma, ysigma, theta]

    Returns
    -------
    fwhm : float
       The full-width at half maximum of the Gaussian.
    
    Example
    -------

    fwhm = gaussian2d_fwhm(pars)

    """
    cdef double fwhm

    # pars = [amplitude, x0, y0, xsig, ysig, theta]

    # xdiff = x-x0
    # ydiff = y-y0
    # f(x,y) = A*exp(-0.5 * (a*xdiff**2 + b*xdiff*ydiff + c*ydiff**2))

    xsig = pars[3]
    ysig = pars[4]

    # The mean radius of an ellipse is: (2a+b)/3
    #sig_major = np.max(np.array([xsig,ysig]))
    #sig_minor = np.min(np.array([xsig,ysig]))
    sig_major = max([xsig,ysig])
    sig_minor = min([xsig,ysig])
    mnsig = (2.0*sig_major+sig_minor)/3.0
    # Convert sigma to FWHM
    # FWHM = 2*sqrt(2*ln(2))*sig ~ 2.35482*sig
    fwhm = mnsig*2.35482

    return fwhm


cdef void agaussian2d2(double* x, double* y, int npix, double[11] pars, int nderiv, int osamp, double* out):
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
    cdef double amp,xc,yc,xsig,ysig,theta,cxx,cyy,cxy
    cdef double x1,y1
    cdef int i,j,index

    amp = pars[0]
    xc = pars[1]
    yc = pars[2]
    xsig = pars[3]
    ysig = pars[4]
    theta = pars[5]
    cxx,cyy,cxy = gauss_abt2cxy(xsig,ysig,theta)

    cdef double allpars[9]
    #cdef double* allpars = <double*>malloc(9 * sizeof(double))
    allpars[0] = amp
    allpars[1] = xc
    allpars[2] = yc
    allpars[3] = xsig
    allpars[4] = ysig
    allpars[5] = theta
    allpars[6] = cxx
    allpars[7] = cyy
    allpars[8] = cxy

    #npix = len(x)

    # 2D arrays
    #out = cvarray(shape=(npix,7),itemsize=sizeof(double),format="d")
    #cdef double[:,:] mout = out

    cdef double *out1 = <double*>malloc(7 * sizeof(double))

    #cdef long ncols = 7

    # Loop over the points
    for i in range(npix):
        index = i * 7
        x1 = x[i]
        y1 = y[i]
        gaussian2d_integrate(x1,y1,allpars,nderiv,osamp,out1)
        for j in range(nderiv+1):
            #out[i,j] = out1[j]
            out[index+j] = out1[j]

    free(out1)

cpdef void testgaussian2d2(double[:] xin, double[:] yin, double[:] parsin, int nderiv, int osamp):
    cdef int npix = len(xin)
    cdef double* x = <double*>malloc(npix * sizeof(double))
    cdef double* y = <double*>malloc(npix * sizeof(double))
    cdef double* out = <double*>malloc(7 * npix * sizeof(double))
    cdef double pars[11]
    for i in range(npix):
        x[i] = xin[i]
        y[i] = yin[i]
    for i in range(6):
        pars[i] = parsin[i]

    cdef clock_t start_t, end_t
    cdef double total_t
    start_t = clock()

    agaussian2d2(x,y,npix,pars,nderiv,osamp,out)

    end_t = clock()
    total_t = <double>(end_t - start_t) / CLOCKS_PER_SEC
    print(total_t)
    #return total_t

    free(out)
    

cdef double[:,:] agaussian2d(double[:] x, double[:] y, double[:] pars, int nderiv, int osamp):
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
    cdef double amp,xc,yc,xsig,ysig,theta,cxx,cyy,cxy
    cdef double x1,y1
    cdef long i,j,npix,index

    amp = pars[0]
    xc = pars[1]
    yc = pars[2]
    xsig = pars[3]
    ysig = pars[4]
    theta = pars[5]
    if len(pars)==6:
        cxx,cyy,cxy = gauss_abt2cxy(xsig,ysig,theta)
    else:
        cxx = pars[6]
        cyy = pars[7]
        cxy = pars[8]

    cdef double allpars[9]
    #cdef double* allpars = <double*>malloc(9 * sizeof(double))
    allpars[0] = amp
    allpars[1] = xc
    allpars[2] = yc
    allpars[3] = xsig
    allpars[4] = ysig
    allpars[5] = theta
    allpars[6] = cxx
    allpars[7] = cyy
    allpars[8] = cxy

    npix = len(x)

    # 2D arrays
    out = cvarray(shape=(npix,7),itemsize=sizeof(double),format="d")
    cdef double[:,:] mout = out

    cdef double *out1 = <double*>malloc(7 * sizeof(double))

    # Loop over the points
    for i in range(npix):
        x1 = x[i]
        y1 = y[i]
        gaussian2d_integrate(x1,y1,allpars,nderiv,osamp,out1)
        for j in range(nderiv+1):
            mout[i,j] = out1[j]

    free(out1)

    return mout


cpdef void testgaussian2d(double[:] x, double[:] y, double[:] pars, int nderiv, int osamp):
    out = agaussian2d(x,y,pars,nderiv,osamp)
    #free(out)


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
    for i in range(npix):
        xdiff[i] = x[i] - pars[1]
        ydiff[i] = y[i] - pars[2]
    # g = amp * np.exp(-0.5*((a * xdiff**2) + (b * xdiff * ydiff) +
    #                        (c * ydiff**2)))

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
    # amp = 1/(asemi*bsemi*2*pi)
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

cdef void gaussian2d_integrate(double x, double y, double[9] pars, int nderiv, int osamp, double* out):
    cdef double theta,cost2,sint2,amp
    cdef double xsig2,ysig2,a,b,c,u,v,u2,v2,x0,y0,dx,dy
    cdef int nx,ny,col,row,nsamp,hosamp,i
    #cdef double[:] x2,y2,u,v,g
    cdef double dg_dA,dg_dx_mean,dg_dy_mean,dg_dxsig,dg_dysig,dg_dtheta
    cdef double cost,sint,xsig3,ysig3,da_dxsig,db_dxsig,dc_dxsig
    cdef double da_dysig,db_dysig,dc_dysig
    cdef double da_dtheta,db_dtheta,dc_dtheta

    # pars = [amplitude, x0, y0, xsigma, ysigma, theta]
    amp = pars[0]
    x0 = pars[1]
    y0 = pars[2]
    xsig = pars[3]
    ysig = pars[4]
    theta = pars[5]
    cxx = pars[6]
    cyy = pars[7]
    cxy = pars[8]
    cost = cos(theta)
    sint = sin(theta)
    cost2 = cost ** 2
    sint2 = sint ** 2
    sin2t = sin(2. * theta)
    xsig2 = pars[3] ** 2
    ysig2 = pars[4] ** 2
    #a = 0.5 * ((cost2 / xsig2) + (sint2 / ysig2))
    #b = 0.5 * ((sin2t / xsig2) - (sin2t / ysig2))
    #c = 0.5 * ((sint2 / xsig2) + (cost2 / ysig2))

    u = x-x0
    v = y-y0
    cdef double f = 0.0
    if osamp < 1:
        f = exp(-((cxx * u ** 2) + (cxy * u * v) +
                  (cyy * v ** 2)))

    # Automatically determine the oversampling
    # These are the thresholds that daophot uses
    # from the IRAF daophot version in
    # noao/digiphot/daophot/daolib/profile.x
    if osamp < 1:
        if (f >= 0.046):
            osamp = 4
        elif (f >= 0.0022):
            osamp = 3
        elif (f >= 0.0001):
            osamp = 2
        elif (f >= 1.0e-10):
            osamp = 1

    nsamp = osamp*osamp
    cdef double dd = 0.0
    cdef double dd0 = 0.0
    # dx = (np.arange(osamp).astype(float)+1)/osamp-(1/(2*osamp))-0.5
    if osamp>1:
        dd = 1/float(osamp)
        dd0 = 1/(2*float(osamp))-0.5  

    cdef double g = 0.0
    for i in range(7):
        out[i] = 0.0
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
        u = (x+dx)-x0
        v = (y+dy)-y0
        u2 = u*u
        v2 = v*v
        g = amp * exp(-((cxx * u ** 2) + (cxy * u * v) +
                        (cyy * v ** 2)))
        out[0] += g

        # Compute derivative as well
        if nderiv>=1:
            dg_dA = g / amp
            out[1] += dg_dA
        if nderiv>=2:        
            dg_dx_mean = g * ((2. * cxx * u) + (cxy * v))
            out[2] += dg_dx_mean
        if nderiv>=3:
            dg_dy_mean = g * ((cxy * u) + (2. * cyy * v))
            out[3] += dg_dy_mean
        if nderiv>=4:
            cost = cos(theta)
            sint = sin(theta)
            xsig3 = xsig ** 3
            da_dxsig = -cost2 / xsig3
            db_dxsig = -sin2t / xsig3
            dc_dxsig = -sint2 / xsig3        
            dg_dxsig = g * (-(da_dxsig * u2 +
                                   db_dxsig * u * v +
                                   dc_dxsig * v2))
            out[4] += dg_dxsig
        if nderiv>=5:
            ysig3 = ysig ** 3            
            da_dysig = -sint2 / ysig3
            db_dysig = sin2t / ysig3
            dc_dysig = -cost2 / ysig3        
            dg_dysig = g * (-(da_dysig * u2 +
                                   db_dysig * u * v +
                                   dc_dysig * v2))
            out[5] += dg_dysig
        if nderiv>=6:
            cos2t = cos(2. * theta)            
            da_dtheta = (sint * cost * ((1. / ysig2) - (1. / xsig2)))
            db_dtheta = (cos2t / xsig2) - (cos2t / ysig2)
            dc_dtheta = -da_dtheta        
            dg_dtheta = g * (-(da_dtheta * u2 +
                                db_dtheta * u * v +
                                dc_dtheta * v2))
            out[6] += dg_dtheta

    if osamp>1:
        for i in range(nderiv+1):
            out[i] /= nsamp   # take average


# #
# cpdef gaussian2dfit(im,err,ampc,xc,yc,verbose):
#     """
#     Fit a single Gaussian 2D model to data.

#     Parameters
#     ----------
#     im : numpy array
#        Flux array.  Can be 1D or 2D array.
#     err : numpy array
#        Uncertainty array of im.  Same dimensions as im.
#     ampc : float
#        Initial guess of amplitude.
#     xc : float
#        Initial guess of central X coordinate.
#     yc : float
#        Initial guess of central Y coordinate.
#     verbose : bool
#        Verbose output to the screen.

#     Returns
#     -------
#     pars : numpy array
#        Best fit pararmeters.
#     perror : numpy array
#        Uncertainties in pars.
#     pcov : numpy array
#        Covariance matrix.
#     flux : float
#        Best fit flux.
#     fluxerr : float
#        Uncertainty in flux.
    
#     Example
#     -------

#     pars,perror,cov,flux,fluxerr = gaussian2dfit(im,err,1,100.0,5.5,6.5,False)

#     """

#     # xc/yc are with respect to the image origin (0,0)
    
#     # Solve for x, y, amplitude and asemi/bsemi/theta

#     maxiter = 10
#     minpercdiff = 0.5
    
#     ny,nx = im.shape
#     im1d = im.ravel()

#     x2d,y2d = utils.meshgrid(np.arange(nx),np.arange(ny))
#     x1d = x2d.ravel()
#     y1d = y2d.ravel()
    
#     wt = 1/err**2
#     wt1d = wt.ravel()

#     asemi = 2.5
#     bsemi = 2.4
#     theta = 0.1

#     # theta in radians
    
#     # Initial values
#     bestpar = np.zeros(6,float)
#     bestpar[0] = ampc
#     bestpar[1] = xc
#     bestpar[2] = yc
#     bestpar[3] = asemi
#     bestpar[4] = bsemi
#     bestpar[5] = theta
    
#     # Iteration loop
#     maxpercdiff = 1e10
#     niter = 0
#     while (niter<maxiter and maxpercdiff>minpercdiff):
#         model,deriv = agaussian2d(x1d,y1d,bestpar,6)
#         resid = im1d-model
#         dbeta = utils.qr_jac_solve(deriv,resid,weight=wt1d)
        
#         if verbose:
#             print(niter,bestpar)
#             print(dbeta)
        
#         # Update parameters
#         last_bestpar = bestpar.copy()
#         # limit the steps to the maximum step sizes and boundaries
#         #if bounds is not None or maxsteps is not None:
#         #    bestpar = newpars(bestpar,dbeta,bounds,maxsteps)
#         bounds = np.zeros((6,2),float)
#         bounds[:,0] = [0.00, 0, 0, 0.1, 0.1, -180]
#         bounds[:,1] = [1e30,nx,ny, nx//2, ny//2, 180]
#         maxsteps = np.zeros(6,float)
#         maxsteps[:] = [0.5*bestpar[0],0.5,0.5,0.5,0.5,2.0]
#         bestpar = newlsqpars(bestpar,dbeta,bounds,maxsteps)
        
#         # Check differences and changes
#         diff = np.abs(bestpar-last_bestpar)
#         denom = np.maximum(np.abs(bestpar.copy()),0.0001)
#         percdiff = diff.copy()/denom*100  # percent differences
#         maxpercdiff = np.max(percdiff)
#         chisq = np.sum((im1d-model)**2 * wt1d)/(nx*ny)
#         if verbose:
#             print('chisq=',chisq)
#         #if verbose:
#         #    print(niter,percdiff,chisq)
#         #    print()
#         last_dbeta = dbeta
#         niter += 1

#     model,deriv = agaussian2d(x1d,y1d,bestpar,6)
#     resid = im1d-model
    
#     # Get covariance and errors
#     cov = utils.jac_covariance(deriv,resid,wt1d)
#     perror = np.sqrt(np.diag(cov))

#     # Now get the flux, multiply by the volume of the Gaussian
#     asemi,bsemi,theta = bestpar[3],bestpar[4],bestpar[5]
#     gvolume = asemi*bsemi*2*np.pi
#     flux = bestpar[0]*gvolume
#     fluxerr = perror[0]*gvolume

#     # USE GAUSSIAN_FLUX
#     # pars = [amplitude, x0, y0, xsigma, ysigma, theta]
#     flux = gaussian2d_flux(bestpar)
#     fluxerr = perror[0]*(flux/bestpar[0]) 
    
#     return bestpar,perror,cov,flux,fluxerr


# ####### MOFFAT ########


cpdef double moffat2d_fwhm(double[:] pars):
    """
    Return the FWHM of a 2D Moffat function.

    Parameters
    ----------
    pars : numpy array
       Parameter list. pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta]

    Returns
    -------
    fwhm : float
       The full-width at half maximum of the Moffat.
    
    Example
    -------

    fwhm = moffat2d_fwhm(pars)

    """
    cdef double xsig,ysig,beta,sig_major,sig_minor,mnsig
    # [amplitude, x0, y0, xsig, ysig, theta, beta]
    # https://nbviewer.jupyter.org/github/ysbach/AO_2017/blob/master/04_Ground_Based_Concept.ipynb#1.2.-Moffat

    xsig = pars[3]
    ysig = pars[4]
    beta = pars[6]
    
    # The mean radius of an ellipse is: (2a+b)/3
    sig_major = np.max(np.array([xsig,ysig]))
    sig_minor = np.min(np.array([xsig,ysig]))
    mnsig = (2.0*sig_major+sig_minor)/3.0
    
    return 2.0 * np.abs(mnsig) * np.sqrt(2.0 ** (1.0/beta) - 1.0)


cpdef double moffat2d_flux(double[:] pars):
    """
    Return the total Flux of a 2D Moffat.

    Parameters
    ----------
    pars : numpy array
       Parameter list. pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta]

    Returns
    -------
    flux : float
       Total flux or volumne of the 2D Moffat.
    
    Example
    -------

    flux = moffat2d_flux(pars)

    """
    cdef double amp,xsig,ysig,beta,volume

    # [amplitude, x0, y0, xsig, ysig, theta, beta]
    # Volume is 2*pi*A*sigx*sigy
    # area of 1D moffat function is pi*alpha**2 / (beta-1)
    # maybe the 2D moffat volume is (xsig*ysig*pi**2/(beta-1))**2

    amp = pars[0]
    xsig = pars[3]
    ysig = pars[4]
    beta = pars[6]

    # This worked for beta=2.5, but was too high by ~1.05-1.09 for beta=1.5
    #volume = amp * xstd*ystd*pi/(beta-1)
    volume = amp * xsig*ysig*pi/(beta-1)
    # what is the beta dependence?? linear is very close!

    # I think undersampling is becoming an issue at beta=3.5 with fwhm=2.78
    
    return volume



cpdef double[:,:] amoffat2d(double[:] x, double[:] y, double[:] pars, int nderiv, int osamp):
    """
    Two dimensional Moffat model function with x/y array inputs.
    
    Parameters
    ----------
    x : numpy array
      Array of X-values of points for which to compute the Moffat model
    y : numpy array
      Array of Y-values of points for which to compute the Moffat model.
    pars : numpy array
       Parameter list. pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta].
    nderiv : int
       The number of derivatives to return.

    Returns
    -------
    g : numpy array
      The Moffat model for the input x/y values and parameters.  Always
        returned as 1D raveled() array.
    derivative : numpy array
      Array derivatives of g relative to the input parameters.
        Always 2D [Npix,Nderiv] with the 1st dimension being the x/y arrays
        raveled() to 1D.

    Example
    -------

    g,derivative = amoffat2d(x,y,pars,3)

    """

    cdef double amp,xc,yc,xsig,ysig,theta,cxx,cyy,cxy,beta
    cdef double x1,y1
    cdef long i,j,npix,index

    amp = pars[0]
    xc = pars[1]
    yc = pars[2]
    xsig = pars[3]
    ysig = pars[4]
    theta = pars[5]
    beta = pars[6]
    if len(pars)==7:
        cxx,cyy,cxy = gauss_abt2cxy(xsig,ysig,theta)
    else:
        cxx = pars[7]
        cyy = pars[8]
        cxy = pars[9]

    cdef double allpars[10]
    #cdef double* allpars = <double*>malloc(9 * sizeof(double))
    allpars[0] = amp
    allpars[1] = xc
    allpars[2] = yc
    allpars[3] = xsig
    allpars[4] = ysig
    allpars[5] = theta
    allpars[6] = beta
    allpars[7] = cxx
    allpars[8] = cyy
    allpars[9] = cxy

    npix = len(x)

    # 2D arrays
    out = cvarray(shape=(npix,8),itemsize=sizeof(double),format="d")
    cdef double[:,:] mout = out

    cdef double *out1 = <double*>malloc(8 * sizeof(double))

    # Loop over the points
    for i in range(npix):
        x1 = x[i]
        y1 = y[i]
        moffat2d_integrate(x1,y1,allpars,nderiv,osamp,out1)
        for j in range(nderiv+1):
            mout[i,j] = out1[j]

    free(out1)

    return mout


cdef void moffat2d_integrate(double x, double y, double[10] pars, int nderiv, int osamp, double* out):
    cdef double theta,cost2,sint2,amp,beta
    cdef double xsig2,ysig2,a,b,c,u,v,u2,v2,x0,y0,dx,dy
    cdef int nx,ny,col,row,nsamp,hosamp,i
    #cdef double[:] x2,y2,u,v,g
    cdef double dg_dA,dg_dx_mean,dg_dy_mean,dg_dxsig,dg_dysig,dg_dtheta,dg_dbeta
    cdef double cost,sint,xsig3,ysig3,da_dxsig,db_dxsig,dc_dxsig
    cdef double da_dysig,db_dysig,dc_dysig
    cdef double da_dtheta,db_dtheta,dc_dtheta

    # pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta, cxx, cyy, cxy]
    amp = pars[0]
    x0 = pars[1]
    y0 = pars[2]
    xsig = pars[3]
    ysig = pars[4]
    theta = pars[5]
    beta = pars[6]
    cxx = pars[7]
    cyy = pars[8]
    cxy = pars[9]
    sint = sin(theta)
    cost = cos(theta)
    cost2 = cost ** 2
    sint2 = sint ** 2
    sin2t = sin(2. * theta)
    xsig2 = xsig ** 2
    ysig2 = ysig ** 2

    u = x-x0
    v = y-y0
    cdef double f = 0.0
    if osamp < 1:
        f = exp(-((cxx * u ** 2) + (cxy * u * v) +
                  (cyy * v ** 2)))

    # Automatically determine the oversampling
    # These are the thresholds that daophot uses
    # from the IRAF daophot version in
    # noao/digiphot/daophot/daolib/profile.x
    if osamp < 1:
        if (f >= 0.046):
            osamp = 4
        elif (f >= 0.0022):
            osamp = 3
        elif (f >= 0.0001):
            osamp = 2
        elif (f >= 1.0e-10):
            osamp = 1

    nsamp = osamp*osamp
    cdef double dd = 0.0
    cdef double dd0 = 0.0
    # dx = (np.arange(osamp).astype(float)+1)/osamp-(1/(2*osamp))-0.5
    if osamp>1:
        dd = 1/float(osamp)
        dd0 = 1/(2*float(osamp))-0.5

    cdef double g = 0.0
    for i in range(8):
        out[i] = 0.0
    hosamp = osamp//2
    dg_dA = 0.0
    dg_dx_mean = 0.0
    dg_dy_mean = 0.0
    dg_dxsig = 0.0
    dg_dysig = 0.0
    dg_dtheta = 0.0
    dg_dbeta = 0.0
    for i in range(nsamp):
        col = i // osamp
        row = i % osamp
        dx = col*dd+dd0
        dy = row*dd+dd0
        u = (x+dx)-x0
        v = (y+dy)-y0
        u2 = u*u
        v2 = v*v

        rr_gg = (cxx*u**2 + cyy*v**2 + cxy*u*v)
        #g = amp * (1 + rr_gg) ** (-beta)
        g = amp / (1 + rr_gg) ** beta
        out[0] += g

        # Compute derivative as well
        if nderiv>=1:
            dg_dA = g / amp
            out[1] += dg_dA
        if nderiv>=2:
            dg_dx_mean = beta * g/(1+rr_gg) * ((2. * cxx * u) + (cxy * v))
            out[2] += dg_dx_mean
        if nderiv>=3:
            dg_dy_mean = beta * g/(1+rr_gg) * ((cxy * u) + (2. * cyy * v))
            out[3] += dg_dy_mean
        if nderiv>=4:
            xsig3 = xsig ** 3
            da_dxsig = -cost2 / xsig3
            db_dxsig = -sin2t / xsig3
            dc_dxsig = -sint2 / xsig3
            dg_dxsig = (-beta)*g/(1+rr_gg) * 2*(da_dxsig * u2 +
                                                db_dxsig * u * v +
                                                dc_dxsig * v2)
            out[4] += dg_dxsig
        if nderiv>=5:
            ysig3 = ysig ** 3
            da_dysig = -sint2 / ysig3
            db_dysig = sin2t / ysig3
            dc_dysig = -cost2 / ysig3
            dg_dysig = (-beta)*g/(1+rr_gg) * 2*(da_dysig * u2 +
                                                db_dysig * u * v +
                                                dc_dysig * v2)
            out[5] += dg_dysig
        if nderiv>=6 and xsig != ysig:
            cos2t = cos(2.0*theta)
            da_dtheta = (sint * cost * ((1. / ysig2) - (1. / xsig2)))
            db_dtheta = (cos2t / xsig2) - (cos2t / ysig2)
            dc_dtheta = -da_dtheta
            dg_dtheta = (-beta)*g/(1+rr_gg) * 2*(da_dtheta * u2 +
                                                 db_dtheta * u * v +
                                                 dc_dtheta * v2)
            out[6] += dg_dtheta
        if nderiv>=7:
            dg_dbeta = -g * log(1 + rr_gg)
            out[7] += dg_dbeta

    if osamp>1:
        for i in range(nderiv+1):
            out[i] /= nsamp   # take average


# cpdef moffat2dfit(im,err,ampc,xc,yc,verbose):
#     """
#     Fit a single Moffat 2D model to data.

#     Parameters
#     ----------
#     im : numpy array
#        Flux array.  Can be 1D or 2D array.
#     err : numpy array
#        Uncertainty array of im.  Same dimensions as im.
#     ampc : float
#        Initial guess of amplitude.
#     xc : float
#        Initial guess of central X coordinate.
#     yc : float
#        Initial guess of central Y coordinate.
#     verbose : bool
#        Verbose output to the screen.

#     Returns
#     -------
#     pars : numpy array
#        Best fit pararmeters.
#     perror : numpy array
#        Uncertainties in pars.
#     pcov : numpy array
#        Covariance matrix.
#     flux : float
#        Best fit flux.
#     fluxerr : float
#        Uncertainty in flux.
    
#     Example
#     -------

#     pars,perror,cov,flux,fluxerr = moffat2dfit(im,err,1,100.0,5.5,6.5,False)

#     """

#     # xc/yc are with respect to the image origin (0,0)
    
#     # Solve for x, y, amplitude and asemi/bsemi/theta

#     maxiter = 10
#     minpercdiff = 0.5
    
#     ny,nx = im.shape
#     im1d = im.ravel()

#     x2d,y2d = utils.meshgrid(np.arange(nx),np.arange(ny))
#     x1d = x2d.ravel()
#     y1d = y2d.ravel()
    
#     wt = 1/err**2
#     wt1d = wt.ravel()

#     asemi = 2.5
#     bsemi = 2.4
#     theta = 0.1
#     beta = 2.5

#     # theta in radians
    
#     # Initial values
#     bestpar = np.zeros(7,float)
#     bestpar[0] = ampc
#     bestpar[1] = xc
#     bestpar[2] = yc
#     bestpar[3] = asemi
#     bestpar[4] = bsemi
#     bestpar[5] = theta
#     bestpar[6] = beta
    
#     # Iteration loop
#     maxpercdiff = 1e10
#     niter = 0
#     while (niter<maxiter and maxpercdiff>minpercdiff):
#         model,deriv = amoffat2d(x1d,y1d,bestpar,7)
#         resid = im1d-model
#         dbeta = utils.qr_jac_solve(deriv,resid,weight=wt1d)
        
#         if verbose:
#             print(niter,bestpar)
#             print(dbeta)
        
#         # Update parameters
#         last_bestpar = bestpar.copy()
#         # limit the steps to the maximum step sizes and boundaries
#         #if bounds is not None or maxsteps is not None:
#         #    bestpar = newpars(bestpar,dbeta,bounds,maxsteps)
#         bounds = np.zeros((7,2),float)
#         bounds[:,0] = [0.00, 0, 0, 0.1, 0.1, -180, 0.1]
#         bounds[:,1] = [1e30,nx,ny, nx//2, ny//2, 180, 10]
#         maxsteps = np.zeros(7,float)
#         maxsteps[:] = [0.5*bestpar[0],0.5,0.5,0.5,0.5,2.0,0.5]
#         bestpar = newlsqpars(bestpar,dbeta,bounds,maxsteps)
        
#         # Check differences and changes
#         diff = np.abs(bestpar-last_bestpar)
#         denom = np.maximum(np.abs(bestpar.copy()),0.0001)
#         percdiff = diff.copy()/denom*100  # percent differences
#         maxpercdiff = np.max(percdiff)
#         chisq = np.sum((im1d-model)**2 * wt1d)/(nx*ny)
#         if verbose:
#             print('chisq=',chisq)
#         #if verbose:
#         #    print(niter,percdiff,chisq)
#         #    print()
#         last_dbeta = dbeta
#         niter += 1

#     model,deriv = amoffat2d(x1d,y1d,bestpar,7)
#     resid = im1d-model
    
#     # Get covariance and errors
#     cov = utils.jac_covariance(deriv,resid,wt1d)
#     perror = np.sqrt(np.diag(cov))

#     # Now get the flux, multiply by the volume of the Gaussian
#     #asemi,bsemi,theta = bestpar[3],bestpar[4],bestpar[5]
#     #gvolume = asemi*bsemi*2*np.pi
#     #flux = bestpar[0]*gvolume
#     #fluxerr = perror[0]*gvolume

#     # USE MOFFAT_FLUX
#     # pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta]
#     flux = moffat2d_flux(bestpar)
#     fluxerr = perror[0]*(flux/bestpar[0]) 
    
#     return bestpar,perror,cov,flux,fluxerr


# ####### PENNY ########


cpdef double penny2d_fwhm(double[:] pars):
    """
    Return the FWHM of a 2D Penny function.

    Parameters
    ----------
    pars : numpy array
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
    cdef double amp,xsig,ysig,relamp,sigma,beta
    cdef double sig_major,sig_minor,mnsig,gfwhm
    cdef double mnfwhm,hwhm,fwhm,x1,x2
    cdef double[:] x,f
    cdef int nx

    # [amplitude, x0, y0, xsig, ysig, theta, relative amplitude, sigma]

    amp = pars[0]
    xsig = pars[3]
    ysig = pars[4]
    relamp = np.clip(pars[6],0.0,1.0)  # 0<relamp<1
    sigma = np.maximum(pars[7],0)
    beta = 1.2   # Moffat
    
    # The mean radius of an ellipse is: (2a+b)/3
    sig_major = np.max(np.array([xsig,ysig]))
    sig_minor = np.min(np.array([xsig,ysig]))
    mnsig = (2.0*sig_major+sig_minor)/3.0
    # Convert sigma to FWHM
    # FWHM = 2*sqrt(2*ln(2))*sig ~ 2.35482*sig
    gfwhm = mnsig*2.35482
    if relamp==0:
        return gfwhm
    
    # Moffat beta=1.2 FWHM
    mfwhm = 2.0 * np.abs(sigma) * np.sqrt(2.0 ** (1.0/beta) - 1.0)

    # Generate a small profile
    x1 = np.min(np.array([gfwhm,mfwhm]))/2.35/2
    x2 = np.max(np.array([gfwhm,mfwhm]))
    x = np.arange(x1,x2,0.5)
    nx = len(x)
    f = np.zeros(nx,float)
    for i in range(nx):
        f[i] = (1-relamp)*np.exp(-0.5*(x[i]/mnsig)**2) + relamp/(1+(x[i]/sigma)**2)**beta
    hwhm = np.interp(0.5,f[::-1],x[::-1])
    fwhm = 2*hwhm

    return fwhm


cpdef double penny2d_flux(double[:] pars):
    """
    Return the total Flux of a 2D Penny function.

    Parameters
    ----------
    pars : numpy array
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
    cdef double amp,xsig,ysig,sigma,beta
    cdef double gvolume,lvolume,volume

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
    gvolume = 2*pi*amp*(1-relamp)*xsig*ysig

    # Moffat beta=1.2 wings portion
    lvolume = amp*relamp * sigma**2 * pi/(beta-1)
    
    # Sum
    volume = gvolume + lvolume
    
    return volume


cpdef double[:,:] apenny2d(double[:] x, double[:] y, double[:] pars, int nderiv, int osamp):
    """
    Two dimensional Penny model function with x/y array inputs.
    
    Parameters
    ----------
    x : numpy array
      Array of X-values of points for which to compute the Penny model
    y : numpy array
      Array of Y-values of points for which to compute the Penny model.
    pars : numpy array
       Parameter list. pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta].
    nderiv : int
       The number of derivatives to return.

    Returns
    -------
    g : numpy array
      The Penny model for the input x/y values and parameters.  Always
        returned as 1D raveled() array.
    derivative : numpy array
      Array derivatives of g relative to the input parameters.
        Always 2D [Npix,Nderiv] with the 1st dimension being the x/y arrays
        raveled() to 1D.

    Example
    -------

    g,derivative = apenny2d(x,y,pars,3)

    """
    cdef double amp,xc,yc,asemi,bsemi,theta,relamp,sigma,cxx,cyy,cxy
    cdef double x1,y1
    cdef long i,j,npix,index

    amp = pars[0]
    xc = pars[1]
    yc = pars[2]
    xsig = pars[3]
    ysig = pars[4]
    theta = pars[5]
    relamp = pars[6]
    sigma = pars[7]
    if len(pars)==8:
        cxx,cyy,cxy = gauss_abt2cxy(xsig,ysig,theta)
    else:
        cxx = pars[8]
        cyy = pars[9]
        cxy = pars[10]

    cdef double allpars[11]
    allpars[0] = amp
    allpars[1] = xc
    allpars[2] = yc
    allpars[3] = xsig
    allpars[4] = ysig
    allpars[5] = theta
    allpars[6] = relamp
    allpars[7] = sigma
    allpars[8] = cxx
    allpars[9] = cyy
    allpars[10] = cxy

    npix = len(x)

    # 2D arrays
    out = cvarray(shape=(npix,9),itemsize=sizeof(double),format="d")
    cdef double[:,:] mout = out

    cdef double *out1 = <double*>malloc(9 * sizeof(double))

    # Loop over the points
    for i in range(npix):
        x1 = x[i]
        y1 = y[i]
        penny2d_integrate(x1,y1,allpars,nderiv,osamp,out1)
        for j in range(nderiv+1):
            mout[i,j] = out1[j]

    free(out1)

    return mout    

cdef void penny2d_integrate(double x, double y, double[11] pars, int nderiv, int osamp, double* out):
    """
    Two dimensional Penny model function for a single point.
    Gaussian core and Lorentzian-like wings, only Gaussian is tilted.

    Parameters
    ----------
    x : float
      Single X-value for which to compute the Penny model.
    y : float
      Single Y-value for which to compute the Penny model.
    pars : numpy array
       Parameter list. pars = [amplitude, x0, y0, xsigma, ysigma, theta,
                               relamp, sigma]
         The cxx, cyy, cxy parameter can be added to the end so they don't
         have to be computed.
    nderiv : int
       The number of derivatives to return.

    Returns
    -------
    g : float
      The Penny model for the input x/y values and parameters.
    derivative : numpy array
      Array of derivatives of g relative to the input parameters.

    Example
    -------

    g,derivative = penny2d(x,y,pars,nderiv)

    """

    cdef double theta,cost2,sint2,amp,relamp,sigma,beta,rr_gg
    cdef double xsig2,ysig2,a,b,c,u,v,u2,v2,x0,y0,dx,dy
    cdef int nx,ny,col,row,nsamp,hosamp,i
    cdef double df_dA,df_dx_mean,df_dy_mean,df_dxsig,df_dysig,df_dtheta,df_drelamp,df_dsigma
    cdef double cost,sint,xsig3,ysig3,da_dxsig,db_dxsig,dc_dxsig
    cdef double da_dysig,db_dysig,dc_dysig
    cdef double da_dtheta,db_dtheta,dc_dtheta

    # pars = [amplitude, x0, y0, xsigma, ysigma, theta, relamp, sigma, cxx, cyy, cxy]
    amp = pars[0]
    x0 = pars[1]
    y0 = pars[2]
    xsig = pars[3]
    ysig = pars[4]
    theta = pars[5]
    relamp = pars[6]
    sigma = pars[7]
    cxx = pars[8]
    cyy = pars[9]
    cxy = pars[10]
    sint = sin(theta)
    cost = cos(theta)
    cost2 = cost ** 2
    sint2 = sint ** 2
    sin2t = sin(2. * theta)
    xsig2 = xsig ** 2
    ysig2 = ysig ** 2

    if relamp < 0:
        relamp = 0.0
    if relamp > 1:
        relamp = 1.0
    if sigma < 0:
        sigma = 0.0

    u = x-x0
    v = y-y0
    cdef double f = 0.0
    if osamp < 1:
        f = exp(-((cxx * u ** 2) + (cxy * u * v) +
                  (cyy * v ** 2)))

    # Automatically determine the oversampling
    # These are the thresholds that daophot uses
    # from the IRAF daophot version in
    # noao/digiphot/daophot/daolib/profile.x
    if osamp < 1:
        if (f >= 0.046):
            osamp = 4
        elif (f >= 0.0022):
            osamp = 3
        elif (f >= 0.0001):
            osamp = 2
        elif (f >= 1.0e-10):
            osamp = 1

    nsamp = osamp*osamp
    cdef double dd = 0.0
    cdef double dd0 = 0.0
    # dx = (np.arange(osamp).astype(float)+1)/osamp-(1/(2*osamp))-0.5
    if osamp>1:
        dd = 1/float(osamp)
        dd0 = 1/(2*float(osamp))-0.5

    cdef double g = 0.0
    cdef double l = 0.0
    for i in range(8):
        out[i] = 0.0
    hosamp = osamp//2
    df_dA = 0.0
    df_dx_mean = 0.0
    df_dy_mean = 0.0
    df_dxsig = 0.0
    df_dysig = 0.0
    df_dtheta = 0.0
    df_drelamp = 0.0
    df_dsigma = 0.0
    for i in range(nsamp):
        col = i // osamp
        row = i % osamp
        dx = col*dd+dd0
        dy = row*dd+dd0
        u = (x+dx)-x0
        v = (y+dy)-y0
        u2 = u*u
        v2 = v*v

        # Gaussian component
        g = amp * (1-relamp) * exp(-0.5*((cxx * u2) + (cxy * u*v) +
                                         (cyy * v2)))
        # Add Lorentzian/Moffat beta=1.2 wings
        rr_gg = (u2+v2) / sigma ** 2
        beta = 1.2
        l = amp * relamp / (1 + rr_gg)**(beta)
        # Sum of Gaussian + Lorentzian
        f = g + l
        out[0] += f

        # Compute derivative as well
        if nderiv>=1:
            df_dA = f / amp
            out[1] += df_dA
        if nderiv>=2:
            df_dx_mean = ( g * 0.5*((2 * cxx * u) + (cxy * v)) +                           
                           2*beta*l*u/(sigma**2 * (1+rr_gg)) )  
            out[2] += df_dx_mean
        if nderiv>=3:
            df_dy_mean = ( g * 0.5*((2 * cyy * v) + (cxy * u)) +
                           2*beta*l*v/(sigma**2 * (1+rr_gg)) ) 
            out[3] += df_dy_mean
        if nderiv>=4:
            xsig3 = xsig ** 3
            da_dxsig = -cost2 / xsig3
            db_dxsig = -sin2t / xsig3
            dc_dxsig = -sint2 / xsig3
            df_dxsig = g * (-(da_dxsig * u2 +
                              db_dxsig * u * v +
                              dc_dxsig * v2))
            out[4] += df_dxsig
        if nderiv>=5:
            ysig3 = ysig ** 3
            da_dysig = -sint2 / ysig3
            db_dysig = sin2t / ysig3
            dc_dysig = -cost2 / ysig3
            df_dysig = g * (-(da_dysig * u2 +
                              db_dysig * u * v +
                              dc_dysig * v2))
            out[5] += df_dysig
        if nderiv>=6 and xsig != ysig:
            cos2t = cos(2.0*theta)
            da_dtheta = (sint * cost * ((1. / ysig2) - (1. / xsig2)))
            db_dtheta = (cos2t / xsig2) - (cos2t / ysig2)
            dc_dtheta = -da_dtheta
            df_dtheta = g * (-(da_dtheta * u2 +
                               db_dtheta * u * v +
                               dc_dtheta * v2))
            out[6] += df_dtheta
        if nderiv>=7:
            df_drelamp = -g/(1-relamp) + l/relamp
            out[7] += df_drelamp
        if nderiv>=8:
            df_dsigma = beta*l/(1+rr_gg) * 2*(u2+v2)/sigma**3 
            out[8] += df_dsigma

    if osamp>1:
        for i in range(nderiv+1):
            out[i] /= nsamp   # take average


# #
# cpdef penny2dfit(im,err,ampc,xc,yc,verbose):
#     """
#     Fit a single Penny 2D model to data.

#     Parameters
#     ----------
#     im : numpy array
#        Flux array.  Can be 1D or 2D array.
#     err : numpy array
#        Uncertainty array of im.  Same dimensions as im.
#     ampc : float
#        Initial guess of amplitude.
#     xc : float
#        Initial guess of central X coordinate.
#     yc : float
#        Initial guess of central Y coordinate.
#     verbose : bool
#        Verbose output to the screen.

#     Returns
#     -------
#     pars : numpy array
#        Best fit pararmeters.
#     perror : numpy array
#        Uncertainties in pars.
#     pcov : numpy array
#        Covariance matrix.
#     flux : float
#        Best fit flux.
#     fluxerr : float
#        Uncertainty in flux.
    
#     Example
#     -------

#     pars,perror,cov,flux,fluxerr = penny2dfit(im,err,1,100.0,5.5,6.5,False)

#     """
#     # xc/yc are with respect to the image origin (0,0)
    
#     # Solve for x, y, amplitude and asemi/bsemi/theta

#     maxiter = 10
#     minpercdiff = 0.5
    
#     ny,nx = im.shape
#     im1d = im.ravel()

#     x2d,y2d = utils.meshgrid(np.arange(nx),np.arange(ny))
#     x1d = x2d.ravel()
#     y1d = y2d.ravel()
    
#     wt = 1/err**2
#     wt1d = wt.ravel()

#     asemi = 2.5
#     bsemi = 2.4
#     theta = 0.1
#     relamp = 0.2
#     sigma = 2*asemi

#     # theta in radians
    
#     # Initial values
#     bestpar = np.zeros(8,float)
#     bestpar[0] = ampc
#     bestpar[1] = xc
#     bestpar[2] = yc
#     bestpar[3] = asemi
#     bestpar[4] = bsemi
#     bestpar[5] = theta
#     bestpar[6] = relamp
#     bestpar[7] = sigma
    
#     # Iteration loop
#     maxpercdiff = 1e10
#     niter = 0
#     while (niter<maxiter and maxpercdiff>minpercdiff):
#         model,deriv = apenny2d(x1d,y1d,bestpar,8)
#         resid = im1d-model
#         dbeta = utils.qr_jac_solve(deriv,resid,weight=wt1d)
        
#         if verbose:
#             print(niter,bestpar)
#             print(dbeta)
        
#         # Update parameters
#         last_bestpar = bestpar.copy()
#         # limit the steps to the maximum step sizes and boundaries
#         #if bounds is not None or maxsteps is not None:
#         #    bestpar = newpars(bestpar,dbeta,bounds,maxsteps)
#         bounds = np.zeros((8,2),float)
#         bounds[:,0] = [0.00, 0, 0, 0.1, 0.1, -180, 0.00, 0.1]
#         bounds[:,1] = [1e30,nx,ny, nx//2, ny//2, 180, 1, 10]
#         maxsteps = np.zeros(8,float)
#         maxsteps[:] = [0.5*bestpar[0],0.5,0.5,0.5,0.5,2.0,0.02,0.5]
#         bestpar = newlsqpars(bestpar,dbeta,bounds,maxsteps)
        
#         # Check differences and changes
#         diff = np.abs(bestpar-last_bestpar)
#         denom = np.maximum(np.abs(bestpar.copy()),0.0001)
#         percdiff = diff.copy()/denom*100  # percent differences
#         maxpercdiff = np.max(percdiff)
#         chisq = np.sum((im1d-model)**2 * wt1d)/(nx*ny)
#         if verbose:
#             print('chisq=',chisq)
#         #if verbose:
#         #    print(niter,percdiff,chisq)
#         #    print()
#         last_dbeta = dbeta
#         niter += 1

#     model,deriv = apenny2d(x1d,y1d,bestpar,8)
#     resid = im1d-model
    
#     # Get covariance and errors
#     cov = utils.jac_covariance(deriv,resid,wt1d)
#     perror = np.sqrt(np.diag(cov))

#     # Now get the flux, multiply by the volume of the Gaussian
#     #asemi,bsemi,theta = bestpar[3],bestpar[4],bestpar[5]
#     #gvolume = asemi*bsemi*2*pi
#     #flux = bestpar[0]*gvolume
#     #fluxerr = perror[0]*gvolume

#     # USE PENNY_FLUX
#     # pars = [amplitude, x0, y0, xsigma, ysigma, theta, relamp, sigma]
#     flux = penny2d_flux(bestpar)
#     fluxerr = perror[0]*(flux/bestpar[0])    
    
#     return bestpar,perror,cov,flux,fluxerr


# ####### GAUSSPOW ########


cpdef double gausspow2d_fwhm(double[:] pars):
    """
    Return the FWHM of a 2D DoPHOT Gausspow function.

    Parameters
    ----------
    pars : numpy array
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
    cdef double amp,xsig,ysig,theta,beta4,beta6
    cdef double cost2,sint2,sin2t,xsig2,ysig2
    cdef double a,b,c,sig_major,sig_minor,mnsig
    cdef double gfwhm,hwhm,fwhm
    cdef double[:] x,z2,gxy,f
    cdef int nx

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
    sig_major = np.max(np.array([xsig,ysig]))
    sig_minor = np.min(np.array([xsig,ysig]))
    mnsig = (2.0*sig_major+sig_minor)/3.0
    # Convert sigma to FWHM
    # FWHM = 2*sqrt(2*ln(2))*sig ~ 2.35482*sig
    gfwhm = mnsig*2.35482

    # Generate a small profile along one axis with xsig=mnsig
    x = np.arange(gfwhm/2.35/2, gfwhm, 0.5)
    nx = len(x)
    z2 = np.zeros(nx,float)
    gxy = np.zeros(nx,float)
    f = np.zeros(nx,float)
    for i in range(nx):
        z2[i] = 0.5*(x[i]/mnsig)**2
        gxy[i] = (1+z2[i]+0.5*beta4*z2[i]**2+(1.0/6.0)*beta6*z2[i]**3)
        f[i] = amp / gxy[i]

    hwhm = np.interp(0.5,f[::-1],x[::-1])
    fwhm = 2*hwhm
    
    return fwhm


cpdef double gausspow2d_flux(double[:] pars):
    """
    Return the flux of a 2D DoPHOT Gausspow function.

    Parameters
    ----------
    pars : numpy array
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
    cdef double amp,xsig,ysig,beta4,beta6
    cdef double integral,volume
    cdef double[:] p

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
    p = np.array([0.20326739, 0.019689948, 0.023564239,
                  0.63367201, 0.044905046, 0.28862448])
    integral = p[0]/(p[1]+p[2]*beta4**p[3]+p[4]*beta6**p[5])
    # The integral is then multiplied by amp*pi*xsig*ysig

    # This seems to be accurate to ~0.5%
    
    volume = pi*amp*xsig*ysig*integral
    
    return volume



cpdef double[:,:] agausspow2d(double[:] x, double[:] y, double[:] pars, int nderiv, int osamp):
    """
    Two dimensional Gausspow model function with x/y array inputs.
    
    Parameters
    ----------
    x : numpy array
      Array of X-values of points for which to compute the Gausspow model
    y : numpy array
      Array of Y-values of points for which to compute the Gausspow model.
    pars : numpy array
       Parameter list.
        pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta4, beta6].
    nderiv : int
       The number of derivatives to return.

    Returns
    -------
    g : numpy array
      The Gausspow model for the input x/y values and parameters.  Always
        returned as 1D raveled() array.
    derivative : numpy array
      Array derivatives of g relative to the input parameters.
        Always 2D [Npix,Nderiv] with the 1st dimension being the x/y arrays
        raveled() to 1D.

    Example
    -------

    g,derivative = agausspow2d(x,y,pars,nderiv)

    """
    cdef double amp,xc,yc,xsig,ysig,theta,cxx,cyy,cxy,beta4,beta6
    cdef double x1,y1
    cdef long i,j,npix,index

    amp = pars[0]
    xc = pars[1]
    yc = pars[2]
    xsig = pars[3]
    ysig = pars[4]
    theta = pars[5]
    beta4 = pars[6]
    beta6 = pars[7]
    if len(pars)==8:
        cxx,cyy,cxy = gauss_abt2cxy(xsig,ysig,theta)
    else:
        cxx = pars[8]
        cyy = pars[9]
        cxy = pars[10]

    cdef double allpars[11]
    allpars[0] = amp
    allpars[1] = xc
    allpars[2] = yc
    allpars[3] = xsig
    allpars[4] = ysig
    allpars[5] = theta
    allpars[6] = beta4
    allpars[7] = beta6
    allpars[8] = cxx
    allpars[9] = cyy
    allpars[10] = cxy

    npix = len(x)

    # 2D arrays
    out = cvarray(shape=(npix,9),itemsize=sizeof(double),format="d")
    cdef double[:,:] mout = out

    cdef double *out1 = <double*>malloc(9 * sizeof(double))

    # Loop over the points
    for i in range(npix):
        x1 = x[i]
        y1 = y[i]
        gausspow2d_integrate(x1,y1,allpars,nderiv,osamp,out1)
        for j in range(nderiv+1):
            mout[i,j] = out1[j]

    free(out1)

    return mout


cdef void gausspow2d_integrate(double x, double y, double[11] pars, int nderiv, int osamp, double* out):
    """
    DoPHOT PSF, sum of elliptical Gaussians.
    For a single point.

    Parameters
    ----------
    x : float
      Single X-value for which to compute the Gausspow model.
    y : float
      Single Y-value for which to compute the Gausspow model.
    pars : numpy array
       Parameter list.
        pars = [amplitude, x0, y0, sigx, sigy, theta, beta4, beta6]
         The cxx, cyy, cxy parameter can be added to the end so they don't
         have to be computed.
    nderiv : int
       The number of derivatives to return.

    Returns
    -------
    g : float
      The Gausspow model for the input x/y values and parameters.
    derivative : numpy array
      Array of derivatives of g relative to the input parameters.

    Example
    -------

    g,derivative = gausspow2d(x,y,pars,nderiv)

    """
    cdef double theta,cost2,sint2,sin2t,amp,beta4,beta6
    cdef double xsig2,ysig2,a,b,c,u,v,u2,v2,x0,y0,dx,dy
    cdef int nx,ny,col,row,nsamp,hosamp,i
    #cdef double[:] x2,y2,u,v,g
    cdef double dg_dA,dg_dx_mean,dg_dy_mean,dg_dxsig,dg_dysig,dg_dtheta,dg_dbeta
    cdef double cost,sint,xsig3,ysig3,da_dxsig,db_dxsig,dc_dxsig
    cdef double da_dysig,db_dysig,dc_dysig
    cdef double da_dtheta,db_dtheta,dc_dtheta
    cdef double z2,gxy,dgxy_dz2,g_gxy

    # pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta4, beta6, cxx, cyy, cxy]
    amp = pars[0]
    x0 = pars[1]
    y0 = pars[2]
    xsig = pars[3]
    ysig = pars[4]
    theta = pars[5]
    beta4 = pars[6]
    beta6 = pars[7]
    cxx = pars[8]
    cyy = pars[9]
    cxy = pars[10]
    sint = sin(theta)
    cost = cos(theta)
    cost2 = cost ** 2
    sint2 = sint ** 2
    sin2t = sin(2. * theta)
    xsig2 = xsig ** 2
    ysig2 = ysig ** 2

    # Schechter, Mateo & Saha (1993), eq. 1 on pg.4
    # I(x,y) = Io * (1+z2+0.5*beta4*z2**2+(1/6)*beta6*z2**3)**(-1)
    # z2 = [0.5*(x**2/sigx**2 + 2*sigxy*x*y + y**2/sigy**2]
    # x = (x'-x0)
    # y = (y'-y0)
    # nominal center of image at (x0,y0)
    # if beta4=beta6=1, then it's just a truncated power series for a Gaussian
    # 8 free parameters
    # pars = [amplitude, x0, y0, sigx, sigy, theta, beta4, beta6]

    u = x-x0
    v = y-y0
    cdef double f = 0.0
    if osamp < 1:
        f = exp(-((cxx * u ** 2) + (cxy * u * v) +
                  (cyy * v ** 2)))

    # Automatically determine the oversampling
    # These are the thresholds that daophot uses
    # from the IRAF daophot version in
    # noao/digiphot/daophot/daolib/profile.x
    if osamp < 1:
        if (f >= 0.046):
            osamp = 4
        elif (f >= 0.0022):
            osamp = 3
        elif (f >= 0.0001):
            osamp = 2
        elif (f >= 1.0e-10):
            osamp = 1

    nsamp = osamp*osamp
    cdef double dd = 0.0
    cdef double dd0 = 0.0
    # dx = (np.arange(osamp).astype(float)+1)/osamp-(1/(2*osamp))-0.5
    if osamp>1:
        dd = 1/float(osamp)
        dd0 = 1/(2*float(osamp))-0.5

    cdef double g = 0.0
    for i in range(8):
        out[i] = 0.0
    hosamp = osamp//2
    dg_dA = 0.0
    dg_dx_mean = 0.0
    dg_dy_mean = 0.0
    dg_dxsig = 0.0
    dg_dysig = 0.0
    dg_dtheta = 0.0
    dg_dbeta4 = 0.0
    dg_dbeta6 = 0.0
    for i in range(nsamp):
        col = i // osamp
        row = i % osamp
        dx = col*dd+dd0
        dy = row*dd+dd0
        u = (x+dx)-x0
        v = (y+dy)-y0
        u2 = u*u
        v2 = v*v

        z2 = 0.5 * (cxx*u2 + cxy*u*v + cyy*v2)
        gxy = (1 + z2 + 0.5*beta4*z2**2 + (1.0/6.0)*beta6*z2**3)
        g = amp / gxy
        out[0] += g

        # Compute derivative as well
        if nderiv>=1:
            dgxy_dz2 = (1 + beta4*z2 + 0.5*beta6*z2**2)
            g_gxy = g / gxy
            dg_dA = g / amp
            out[1] += dg_dA
        if nderiv>=2:
            dg_dx_mean = g_gxy * dgxy_dz2 * 0.5 * (2*cxx*u + cxy*v)
            out[2] += dg_dx_mean
        if nderiv>=3:
            dg_dy_mean = g_gxy * dgxy_dz2 * 0.5 * (2*cyy*v + cxy*u)
            out[3] += dg_dy_mean
        if nderiv>=4:
            xsig3 = xsig ** 3
            da_dxsig = -cost2 / xsig3
            db_dxsig = -sin2t / xsig3
            dc_dxsig = -sint2 / xsig3
            dg_dxsig = -g_gxy * dgxy_dz2 * (da_dxsig * u2 +
                                            db_dxsig * u * v +
                                            dc_dxsig * v2)   
            out[4] += dg_dxsig
        if nderiv>=5:
            ysig3 = ysig ** 3
            da_dysig = -sint2 / ysig3
            db_dysig = sin2t / ysig3
            dc_dysig = -cost2 / ysig3
            dg_dysig = -g_gxy * dgxy_dz2 * (da_dysig * u2 +
                                            db_dysig * u * v +
                                            dc_dysig * v2)
            out[5] += dg_dysig
        if nderiv>=6 and xsig != ysig:
            cos2t = cos(2.0*theta)
            da_dtheta = (sint * cost * ((1. / ysig2) - (1. / xsig2)))
            db_dtheta = (cos2t / xsig2) - (cos2t / ysig2)
            dc_dtheta = -da_dtheta
            dg_dtheta = -g_gxy * dgxy_dz2 * (da_dtheta * u2 +
                                             db_dtheta * u * v +
                                             dc_dtheta * v2)
            out[6] += dg_dtheta
        if nderiv>=7:
            dg_dbeta4 = -g_gxy * (0.5*z2**2)
            out[7] = dg_dbeta4
        if nderiv>=8:
            dg_dbeta6 = -g_gxy * ((1.0/6.0)*z2**3)
            out[8] = dg_dbeta6

    if osamp>1:
        for i in range(nderiv+1):
            out[i] /= nsamp   # take average


cpdef list gausspow2d(double x, double y, double[:] pars, int nderiv):
    """
    DoPHOT PSF, sum of elliptical Gaussians.
    For a single point.

    Parameters
    ----------
    x : float
      Single X-value for which to compute the Gausspow model.
    y : float
      Single Y-value for which to compute the Gausspow model.
    pars : numpy array
       Parameter list.
        pars = [amplitude, x0, y0, sigx, sigy, theta, beta4, beta6]
         The cxx, cyy, cxy parameter can be added to the end so they don't
         have to be computed.
    nderiv : int
       The number of derivatives to return.

    Returns
    -------
    g : float
      The Gausspow model for the input x/y values and parameters.
    derivative : numpy array
      Array of derivatives of g relative to the input parameters.

    Example
    -------

    g,derivative = gausspow2d(x,y,pars,nderiv)

    """

    cdef double u,u2,v,v2,z2,gxy
    cdef double g
    cdef double[:] deriv

    if len(pars)==8:
        amp,xc,yc,asemi,bsemi,theta,beta4,beta6 = pars
        cxx,cyy,cxy = gauss_abt2cxy(asemi,bsemi,theta)
    else:
        amp,xc,yc,asemi,bsemi,theta,beta4,beta6,cxx,cyy,cxy = pars

    # Schechter, Mateo & Saha (1993), eq. 1 on pg.4
    # I(x,y) = Io * (1+z2+0.5*beta4*z2**2+(1/6)*beta6*z2**3)**(-1)
    # z2 = [0.5*(x**2/sigx**2 + 2*sigxy*x*y + y**2/sigy**2]
    # x = (x'-x0)
    # y = (y'-y0)
    # nominal center of image at (x0,y0)
    # if beta4=beta6=1, then it's just a truncated power series for a Gaussian
    # 8 free parameters
    # pars = [amplitude, x0, y0, sigx, sigy, theta, beta4, beta6]
        
    u = (x-xc)
    u2 = u**2
    v = (y-yc)
    v2 = v**2
    z2 = 0.5 * (cxx*u2 + cxy*u*v + cyy*v2)
    gxy = (1 + z2 + 0.5*beta4*z2**2 + (1.0/6.0)*beta6*z2**3)
    g = amp / gxy
    
    #  pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta4, beta6]
    deriv = np.zeros(nderiv,float)    
    if nderiv>0:
        dgxy_dz2 = (1 + beta4*z2 + 0.5*beta6*z2**2)
        g_gxy = g / gxy
        # amplitude
        dg_dA = g / amp
        deriv[0] = dg_dA
        # x0
        dg_dx_mean = g_gxy * dgxy_dz2 * 0.5 * (2*cxx*u + cxy*v)
        deriv[1] = dg_dx_mean
        # y0
        dg_dy_mean = g_gxy * dgxy_dz2 * 0.5 * (2*cyy*v + cxy*u)
        deriv[2] = dg_dy_mean
        if nderiv>3:
            sint = np.sin(theta)        
            cost = np.cos(theta)        
            sint2 = sint ** 2
            cost2 = cost ** 2
            sin2t = np.sin(2. * theta)
            # asemi/xsig
            asemi2 = asemi ** 2
            asemi3 = asemi ** 3
            da_dxsig = -cost2 / asemi3
            db_dxsig = -sin2t / asemi3
            dc_dxsig = -sint2 / asemi3
            dg_dxsig = -g_gxy * dgxy_dz2 * (da_dxsig * u2 +
                                            db_dxsig * u * v +
                                            dc_dxsig * v2)   
            deriv[3] = dg_dxsig
            # bsemi/ysig
            bsemi2 = bsemi ** 2
            bsemi3 = bsemi ** 3
            da_dysig = -sint2 / bsemi3
            db_dysig = sin2t / bsemi3
            dc_dysig = -cost2 / bsemi3
            dg_dysig = -g_gxy * dgxy_dz2 * (da_dysig * u2 +
                                            db_dysig * u * v +
                                            dc_dysig * v2)
            deriv[4] = dg_dysig
            # dtheta
            if asemi != bsemi:
                cos2t = np.cos(2.0*theta)
                da_dtheta = (sint * cost * ((1. / bsemi2) - (1. / asemi2)))
                db_dtheta = (cos2t / asemi2) - (cos2t / bsemi2)
                dc_dtheta = -da_dtheta
                dg_dtheta = -g_gxy * dgxy_dz2 * (da_dtheta * u2 +
                                                 db_dtheta * u * v +
                                                 dc_dtheta * v2)
                deriv[5] = dg_dtheta
            # beta4
            dg_dbeta4 = -g_gxy * (0.5*z2**2)
            deriv[6] = dg_dbeta4
            # beta6
            dg_dbeta6 = -g_gxy * ((1.0/6.0)*z2**3)
            deriv[7] = dg_dbeta6
            
    return [g,deriv]

# #
# cpdef gausspow2dfit(im,err,ampc,xc,yc,verbose):
#     """
#     Fit a single GaussPOW 2D model to data.

#     Parameters
#     ----------
#     im : numpy array
#        Flux array.  Can be 1D or 2D array.
#     err : numpy array
#        Uncertainty array of im.  Same dimensions as im.
#     ampc : float
#        Initial guess of amplitude.
#     xc : float
#        Initial guess of central X coordinate.
#     yc : float
#        Initial guess of central Y coordinate.
#     verbose : bool
#        Verbose output to the screen.

#     Returns
#     -------
#     pars : numpy array
#        Best fit pararmeters.
#     perror : numpy array
#        Uncertainties in pars.
#     pcov : numpy array
#        Covariance matrix.
#     flux : float
#        Best fit flux.
#     fluxerr : float
#        Uncertainty in flux.
    
#     Example
#     -------

#     pars,perror,cov,flux,fluxerr = gausspow2dfit(im,err,1,100.0,5.5,6.5,False)

#     """

#     # xc/yc are with respect to the image origin (0,0)
    
#     # Solve for x, y, amplitude and asemi/bsemi/theta

#     maxiter = 10
#     minpercdiff = 0.5
    
#     ny,nx = im.shape
#     im1d = im.ravel()

#     x2d,y2d = utils.meshgrid(np.arange(nx),np.arange(ny))
#     x1d = x2d.ravel()
#     y1d = y2d.ravel()
    
#     wt = 1/err**2
#     wt1d = wt.ravel()

#     asemi = 2.5
#     bsemi = 2.4
#     theta = 0.1
#     beta4 = 3.5
#     beta6 = 4.5

#     # theta in radians
    
#     # Initial values
#     bestpar = np.zeros(8,float)
#     bestpar[0] = ampc
#     bestpar[1] = xc
#     bestpar[2] = yc
#     bestpar[3] = asemi
#     bestpar[4] = bsemi
#     bestpar[5] = theta
#     bestpar[6] = beta4
#     bestpar[7] = beta6
    
#     # Iteration loop
#     maxpercdiff = 1e10
#     niter = 0
#     while (niter<maxiter and maxpercdiff>minpercdiff):
#         model,deriv = agausspow2d(x1d,y1d,bestpar,8)
#         resid = im1d-model
#         dbeta = utils.qr_jac_solve(deriv,resid,weight=wt1d)
        
#         if verbose:
#             print(niter,bestpar)
#             print(dbeta)
        
#         # Update parameters
#         last_bestpar = bestpar.copy()
#         # limit the steps to the maximum step sizes and boundaries
#         #if bounds is not None or maxsteps is not None:
#         #    bestpar = newpars(bestpar,dbeta,bounds,maxsteps)
#         bounds = np.zeros((8,2),float)
#         bounds[:,0] = [0.00, 0, 0, 0.1, 0.1, -180, 0.1, 0.1]
#         bounds[:,1] = [1e30,nx,ny, nx//2, ny//2, 180, nx//2, nx//2]
#         maxsteps = np.zeros(8,float)
#         maxsteps[:] = [0.5*bestpar[0],0.5,0.5,0.5,0.5,2.0,0.5,0.5]
#         bestpar = newlsqpars(bestpar,dbeta,bounds,maxsteps)
        
#         # Check differences and changes
#         diff = np.abs(bestpar-last_bestpar)
#         denom = np.maximum(np.abs(bestpar.copy()),0.0001)
#         percdiff = diff.copy()/denom*100  # percent differences
#         maxpercdiff = np.max(percdiff)
#         chisq = np.sum((im1d-model)**2 * wt1d)/(nx*ny)
#         if verbose:
#             print('chisq=',chisq)
#         #if verbose:
#         #    print(niter,percdiff,chisq)
#         #    print()
#         last_dbeta = dbeta
#         niter += 1

#     model,deriv = agausspow2d(x1d,y1d,bestpar,8)
#     resid = im1d-model
    
#     # Get covariance and errors
#     cov = utils.jac_covariance(deriv,resid,wt1d)
#     perror = np.sqrt(np.diag(cov))

#     # Now get the flux, multiply by the volume of the Gaussian
#     #asemi,bsemi,theta = bestpar[3],bestpar[4],bestpar[5]
#     #gvolume = asemi*bsemi*2*pi
#     #flux = bestpar[0]*gvolume
#     #fluxerr = perror[0]*gvolume

#     # USE GAUSSPOW_FLUX
#     # pars = [amplitude, x0, y0, xsigma, ysigma, theta, relamp, sigma]
#     flux = gausspow2d_flux(bestpar)
#     fluxerr = perror[0]*(flux/bestpar[0])    
    
#     return bestpar,perror,cov,flux,fluxerr


# ####### SERSIC ########


cpdef double[:,:] asersic2d(double[:] x, double[:] y, double[:] pars, int nderiv, int osamp):
    """
    Sersic profile and can be elliptical and rotated.
    With x/y arrays input.
    
    Parameters
    ----------
    x : numpy array
      Array of X-values of points for which to compute the Sersic model.
    y : numpy array
      Array of Y-values of points for which to compute the Sersic model.
    pars : numpy array
       Parameter list.
        pars = [amp,x0,y0,k,alpha,recc,theta]
    nderiv : int
       The number of derivatives to return.

    Returns
    -------
    g : numpy array
      The Sersic model for the input x/y values and parameters (same
        shape as x/y).
    derivative : numpy array
      Array of derivatives of g relative to the input parameters.

    Example
    -------
    
    g,derivative = sersic2d(x,y,pars,nderiv)

    """
    cdef double amp,xc,yc,kserc,alpha,recc,theta,xsig,ysig,xsig2,ysig2,cxx,cyy,cxy
    cdef double x1,y1
    cdef long i,j,npix,index

    amp = pars[0]
    xc = pars[1]
    yc = pars[2]
    kserc = pars[3]
    alpha = pars[4]
    recc = pars[5]
    theta = pars[6]
    xsig2 = 1.0           # major axis
    ysig2 = recc ** 2     # minor axis
    if len(pars)==7:
        xsig = 1.0
        ysig = sqrt(ysig2)
        cxx,cyy,cxy = gauss_abt2cxy(xsig,ysig,theta)
    else:
        cxx = pars[7]
        cyy = pars[8]
        cxy = pars[9]

    cdef double allpars[10]
    allpars[0] = amp
    allpars[1] = xc
    allpars[2] = yc
    allpars[3] = kserc
    allpars[4] = alpha
    allpars[5] = recc
    allpars[6] = theta
    allpars[7] = cxx
    allpars[8] = cyy
    allpars[9] = cxy

    npix = len(x)

    # 2D arrays
    out = cvarray(shape=(npix,8),itemsize=sizeof(double),format="d")
    cdef double[:,:] mout = out

    cdef double *out1 = <double*>malloc(8 * sizeof(double))

    # Loop over the points
    for i in range(npix):
        x1 = x[i]
        y1 = y[i]
        sersic2d_integrate(x1,y1,allpars,nderiv,osamp,out1)
        for j in range(nderiv+1):
            mout[i,j] = out1[j]

    free(out1)

    return mout

cdef void sersic2d_integrate(double x, double y, double[10] pars, int nderiv, int osamp, double* out):
    """
    Sersic profile and can be elliptical and rotated.
    For a single point.

    Parameters
    ----------
    x : float
      Single X-value for which to compute the Sersic model.
    y : float
      Single Y-value for which to compute the Sersic model.
    pars : numpy array
       Parameter list.
        pars = [amp,x0,y0,k,alpha,recc,theta]
    nderiv : int
       The number of derivatives to return.

    Returns
    -------
    g : float
      The Sersic model for the input x/y values and parameters (same
        shape as x/y).
    derivative : numpy array
      Array of derivatives of g relative to the input parameters.

    Example
    -------
    
    g,derivative = sersic2d(x,y,pars,nderiv)

    """
    cdef double theta,cost2,sint2,sin2t,amp,kserc,recc,alpha
    cdef double xsig2,ysig2,a,b,c,u,v,u2,v2,x0,y0,dx,dy
    cdef int nx,ny,col,row,nsamp,hosamp,i
    #cdef double[:] x2,y2,u,v,g
    cdef double dg_dA,dg_dx_mean,dg_dy_mean,dg_dkserc,dg_dalpha,dg_drecc,dg_dtheta,rr,du_drr
    cdef double cost,sint,xsig3,ysig3,da_dxsig,db_dxsig,dc_dxsig
    cdef double da_dysig,db_dysig,dc_dysig
    cdef double da_dtheta,db_dtheta,dc_dtheta
    cdef double z2,gxy,dgxy_dz2,g_gxy

    # Sersic radial profile
    # I(R) = I0 * exp(-k*R**(1/n))
    # n is the sersic index
    # I'm going to use alpha = 1/n instead
    # I(R) = I0 * exp(-k*R**alpha)    
    # most galaxies have indices in the range 1/2 < n < 10
    # n=4 is the de Vaucouleurs profile
    # n=1 is the exponential

    # pars = [amp,x0,y0,k,alpha,recc,theta]
    amp = pars[0]
    x0 = pars[1]
    y0 = pars[2]
    kserc = pars[3]
    alpha = pars[4]
    recc = pars[5]
    theta = pars[6]
    cxx = pars[7]
    cyy = pars[8]
    cxy = pars[9]
    sint = sin(theta)
    cost = cos(theta)
    cost2 = cost ** 2
    sint2 = sint ** 2
    sin2t = sin(2. * theta)
    # recc = b/c
    xsig2 = 1.0           # major axis
    ysig2 = recc ** 2     # minor axis

    if kserc<0:
        kserc = 0

    u = x-x0
    v = y-y0
    cdef double f = 0.0
    if osamp < 1:
        f = exp(-((cxx * u ** 2) + (cxy * u * v) +
                  (cyy * v ** 2)))

    # Automatically determine the oversampling
    # These are the thresholds that daophot uses
    # from the IRAF daophot version in
    # noao/digiphot/daophot/daolib/profile.x
    if osamp < 1:
        if (f >= 0.046):
            osamp = 4
        elif (f >= 0.0022):
            osamp = 3
        elif (f >= 0.0001):
            osamp = 2
        elif (f >= 1.0e-10):
            osamp = 1

    nsamp = osamp*osamp
    cdef double dd = 0.0
    cdef double dd0 = 0.0
    # dx = (np.arange(osamp).astype(float)+1)/osamp-(1/(2*osamp))-0.5
    if osamp>1:
        dd = 1/float(osamp)
        dd0 = 1/(2*float(osamp))-0.5

    cdef double g = 0.0
    for i in range(8):
        out[i] = 0.0
    hosamp = osamp//2
    dg_dA = 0.0
    dg_dx_mean = 0.0
    dg_dy_mean = 0.0
    dg_dkserc = 0.0
    dg_dalpha = 0.0
    dg_drecc = 0.0
    dg_dtheta = 0.0
    for i in range(nsamp):
        col = i // osamp
        row = i % osamp
        dx = col*dd+dd0
        dy = row*dd+dd0
        u = (x+dx)-x0
        v = (y+dy)-y0
        u2 = u*u
        v2 = v*v

        rr = sqrt( (cxx * u2) + (cxy * u * v) + (cyy * v2) )
        g = amp * exp(-kserc*rr**alpha)
        out[0] += g

        # Compute derivative as well
        if nderiv>=1:
            if rr==0:
                du_drr = 1.0
            else:
                du_drr = (kserc*alpha)*(rr**(alpha-2))
            # amplitude
            dg_dA = g / amp
            out[1] = dg_dA
        if nderiv>=2:
            if rr != 0:
                dg_dx_mean = g * du_drr * 0.5 * ((2 * cxx * u) + (cxy * v))
            else:
                # not well defined at rr=0
                # g comes to a sharp point at rr=0
                # if you approach rr=0 from the left, then the slope is +
                # but if you approach rr=0 from the right, then the slope is -
                # use 0 so it is at least well-behaved
                dg_dx_mean = 0.0
            out[2] += dg_dx_mean
        if nderiv>=3:
            if rr != 0:
                dg_dy_mean = g * du_drr * 0.5 * ((2 * cyy * v) + (cxy * u))
            else:
                # not well defined at rr=0, see above
                dg_dy_mean = 0.0
            out[3] += dg_dy_mean
        if nderiv>=4:
            # kserc
            dg_dkserc = -g * rr**alpha
            out[4] = dg_dkserc
        if nderiv>=5:
            # alpha
            if rr != 0:
                dg_dalpha = -g * kserc*log(rr) * rr**alpha
            else:
                dg_dalpha = 0.0
            out[5] = dg_dalpha
        if nderiv>=6:
            # recc
            recc3 = recc**3
            da_drecc = -2*sint2 / recc3
            db_drecc =  2*sin2t / recc3            
            dc_drecc = -2*cost2 / recc3
            if rr==0:
                dg_drecc = 0.0
            else:
                dg_drecc = -g * du_drr * 0.5 * (da_drecc * u2 +
                                                db_drecc * u * v +
                                                dc_drecc * v2)
            out[6] += dg_drecc
        if nderiv>=7:
            # theta
            cos2t = cos(2.0*theta)
            da_dtheta = (sint * cost * ((1. / ysig2) - (1. / xsig2)))
            db_dtheta = (cos2t / xsig2) - (cos2t / ysig2)            
            dc_dtheta = -da_dtheta
            if rr==0:
                dg_dtheta = 0.0
            else:
                dg_dtheta = -g * du_drr * (da_dtheta * u2 +
                                           db_dtheta * u * v +
                                           dc_dtheta * v2)
            out[7] += dg_dtheta

    if osamp>1:
        for i in range(nderiv+1):
            out[i] /= nsamp   # take average



cpdef list sersic2d(double x, double y, double[:] pars, int nderiv):
    """
    Sersic profile and can be elliptical and rotated.
    For a single point.

    Parameters
    ----------
    x : float
      Single X-value for which to compute the Sersic model.
    y : float
      Single Y-value for which to compute the Sersic model.
    pars : numpy array
       Parameter list.
        pars = [amp,x0,y0,k,alpha,recc,theta]
    nderiv : int
       The number of derivatives to return.

    Returns
    -------
    g : float
      The Sersic model for the input x/y values and parameters (same
        shape as x/y).
    derivative : numpy array
      Array of derivatives of g relative to the input parameters.

    Example
    -------
    
    g,derivative = sersic2d(x,y,pars,nderiv)

    """
    cdef double amp,xc,yc,kserc,alpha,recc,theta
    cdef double u,u2,v,v2
    cdef double cost2,sint2,xsig2,ysig2
    cdef double a,b,c,rr,g
    cdef double[:] deriv

    # pars = [amp,x0,y0,k,alpha,recc,theta]
    
    # Sersic radial profile
    # I(R) = I0 * exp(-k*R**(1/n))
    # n is the sersic index
    # I'm going to use alpha = 1/n instead
    # I(R) = I0 * exp(-k*R**alpha)    
    # most galaxies have indices in the range 1/2 < n < 10
    # n=4 is the de Vaucouleurs profile
    # n=1 is the exponential

    amp,xc,yc,kserc,alpha,recc,theta = pars
    u = (x-xc)
    u2 = u**2
    v = (y-yc)
    v2 = v**2
    # recc = b/c
    cost2 = np.cos(theta) ** 2
    sint2 = np.sin(theta) ** 2
    sin2t = np.sin(2. * theta)
    xsig2 = 1.0           # major axis
    ysig2 = recc ** 2     # minor axis
    a = (cost2 + (sint2 / ysig2))
    b = (sin2t - (sin2t / ysig2))    
    c = (sint2 + (cost2 / ysig2))

    rr = np.sqrt( (a * u ** 2) + (b * u * v) + (c * v ** 2) )
    g = amp * np.exp(-kserc*rr**alpha)

    #  pars = [amplitude, x0, y0, kserc, alpha, recc, theta]
    deriv = np.zeros(nderiv,float)    
    if nderiv>0:
        if rr==0:
            du_drr = 1.0
        else:
            du_drr = (kserc*alpha)*(rr**(alpha-2))
        # amplitude
        dg_dA = g / amp
        deriv[0] = dg_dA
        # x0
        if rr != 0:
            dg_dx_mean = g * du_drr * 0.5 * ((2 * a * u) + (b * v))
        else:
            # not well defined at rr=0
            # g comes to a sharp point at rr=0
            # if you approach rr=0 from the left, then the slope is +
            # but if you approach rr=0 from the right, then the slope is -
            # use 0 so it is at least well-behaved
            dg_dx_mean = 0.0
        deriv[1] = dg_dx_mean
        # y0
        if rr != 0:
            dg_dy_mean = g * du_drr * 0.5 * ((2 * c * v) + (b * u))
        else:
            # not well defined at rr=0, see above
            dg_dy_mean = 0.0
        deriv[2] = dg_dy_mean
        if nderiv>3:
            # kserc
            dg_dkserc = -g * rr**alpha
            deriv[3] = dg_dkserc
            # alpha
            if rr != 0:
                dg_dalpha = -g * kserc*np.log(rr) * rr**alpha
            else:
                dg_dalpha = 0.0
            deriv[4] = dg_dalpha
            # recc
            u2 = u ** 2
            v2 = v ** 2
            recc3 = recc**3
            da_drecc = -2*sint2 / recc3
            db_drecc =  2*sin2t / recc3            
            dc_drecc = -2*cost2 / recc3
            if rr==0:
                dg_drecc = 0.0
            else:
                dg_drecc = -g * du_drr * 0.5 * (da_drecc * u2 +
                                                db_drecc * u * v +
                                                dc_drecc * v2)
            deriv[5] = dg_drecc
            # theta
            sint = np.sin(theta)
            cost = np.cos(theta)
            cos2t = np.cos(2.0*theta)
            da_dtheta = (sint * cost * ((1. / ysig2) - (1. / xsig2)))
            db_dtheta = (cos2t / xsig2) - (cos2t / ysig2)            
            dc_dtheta = -da_dtheta
            if rr==0:
                dg_dtheta = 0.0
            else:
                dg_dtheta = -g * du_drr * (da_dtheta * u2 +
                                           db_dtheta * u * v +
                                           dc_dtheta * v2)
            deriv[6] = dg_dtheta

    return [g,deriv]


cpdef double sersic2d_fwhm(double[:] pars):
    """
    Return the FWHM of a 2D Sersic function.

    Parameters
    ----------
    pars : numpy array
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
    cdef double amp,kserc,alpha,recc,rhalf
    cdef double sig_major,sig_minor,mnsig,fwhm

    # pars = [amp,x0,y0,k,alpha,recc,theta]
    # x0,y0 and theta are irrelevant

    amp = pars[0]
    kserc = pars[3]
    alpha = pars[4]
    recc = pars[5]               # b/a

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


cpdef double sersic_b(double n):
    # Normalisation constant
    # bn ~ 2n-1/3 for n>8
    # https://gist.github.com/bamford/b657e3a14c9c567afc4598b1fd10a459
    # n is always positive
    return gammaincinv(2*n, 0.5)
    #return utils.gammaincinv05(2*n)

#
# cpdef create_sersic_function(Ie, re, n):
#     # Not required for integrals - provided for reference
#     # This returns a "closure" function, which is fast to call repeatedly with different radii
#     neg_bn = -b(n)
#     reciprocal_n = 1.0/n
#     f = neg_bn/re**reciprocal_n
#     def sersic_wrapper(r):
#         return Ie * np.exp(f * r ** reciprocal_n - neg_bn)
#     return sersic_wrapper


cpdef double sersic_lum(double Ie, double re, double n):
    # total luminosity (integrated to infinity)
    bn = sersic_b(n)
    #g2n = utils.gamma(2*n)
    g2n = gamma(2*n)
    return Ie * re**2 * 2*pi*n * np.exp(bn)/(bn**(2*n)) * g2n


cpdef list sersic_full2half(double I0, double kserc, double alpha):
    cdef double n, bn, Ie, Re
    # Convert Io and k to Ie and Re
    # Ie = Io * exp(-bn)
    # Re = (bn/k)**n
    n = 1/alpha
    bn = sersic_b(n)
    Ie = I0 * np.exp(-bn)
    Re = (bn/kserc)**n
    return [Ie,Re]


cpdef list sersic_half2full(double Ie, double Re, double alpha):
    cdef double n, bn, I0, kserc
    # Convert Ie and Re to Io and k
    # Ie = Io * exp(-bn)
    # Re = (bn/k)**n
    n = 1/alpha
    bn = sersic_b(n)
    I0 = Ie * np.exp(bn)
    kserc = bn/Re**alpha
    return [I0,kserc]


cpdef double sersic2d_flux(double[:] pars):
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
    cdef double amp,kserc,alpha,recc
    cdef double Ie,Re,n,volume

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

# #
# cpdef sersic2d_estimates(pars):
#     # calculate estimates for the Sersic parameters using
#     # peak, x0, y0, flux, asemi, bsemi, theta
#     # Sersic Parameters are [amp,x0,y0,k,alpha,recc,theta]
#     peak = pars[0]
#     x0 = pars[1]
#     y0 = pars[2]
#     flux = pars[3]
#     asemi = pars[4]
#     bsemi = pars[5]
#     theta = pars[6]
#     recc = bsemi/asemi
    
#     # Calculate FWHM
#     # The mean radius of an ellipse is: (2a+b)/3
#     mnsig = (2.0*asemi+bsemi)/3.0
#     # Convert sigma to FWHM
#     # FWHM = 2*sqrt(2*ln(2))*sig ~ 2.35482*sig
#     fwhm = mnsig*2.35482
#     rhalf = 0.5*fwhm
    
#     # Solve half-max radius equation for kserc
#     # I(R) = I0 * exp(-k*R**alpha) 
#     # 0.5*I0 = I0 * exp(-k*R**alpha)
#     # 0.5 = exp(-k*R**alpha)
#     # ln(0.5) = -k*R**alpha
#     # R = (-ln(0.5)/k)**(1/alpha)
#     # rhalf = (-np.log(0.5)/kserc)**(1/alpha)
#     # kserc = -np.log(0.5)/rhalf**alpha
   
#     # Solve flux equation for kserc
#     # bn = sersic_b(n)
#     # g2n = gamma(2*n)
#     # flux =  recc * Ie * Re**2 * 2*pi*n * np.exp(bn)/(bn**(2*n)) * g2n
#     # Re = np.sqrt(flux/(recc * Ie * 2*pi*n * np.exp(bn)/(bn**(2*n)) * g2n))
#     # kserc = bn/Re**alpha    
#     # kserc = bn * ((recc * Ie * 2*pi*n * np.exp(bn)/(bn**(2*n)) * g2n)/flux)**(alpha/2)

#     # Setting the two equal and then putting everything to one side
#     # 0 = np.log(0.5)/rhalf**alpha + bn * ((recc * Ie * 2*pi*n * np.exp(bn)/(bn**(2*n)) * g2n)/flux)**(alpha/2)
#     cpdef alphafunc(alpha):
#         # rhalf, recc, flux are defined above
#         n = 1/alpha
#         bn = sersic_b(n)
#         g2n = utils.gamma(2*n)
#         Ie,_ = sersic_full2half(peak,1.0,alpha)
#         return np.log(0.5)/rhalf**alpha + bn * ((recc * Ie * 2*pi*n * np.exp(bn)/(bn**(2*n)) * g2n)/flux)**(alpha/2)
    
#     # Solve for the roots
#     res = root_scalar(alphafunc,x0=1.0,x1=0.5)
#     if res.converged:
#         alpha = res.root
#     else:
#         alphas = np.arange(0.1,2.0,0.05)
#         vals = np.zeros(len(alphas),float)
#         for i in range(len(alphas)):
#             vals[i] = alphafunc(alphas[i])
#         bestind = np.argmin(np.abs(vals))
#         alpha = alphas[bestind]
                            
#     # Now solve for ksersic
#     # rhalf = (-np.log(0.5)/kserc)**(1/alpha)
#     kserc = -np.log(0.5)/rhalf**alpha
    
#     # Put all the parameters together
#     spars = [peak,x0,y0,kserc,alpha,recc,theta]
    
#     return spars


#---------------------------------------------------

# Generic model routines


# cpdef double[:] model2d(double x, double y, int psftype, double[:] pars, int nderiv, int osamp):
#     """
#     Two dimensional model function.
    
#     Parameters
#     ----------
#     x : float
#       Single X-value for which to compute the 2D model.
#     y : float
#       Single Y-value of points for which to compute the 2D model.
#     psftype : int
#       Type of PSF model: 1-gaussian, 2-moffat, 3-penny, 4-gausspow, 5-sersic.
#     pars : numpy array
#        Parameter list.
#     nderiv : int
#        The number of derivatives to return.

#     Returns
#     -------
#     g : float
#       The 2D model for the input x/y values and parameters (same
#         shape as x/y).
#     derivative : numpy array
#       Array of derivatives of g relative to the input parameters.

#     Example
#     -------

#     g,derivative = model2d(x,y,1,pars,nderiv)

#     """
#     cdef double asemi,bsemi,theta,xsig,ysig,ysig2,recc,cxx,cyy,cxy
#     cdef double *out1 = <double*>malloc(9 * sizeof(double))
#     out = cvarray(shape=(9,),itemsize=sizeof(double),format="d")
#     cdef double[:] mout = out

#     cdef int npars
#     npars = len(pars)
#     cdef double allpars[11]
#     for i in range(npars):
#         allpars[i] = pars[i]

#     if psftype < 5:
#         xsig = pars[3]
#         ysig = pars[4]
#         theta = pars[5]
#         cxx,cyy,cxy = gauss_abt2cxy(xsig,ysig,theta)
#     else:
#         recc = pars[5]
#         theta = pars[6]
#         #xsig2 = 1.0           # major axis
#         ysig2 = recc ** 2     # minor axis
#         xsig = 1.0
#         ysig = sqrt(ysig2)
#         cxx,cyy,cxy = gauss_abt2cxy(xsig,ysig,theta)
#     allpars[npars] = cxx
#     allpars[npars+1] = cyy
#     allpars[npars+3] = cxy


#     # Gaussian
#     if psftype==1:
#         # pars = [amplitude, x0, y0, xsigma, ysigma, theta]
#         gaussian2d_integrate(x,y,allpars,nderiv,osamp,out1)
#         for i in range(nderiv+1):
#             mout[i] = out1[i]
#         free(out1)
#     # Moffat
#     elif psftype==2:
#         # pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta]
#         moffat2d_integrate(x,y,allpars,nderiv,osamp,out1)
#         for i in range(nderiv+1):
#             mout[i] = out1[i]
#         free(out1)
#     # Penny
#     elif psftype==3:
#         # pars = [amplitude, x0, y0, xsigma, ysigma, theta, relamp, sigma]
#         penny2d_integrate(x,y,allpars,nderiv,osamp,out1)
#         for i in range(nderiv+1):
#             mout[i] = out1[i]
#         free(out1)
#     # Gausspow
#     elif psftype==4:
#         # pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta4, beta6]
#         gausspow2d_integrate(x,y,allpars,nderiv,osamp,out1)
#         for i in range(nderiv+1):
#             mout[i] = out1[i]
#         free(out1)
#     # Sersic
#     elif psftype==5:
#         # pars = [amplitude, x0, y0, kserc, alpha, recc, theta]
#         sersic2d_integrate(x,y,allpars,nderiv,osamp,out1)
#         for i in range(nderiv+1):
#             mout[i] = out1[i]
#         free(out1)
#     else:
#         print('psftype=',psftype,'not supported')
#     return mout

cpdef double[:,:] amodel2d(double[:] x, double[:] y, int psftype, double[:] pars, int nderiv, int osamp):
    """
    Two dimensional model function with x/y array inputs.
    
    Parameters
    ----------
    x : numpy array
      Array of X-values of points for which to compute the 2D model.
    y : numpy array
      Array of Y-values of points for which to compute the 2D model.
    psftype : int
      Type of PSF model: 1-gaussian, 2-moffat, 3-penny, 4-gausspow, 5-sersic.
    pars : numpy array
       Parameter list.
    nderiv : int
       The number of derivatives to return.

    Returns
    -------
    g : numpy array
      The 2D model for the input x/y values and parameters.  Always
        returned as 1D raveled() array.
    derivative : numpy array
      Array of derivatives of g relative to the input parameters.
        Always 2D [Npix,Nderiv] with the 1st dimension being the x/y arrays
        raveled() to 1D.

    Example
    -------

    g,derivative = amodel2d(x,y,1,pars,3)

    """
    cdef int npix = len(x)
    out = cvarray(shape=(npix,9),itemsize=sizeof(double),format="d")
    cdef double[:,:] mout = out

    # Gaussian
    if psftype==1:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta]
        mout = agaussian2d(x,y,pars,nderiv,osamp)
    # Moffat
    elif psftype==2:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta]
        mout = amoffat2d(x,y,pars,nderiv,osamp)
    # Penny
    elif psftype==3:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta, relamp, sigma]
        mout = apenny2d(x,y,pars,nderiv,osamp)
    # Gausspow
    elif psftype==4:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta4, beta6]
        mout = agausspow2d(x,y,pars,nderiv,osamp)
    # Sersic
    elif psftype==5:
        # pars = [amplitude, x0, y0, kserc, alpha, recc, theta]
        mout = asersic2d(x,y,pars,nderiv,osamp)
    else:
        printf("psftype= %d not supported\n",psftype)

    return mout

cpdef double[:,:] amodel2d2(double[:] x, double[:] y, int psftype, double[:] pars, int nderiv, int osamp):
    """
    Two dimensional model function with x/y array inputs.
    
    Parameters
    ----------
    x : numpy array
      Array of X-values of points for which to compute the 2D model.
    y : numpy array
      Array of Y-values of points for which to compute the 2D model.
    psftype : int
      Type of PSF model: 1-gaussian, 2-moffat, 3-penny, 4-gausspow, 5-sersic.
    pars : numpy array
       Parameter list.
    nderiv : int
       The number of derivatives to return.

    Returns
    -------
    g : numpy array
      The 2D model for the input x/y values and parameters.  Always
        returned as 1D raveled() array.
    derivative : numpy array
      Array of derivatives of g relative to the input parameters.
        Always 2D [Npix,Nderiv] with the 1st dimension being the x/y arrays
        raveled() to 1D.

    Example
    -------

    g,derivative = amodel2d(x,y,1,pars,3)

    """


    # Gaussian
    if psftype==1:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta]
        return agaussian2d(x,y,pars,nderiv,osamp)
    # Moffat
    elif psftype==2:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta]
        return amoffat2d(x,y,pars,nderiv,osamp)
    # Penny
    elif psftype==3:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta, relamp, sigma]
        return apenny2d(x,y,pars,nderiv,osamp)
    # Gausspow
    elif psftype==4:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta4, beta6]
        return agausspow2d(x,y,pars,nderiv,osamp)
    # Sersic
    elif psftype==5:
        # pars = [amplitude, x0, y0, kserc, alpha, recc, theta]
        return asersic2d(x,y,pars,nderiv,osamp)
    #else:
    #    print('psftype=',psftype,'not supported')
    #    return [np.nan,np.nan]



cpdef double model2d_flux(int psftype, double[:] pars):
    """
    Return the flux of a 2D model.

    Parameters
    ----------
    psftype : int
       Type of PSF model: 1-gaussian, 2-moffat, 3-penny, 4-gausspow, 5-sersic.
    pars : numpy array
       Parameter list.

    Returns
    -------
    flux : float
       Total flux or volume of the 2D PSF model.
    
    Example
    -------

    flux = model2d_flux(pars)

    """
    # Gaussian
    if psftype==1:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta]
        return gaussian2d_flux(pars)
    # Moffat
    elif psftype==2:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta]
        return moffat2d_flux(pars)
    # Penny
    elif psftype==3:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta, relamp, sigma]
        return penny2d_flux(pars)
    # Gausspow
    elif psftype==4:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta4, beta6]
        return gausspow2d_flux(pars)
    # Sersic
    elif psftype==5:
        # pars = [amplitude, x0, y0, kserc, alpha, recc, theta]
        return sersic2d_flux(pars)
    else:
        print('psftype=',psftype,'not supported')
        return np.nan


cpdef double model2d_fwhm(int psftype,double[:] pars):
    """
    Return the fwhm of a 2D model.

    Parameters
    ----------
    psftype : int
       Type of PSF model: 1-gaussian, 2-moffat, 3-penny, 4-gausspow, 5-sersic.
    pars : numpy array
       Parameter list.

    Returns
    -------
    fwhm : float
       FWHM of the 2D PSF model.
    
    Example
    -------

    fwhm = model2d_fwhm(pars)

    """
    # Gaussian
    if psftype==1:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta]
        fwhm = gaussian2d_fwhm(pars)
    # Moffat
    elif psftype==2:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta]
        fwhm = moffat2d_fwhm(pars)
    # Penny
    elif psftype==3:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta, relamp, sigma]
        fwhm = penny2d_fwhm(pars)
    # Gausspow
    elif psftype==4:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta4, beta6]
        fwhm = gausspow2d_fwhm(pars)
    # Sersic
    elif psftype==5:
        # pars = [amplitude, x0, y0, kserc, alpha, recc, theta]
        fwhm = sersic2d_fwhm(pars)
    else:
        print('psftype=',psftype,'not supported')
        fwhm = np.nan
    return fwhm



cpdef double[:] model2d_estimates(int psftype, double ampc, double xc, double yc):
    """
    Get initial estimates for parameters

    Parameters
    ----------
    psftype : int
       Type of PSF model: 1-gaussian, 2-moffat, 3-penny, 4-gausspow, 5-sersic.
    ampc : float
       Initial guess of amplitude.
    xc : float
       Initial guess of central X coordinate.
    yc : float
       Initial guess of central Y coordinate.

    Returns
    -------
    initpars : numpy array
       The initial estimates
    
    Examples
    --------

    initpars = model2d_estimates(psftype,amp,xc,yc)
    
    """
    cdef int npars
    cdef long[:] npararr
    cdef double[:] initpars
    #npararr = np.zeros(5,long)
    npararr = np.array([6,7,8,8,7])
    npars = npararr[psftype-1]
    initpars = np.zeros(npars,float)
    initpars[0] = ampc
    initpars[1] = xc
    initpars[2] = yc
    # Gaussian
    if psftype==1:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta]
        initpars[3] = 3.5
        initpars[4] = 3.0
        initpars[5] = 0.2
    # Moffat
    elif psftype==2:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta]
        initpars[3] = 3.5
        initpars[4] = 3.0
        initpars[5] = 0.2
        initpars[6] = 2.5
    # Penny
    elif psftype==3:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta, relamp, sigma]
        initpars[3] = 3.5
        initpars[4] = 3.0
        initpars[5] = 0.2
        initpars[6] = 0.1
        initpars[7] = 5.0
    # Gausspow
    elif psftype==4:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta4, beta6]
        #initpars = np.zeros(8,float)
        #initpars[:3] = np.array([ampc,xc,yc])
        initpars[3] = 3.5
        initpars[4] = 3.0
        initpars[5] = 0.2
        initpars[6] = 4.0
        initpars[7] = 6.0
    # Sersic
    elif psftype==5:
        # pars = [amplitude, x0, y0, kserc, alpha, recc, theta]
        initpars[3] = 0.3
        initpars[4] = 0.7
        initpars[5] = 0.2
        initpars[6] = 0.2
    else:
        print('psftype=',psftype,'not supported')
        initpars = np.zeros(7,float)
        initpars[:] = np.nan
    return initpars


cpdef double[:,:] model2d_bounds(int psftype):
    """
    Return upper and lower fitting bounds for the parameters.

    Parameters
    ----------
    psftype : int
       Type of PSF model: 1-gaussian, 2-moffat, 3-penny, 4-gausspow, 5-sersic.
    
    Returns
    -------
    bounds : numpy array
       Upper and lower fitting bounds for each parameter.

    Examples
    --------

    bounds = model2d_bounds(2)
    
    """
    cdef int npars
    cdef long[:] npararr
    cdef double[:] lbounds,ubounds
    cdef double[:,:] bounds

    #npararr = np.zeros(5,int)
    npararr = np.array([6,7,8,8,7])
    npars = npararr[psftype-1]
    #bounds = np.zeros((npars,2),float)
    lbounds = np.zeros(npars,float)
    ubounds = np.zeros(npars,float)
    # Gaussian
    if psftype==1:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta]
        lbounds = np.array([0.00, 0.0, 0.0, 0.1, 0.1, -pi])
        ubounds = np.array([1e30, 1e4, 1e4,  50,  50,  pi])
    # Moffat
    elif psftype==2:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta]
        lbounds = np.array([0.00, 0.0, 0.0, 0.1, 0.1, -pi, 0.1])
        ubounds = np.array([1e30, 1e4, 1e4,  50,  50,  pi, 10])
    # Penny
    elif psftype==3:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta, relamp, sigma]
        lbounds = np.array([0.00, 0.0, 0.0, 0.1, 0.1, -pi, 0.0, 0.1])
        ubounds = np.array([1e30, 1e4, 1e4,  50,  50,  pi, 1.0,  50])
    # Gausspow
    elif psftype==4:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta4, beta6]
        lbounds = np.array([0.00, 0.0, 0.0, 0.1, 0.1, -pi, 0.1, 0.1])
        ubounds = np.array([1e30, 1e4, 1e4,  50,  50,  pi,  50,  50])
    # Sersic
    elif psftype==5:
        # pars = [amplitude, x0, y0, kserc, alpha, recc, theta]
        lbounds = np.array([0.00, 0.0, 0.0, 0.01, 0.02, 0.0, -pi])
        ubounds = np.array([1e30, 1e4, 1e4,   20,  100, 1.0,  pi])
    else:
        print('psftype=',psftype,'not supported')
        bounds = np.zeros((7,2),float)
        bounds[:,:] = np.nan
    bounds = np.array([lbounds,ubounds]).T
    return bounds
    

cpdef double[:] model2d_maxsteps(int psftype, double[:] pars):
    """
    Get maximum steps for parameters.

    Parameters
    ----------
    psftype : int
       Type of PSF model: 1-gaussian, 2-moffat, 3-penny, 4-gausspow, 5-sersic.
    pars : numpy array
       Current best-fit parameters.
    
    Returns
    -------
    maxsteps : numpy array
       Maximum step to allow for each parameter.

    Examples
    --------

    maxsteps = model2d_maxsteps(2,pars)
    
    """
    cdef double[:] maxsteps

    # Gaussian
    if psftype==1:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta]
        maxsteps = np.array([0.5*pars[0],0.5,0.5,0.5,0.5,0.05])
    # Moffat
    elif psftype==2:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta]
        maxsteps = np.array([0.5*pars[0],0.5,0.5,0.5,0.5,0.05,0.03])
    # Penny
    elif psftype==3:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta, relamp, sigma]
        maxsteps = np.array([0.5*pars[0],0.5,0.5,0.5,0.5,0.05,0.01,0.5])
    # Gausspow
    elif psftype==4:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta4, beta6]
        maxsteps = np.array([0.5*pars[0],0.5,0.5,0.5,0.5,0.05,0.5,0.5])
    # Sersic
    elif psftype==5:
        # pars = [amplitude, x0, y0, kserc, alpha, recc, theta]
        maxsteps = np.array([0.5*pars[0],0.5,0.5,0.05,0.1,0.05,0.05])
    else:
        print('psftype=',psftype,'not supported')
        maxsteps = np.zeros(7,float)
        maxsteps[:] = np.nan
    return maxsteps


# cpdef model2dfit(im,err,x,y,psftype,ampc,xc,yc,verbose=False):
#     """
#     Fit all parameters of a single 2D model to data.

#     Parameters
#     ----------
#     im : numpy array
#        Flux array.  Can be 1D or 2D array.
#     err : numpy array
#        Uncertainty array of im.  Same dimensions as im.
#     x : numpy array
#        Array of X-values for im.
#     y : numpy array
#        Array of Y-values for im.
#     psftype : int
#        Type of PSF model: 1-gaussian, 2-moffat, 3-penny, 4-gausspow, 5-sersic.
#     ampc : float
#        Initial guess of amplitude.
#     xc : float
#        Initial guess of central X coordinate.
#     yc : float
#        Initial guess of central Y coordinate.
#     verbose : bool
#        Verbose output to the screen.

#     Returns
#     -------
#     pars : numpy array
#        Best fit pararmeters.
#     perror : numpy array
#        Uncertainties in pars.
#     pcov : numpy array
#        Covariance matrix.
#     flux : float
#        Best fit flux.
#     fluxerr : float
#        Uncertainty in flux.
#     chisq : float
#        Reduced chi-squared of the best-fit.
    
#     Example
#     -------

#     pars,perror,cov,flux,fluxerr,chisq = model2dfit(im,err,x,y,1,100.0,5.5,6.5,False)

#     """

#     maxiter = 10
#     minpercdiff = 0.5

#     if im.ndim==2:
#         im1d = im.ravel()
#         err1d = err.ravel()
#         x1d = x.ravel()
#         y1d = y.ravel()
#     else:
#         im1d = im
#         err1d = err
#         x1d = x
#         y1d = y
#     wt1d = 1/err1d**2
#     npix = len(im1d)
    
#     # Initial values
#     bestpar = model2d_estimates(psftype,ampc,xc,yc)
#     nparams = len(bestpar)
#     nderiv = nparams
#     bounds = model2d_bounds(psftype)

#     if verbose:
#         print('bestpar=',bestpar)
#         print('nderiv=',nderiv)
    
#     # Iteration loop
#     maxpercdiff = 1e10
#     niter = 0
#     while (niter<maxiter and maxpercdiff>minpercdiff):
#         model,deriv = amodel2d(x1d,y1d,psftype,bestpar,nderiv)
#         resid = im1d-model
#         dbeta = utils.qr_jac_solve(deriv,resid,weight=wt1d)
        
#         if verbose:
#             print(niter,bestpar)
#             print(dbeta)
        
#         # Update parameters
#         last_bestpar = bestpar.copy()
#         # limit the steps to the maximum step sizes and boundaries
#         maxsteps = model2d_maxsteps(psftype,bestpar)
#         bestpar = newlsqpars(bestpar,dbeta,bounds,maxsteps)
        
#         # Check differences and changes
#         diff = np.abs(bestpar-last_bestpar)
#         denom = np.maximum(np.abs(bestpar.copy()),0.0001)
#         percdiff = diff.copy()/denom*100  # percent differences
#         maxpercdiff = np.max(percdiff)
#         chisq = np.sum((im1d-model)**2 * wt1d)/npix
#         if verbose:
#             print('chisq=',chisq)
#         #if verbose:
#         #    print(niter,percdiff,chisq)
#         #    print()
#         last_dbeta = dbeta
#         niter += 1

#     model,deriv = amodel2d(x1d,y1d,psftype,bestpar,nderiv)
#     resid = im1d-model
    
#     # Get covariance and errors
#     cov = utils.jac_covariance(deriv,resid,wt1d)
#     perror = np.sqrt(np.diag(cov))

#     # Now get the flux
#     flux = model2d_flux(psftype,bestpar)
#     fluxerr = perror[0]*(flux/bestpar[0]) 
    
#     return bestpar,perror,cov,flux,fluxerr,chisq


# #########################################################################
# # Empirical PSF


cpdef list relcoord(double[:] x, double[:] y, int[:] shape):
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
    midpt = np.array([shape[0]//2,shape[1]//2])
    nx = len(x)
    relx = np.zeros(nx,float)
    rely = np.zeros(nx,float)
    for i in range(nx):
        relx[i] = (x[i]-midpt[1])/shape[1]*2
        rely[i] = (y[i]-midpt[0])/shape[0]*2
    return [relx,rely]


# cpdef list empirical(double[:] x, double[:] y, double[:] pars, double[:,:,:] data,
#                      int[:] imshape, bool deriv):
#     """
#     Evaluate an empirical PSF.

#     Parameters
#     ----------
#     x : numpy array
#       Array of X-values of points for which to compute the empirical model.
#     y : numpy array
#       Array of Y-values of points for which to compute the empirical model.
#     pars : numpy array or list
#        Parameter list.  pars = [amplitude, x0, y0].
#     data : numpy array
#        The empirical PSF information.  This must be in the proper 2D (Ny,Nx)
#          or 3D shape (Ny,Nx,psforder+1).
#     imshape : numpy array
#        The (ny,nx) shape of the full image.  This is needed if the PSF is
#          spatially varying.
#     deriv : boolean, optional
#        Return the derivatives as well.

#     Returns
#     -------
#     g : numpy array
#       The empirical model for the input x/y values and parameters (same
#         shape as x/y).
#     derivative : numpy array
#       Array of derivatives of g relative to the input parameters.
#         This is only returned if deriv=True.

#     Example
#     -------

#     g = empirical(x,y,pars,data)

#     or

#     g,derivative = empirical(x,y,pars,data,deriv=True)

#     """
#     cdef int npsfx,npsfy,npsforder
#     cdef double amp,xc,yc,xoff,yoff,relx,rely
#     cdef double[:] dx,dy,coeff,gxplus,gyplus,g
#     cdef double[:,:] derivative

#     #psftype,pars,npsfx,npsfy,psforder,nxhalf,nyhalf,lookup = unpackpsf(psf)    

#     #if data.ndim != 2 and data.ndim != 3:
#     #    raise Exception('data must be 2D or 3D')
        
#     # Reshape the lookup table
#     #if data.ndim == 2:
#     #    data3d = data.reshape((data.shape[0],data.shape[1],1))
#     #else:
#     #    data3d = data
#     npsfy = data.shape[0]
#     npsfx = data.shape[1]
#     npsforder = data.shape[2]
#     #npsfy,npsfx,npsforder = data.shape
#     if npsfy % 2 == 0 or npsfx % 2 ==0:
#         raise Exception('Empirical PSF dimensions must be odd')
#     npsfyc = npsfy // 2
#     npsfxc = npsfx // 2
    
#     # Parameters for the profile
#     amp = pars[0]
#     xc = pars[1]
#     yc = pars[2]
#     # xc/yc are positions within the large image

#     #if x.ndim==2:
#     #    x1d = x.ravel()
#     #    y1d = y.ravel()
#     #else:
#     #    x1d = x
#     #    y1d = y
#     npix = len(x)
    
#     ## Relative positions
#     #  npsfyc/nsfpxc are the pixel coordinates at the center of
#     #  the lookup table
#     dx = np.zeros(npix,float)
#     dy = np.zeros(npix,float)
#     for i in range(npix):
#         dx[i] = x[i] - xc + npsfxc
#         dy[i] = y[i] - yc + npsfyc
    
#     # Higher-order X/Y terms
#     if npsforder>1:
#         relx,rely = relcoord(xc,yc,imshape)
#         coeff = np.array([1.0, relx, rely, relx*rely])
#     else:
#         coeff = np.array([1.0])

#     # Perform the interpolation
#     g = np.zeros(npix,float)
#     # We must find the derivative with x0/y0 empirically
#     if deriv:
#         gxplus = np.zeros(npix,float)
#         gyplus = np.zeros(npix,float)        
#         xoff = 0.01
#         yoff = 0.01
#     for i in range(npsforder):
#         # spline is initialized with x,y, z(Nx,Ny)
#         # and evaluated with f(x,y)
#         # since we are using im(Ny,Nx), we have to evalute with f(y,x)
#         g[:] += utils.alinearinterp(data[:,:,i],dx,dy) * coeff[i]
#         #g += farr[i](dy,dx,grid=False) * coeff[i]
#         if deriv:
#             gxplus[:] += utils.alinearinterp(data[:,:,i],dx-xoff,dy) * coeff[i]
#             gyplus[:] += utils.alinearinterp(data[:,:,i],dx,dy-yoff) * coeff[i]
#             #gxplus += farr[i](dy,dx-xoff,grid=False) * coeff[i]
#             #gyplus += farr[i](dy-yoff,dx,grid=False) * coeff[i]
#     g *= amp
#     if deriv:
#         gxplus *= amp
#         gyplus *= amp        

#     if deriv is True:
#         # We cannot use np.gradient() because the input x/y values
#         # might not be a regular grid
#         derivative = np.zeros((npix,3),float)
#         derivative[:,0] = g/amp
#         derivative[:,1] = (gxplus-g)/xoff
#         derivative[:,2] = (gyplus-g)/yoff
#     else:
#         derivative = np.zeros((1,1),float)
    
#     return [g,derivative]


# #########################################################################
# # Generic PSF



cpdef list unpackpsf(double[:] psf):
    """
    Unpack the PSF description that are in a 1D array.
    
    Parameters
    ----------
    psf : numpy array
       The PSF model description in a single 1D array.
       If there is no lookup table, then
       pars = [psftype, parameters[
       where the types are: 1-gaussian, 2-moffat, 3-penny, 4-gausspow, 5-sersic, 6-empirical
       if there's a lookup table, then there are extract parameters
       [psftype, parameters, npsfx,npsfy,psforder, nxhalf, nyhalf, raveled lookup table values]
       where nxhalf and nyhalf are the center of the entire image in pixels
       that's only needed when psforder is greater than 0

    Returns
    -------
    psftype : int
       The PSF type type: 1-gaussian, 2-moffat, 3-penny, 4-gausspow, 5-sersic, 6-empirical
    pars : numpy array
       The analytical model parameters.
    lookup : numpy array
       The unraveled lookup table.
    imshape : numpy array
       The size/shape of the whole image.

    Examples
    --------

    psftype,pars,lookup,imshape = unpackpsf(psf)
    
    """
    cdef int npsf,psftype,npars,npsfx,npsfy,psforder,nimx,nimy
    cdef int[:] imshape,npsfarr
    cdef double[:] pars,lookup1d
    cdef double[:,:,:] lookup
    
    npsf = len(psf)
    psftype = int(psf[0])
    imshape = np.zeros(2,np.int32)
    if psftype <= 5:
        nparsarr = np.array([3,4,5,5,4])
        npars = nparsarr[psftype-1]
        pars = psf[1:1+npars]
    else:
        pars = np.zeros(1,float)
        npars = 0
    # There is a lookup table
    if npsf>npars+1 or psftype==6:
        npsfx,npsfy,psforder,nimx,nimy = psf[npars+1:npars+1+5]
        npsfx = int(npsfx)
        npsfy = int(npsfy)
        psforder = int(psforder)
        nimx = int(nimx)
        nimy = int(nimy)
        lookup1d = psf[npars+1+5:]
        imshape[:] = np.array([nimy,nimx])
        # Reshape the lookup table
        lookup = lookup1d.reshape((npsfy,npsfx,psforder+1))
    else:
        lookup = np.zeros((1,1,1),float)
        
    return [psftype,pars,lookup,imshape]


cpdef double[:] packpsf(int psftype, double[:] pars, double[:,:,:] lookup, int[:] imshape):
    """ Put all of the PSF information into a 1D array."""
    cdef int npsfx,npsfy,norder,npsf,psforder
    cdef double[:] psf

    # Figure out how many elements we need in the array
    lshape = lookup.shape
    npsfy = lshape[0]
    npsfx = lshape[1]
    norder = lshape[2]
    lookupsize = lookup.size
    #nlookup = npsfx*npsfy*norder
    psforder = norder-1
    npsf = 1
    if psftype<=5:
        npsf += len(pars)
    if psftype==6 or npsfx>0:
        npsf += 5 + lookup.size
    psf = np.zeros(npsf,float)
    # Add the information
    psf[0] = psftype
    count = 1
    if psftype<=5:
        psf[count:count+len(pars)] = pars
        count += len(pars)
    # Add lookup table/empirical PSF information
    if lookupsize>1:
        #if lookup.ndim==2:
        #    npsfy,npsfx = lookup.shape
        #    psforder = 0
        #else:
        #    npsfy,npsfx,norder = lookup.shape
        #    psforder = norder-1
        if imshape[0] != 0:
            nimy,nimx = imshape
        else:
            nimy,nimx = 0,0
        psf[count:count+5] = np.array([npsfx,npsfy,psforder,nimx,nimy])
        psf[count+5:] = lookup.ravel()
    return psf


# cpdef psfinfo(double[:] psf):
#     """ Print out information about the PSF."""
#     cdef int psftype
#     cdef int[:] imshape
#     cdef double[:] pars
#     cdef double[:,:,:] lookup
    
#     psftype,pars,lookup,imshape = unpackpsf(psf)
#     names = ['Gaussian','Moffat','Penny','Gausspow','Sersic','Empirical']
#     print('PSF type =',psftype,' ('+names[psftype-1]+')')
#     if psftype <=5:
#         print('PARS =',pars)
#     if lookup.size > 1:
#         lshape = lookup.shape
#         npsfy,npsfx = lshape[:2]
#         if len(lshape)==2:
#             psforder = 0
#         else:
#             psforder = lshape[2]-1
#         print('Lookup dims = [',npsfx,npsfy,']')
#         print('PSF order =',psforder)
#         nimy,nimx = imshape
#         if psforder > 0:
#             print('Image size = [',nimx,nimy,']')


# cpdef double psf2d_fwhm(double[:] psf):
#     """
#     Return the FWHM of the PSF
#     """
#     cdef int psftype
#     cdef int[:] imshape
#     cdef double[:] pars,tpars
#     cdef double[:,:,:] lookup
#     cdef double fwhm

#     psftype,psfpars,lookup,imshape = unpackpsf(psf)
#     tpars = np.zeros(3+len(psfpars),float)
#     tpars[0] = 1.0
#     tpars[3:] = psfpars
#     fwhm = model2d_fwhm(psftype,tpars)
#     # Need to make the PSF if it is empirical or has a lookup table
    
#     return fwhm


# cpdef double psf2d_flux(double[:] psf, double amp, double xc, double yc):
#     """
#     Return the flux of the PSF
#     """
#     cdef int psftype
#     cdef double[:] psfpars,tpars
#     cdef double[:,:,:] lookup
#     cdef int[:] imshape
#     cdef double flux

#     psftype,psfpars,lookup,imshape = unpackpsf(psf)
#     tpars = np.zeros(3+len(psfpars),float)
#     tpars[:3] = np.array([amp,xc,yc])
#     tpars[3:] = psfpars
#     flux = model2d_flux(psftype,tpars)
#     # Need to make the PSF if it is empirical or has a lookup table
    
#     return flux


# cpdef list psf(double[:] x, double[:] y, double[:] pars, int psftype, double[:] psfparams,
#                double[:,:,:] lookup, int[:] imshape, int deriv, int verbose):
#     """
#     Return a PSF model.

#     Parameters
#     ----------
#     x : numpy array
#        Array of X-values at which to evaluate the PSF model.
#     y : numpy array
#        Array of Y-values at which to evaluate the PSF model.
#     pars : numpy array
#        Amplitude, xcen, ycen of the model.
#     psftype : int
#        PSF type.
#     psfparams : numpy array
#        Analytical model parameters.
#     lookup : numpy array
#        Empirical lookup table.
#     imshape : numpy array
#        Shape of the whole image.
#     deriv : bool
#        Return derivatives as well.
#     verbose : bool
#        Verbose output to the screen.

#     Returns
#     -------
#     model : numpy array
#        PSF model.
#     derivatives : numpy array
#        Array of partial derivatives.
    
#     Examples
#     --------

#     model,derivatives = psf2d(x,y,pars,1,psfparams,lookup,imshape)

#     """
#     cdef int nderiv,npars
#     cdef int[:] nparsarr
#     cdef double[:] g,allpars
#     cdef double[:,:] derivative

#     # Unpack psf parameters
#     #psftype,psfpars,lookup,imshape = unpackpsf(psf)

#     # Get the analytic portion
#     if deriv==True:
#         nderiv = 3
#     else:
#         nderiv = 0

#     if psftype <= 5:
#         nparsarr = np.zeros(5,np.int32)
#         nparsarr[:] = np.array([6,7,8,8,7])
#         npars = nparsarr[psftype-1]
#         # Add amp, xc, yc to the parameters
#         allpars = np.zeros(npars,float)
#         allpars[:3] = pars
#         allpars[3:] = psfparams
        
#     # Gaussian
#     if psftype==1:
#         g,derivative = agaussian2d(x,y,allpars,nderiv)
#     # Moffat
#     elif psftype==2:
#         g,derivative = amoffat2d(x,y,allpars,nderiv)
#     # Penny
#     elif psftype==3:
#         g,derivative = apenny2d(x,y,allpars,nderiv)
#     # Gausspow
#     elif psftype==4:
#         g,derivative = agausspow2d(x,y,allpars,nderiv)
#     # Sersic
#     elif psftype==5:
#         g,derivative = asersic2d(x,y,allpars,nderiv)
#     # Empirical
#     #elif psftype==6:
#     #    g,derivative = empirical(x,y,pars,lookup,imshape,nderiv)
#     else:
#         print('psftype=',psftype,'not supported')
#         g = np.zeros(1,float)
#         derivative = np.zeros((1,1),float)

#     # Add lookup table portion
#     #if psftype <= 5 and lookup.size > 1:
#     #    eg,ederivative = empirical(x,y,pars,lookup,imshape,(nderiv>0))
#     #    g[:] += eg
#     #    # Make sure the model is positive everywhere
#     #    derivative[:,:] += ederivative
    
#     return [g,derivative]


# cpdef psffit(im,err,x,y,pars,psftype,psfparams,lookup,imshape=None,verbose=False):
#     """
#     Fit a PSF model to data.

#     Parameters
#     ----------
#     im : numpy array
#        Flux array.  Can be 1D or 2D array.
#     err : numpy array
#        Uncertainty array of im.  Same dimensions as im.
#     x : numpy array
#        Array of X-values for im.
#     y : numpy array
#        Array of Y-values for im.
#     pars : numpy array
#        Initial guess of amplitude, xcen, ycen of the model.
#     psftype : int
#        PSF type.
#     psfparams : numpy array
#        Analytical model parameters.
#     lookup : numpy array
#        Empirical lookup table.
#     imshape : numpy array
#        Shape of the whole image.
#     verbose : bool
#        Verbose output to the screen.

#     Returns
#     -------
#     pars : numpy array
#        Best fit pararmeters.
#     perror : numpy array
#        Uncertainties in pars.
#     pcov : numpy array
#        Covariance matrix.
#     flux : float
#        Best fit flux.
#     fluxerr : float
#        Uncertainty in flux.
    
#     Example
#     -------

#     pars,perror,cov,flux,fluxerr = psffit(im,err,x,y,psf,100.0,5.5,6.5,False)

#     """

#     # We are ONLY varying amplitude, xc, and yc

#     maxiter = 10
#     minpercdiff = 0.5

#     if im.ndim==2:
#         im1d = im.ravel()
#         err1d = err.ravel()
#         x1d = x.ravel()
#         y1d = y.ravel()
#     else:
#         im1d = im
#         err1d = err
#         x1d = x
#         y1d = y
#     wt1d = 1/err1d**2
#     npix = len(im1d)
    
#     # Initial values
#     bestpar = np.zeros(3,float)
#     bestpar[:] = pars
#     bounds = np.zeros((3,2),float)
#     bounds[0,:] = [0.0,1e30]
#     bounds[1,:] = [np.maximum(pars[1]-10,np.min(x)),
#                    np.minimum(pars[1]+10,np.max(x))]
#     bounds[2,:] = [np.maximum(pars[2]-10,np.min(y)),
#                    np.minimum(pars[2]+10,np.max(y))]
    
#     if verbose:
#         print('bestpar=',bestpar)
#         print('bounds=',bounds)
        
#     # Iteration loop
#     maxpercdiff = 1e10
#     niter = 0
#     while (niter<maxiter and maxpercdiff>minpercdiff):
#         # psf(x,y,pars,psftype,psfparams,lookup,imshape
#         model,deriv = psf(x1d,y1d,bestpar[:3],psftype,psfparams,lookup,imshape,True)
#         resid = im1d-model
#         dbeta = utils.qr_jac_solve(deriv,resid,weight=wt1d)
        
#         if verbose:
#             print(niter,bestpar)
#             print(dbeta)
        
#         # Update parameters
#         last_bestpar = bestpar.copy()
#         # limit the steps to the maximum step sizes and boundaries
#         maxsteps = np.zeros(3,float)
#         maxsteps[:] = [0.2*bestpar[0],0.5,0.5]
#         bestpar = newlsqpars(bestpar,dbeta,bounds,maxsteps)
        
#         # Check differences and changes
#         diff = np.abs(bestpar-last_bestpar)
#         denom = np.maximum(np.abs(bestpar.copy()),0.0001)
#         percdiff = diff.copy()/denom*100  # percent differences
#         maxpercdiff = np.max(percdiff)
#         chisq = np.sum((im1d-model)**2 * wt1d)/npix
#         if verbose:
#             print('chisq=',chisq)
#         #if verbose:
#         #    print(niter,percdiff,chisq)
#         #    print()
#         last_dbeta = dbeta
#         niter += 1

#     model,deriv = psf(x1d,y1d,bestpar[:3],psftype,psfparams,lookup,imshape,True)
#     #model,deriv = psf2d(x1d,y1d,psf,bestpar[0],bestpar[1],bestpar[2],True)
#     resid = im1d-model
    
#     # Get covariance and errors
#     cov = utils.jac_covariance(deriv,resid,wt1d)
#     perror = np.sqrt(np.diag(cov))

#     # Now get the flux
#     flux = model2d_flux(psftype,bestpar)
#     fluxerr = perror[0]*(flux/bestpar[0]) 
    
#     return bestpar,perror,cov,flux,fluxerr,chisq


# cpdef psf2d(x,y,psf,amp,xc,yc,deriv=False,verbose=False):
#     """
#     Return a PSF model.

#     Parameters
#     ----------
#     x : numpy array
#        Array of X-values at which to evaluate the PSF model.
#     y : numpy array
#        Array of Y-values at which to evaluate the PSF model.
#     psf : int
#        The PSF model description in a single 1D array.
#        If there is no lookup table, then
#        pars = [psftype, parameters[
#        where the types are: 1-gaussian, 2-moffat, 3-penny, 4-gausspow, 5-sersic, 6-empirical
#        if there's a lookup table, then there are extract parameters
#        [psftype, parameters, npsfx,npsfy,psforder, nxhalf, nyhalf, raveled lookup table values]
#        where nxhalf and nyhalf are the center of the entire image in pixels
#        that's only needed when psforder is greater than 0
#     amp : float
#        PSF Amplitude.
#     xc : float
#        PSF central X coordinate.
#     yc : float
#        PSF central Y coordinate.
#     deriv : bool
#        Return derivatives as well.
#     verbose : bool
#        Verbose output to the screen.

#     Returns
#     -------
#     model : numpy array
#        PSF model.
#     derivatives : numpy array
#        Array of partial derivatives.
    
#     Examples
#     --------

#     model,derivatives = psf2d(x,y,psf,100.0,5.5,6.5,True,False)

#     """
    
#     # Unpack psf parameters
#     psftype,psfpars,lookup,imshape = unpackpsf(psf)

#     # Get the analytic portion
#     if deriv==True:
#         nderiv = 3
#     else:
#         nderiv = 0

#     if psftype <= 5:
#         nparsarr = [6,7,8,8,7]
#         npars = nparsarr[psftype-1]
#         # Add amp, xc, yc to the parameters
#         pars = np.zeros(npars,float)
#         pars[:3] = [amp,xc,yc]
#         pars[3:] = psfpars
        
#     # Gaussian
#     if psftype==1:
#         g,derivative = agaussian2d(x,y,pars,nderiv)
#     # Moffat
#     elif psftype==2:
#         g,derivative = amoffat2d(x,y,pars,nderiv)
#     # Penny
#     elif psftype==3:
#         g,derivative = apenny2d(x,y,pars,nderiv)
#     # Gausspow
#     elif psftype==4:
#         g,derivative = agausspow2d(x,y,pars,nderiv)
#     # Sersic
#     elif psftype==5:
#         g,derivative = asersic2d(x,y,pars,nderiv)
#     # Empirical
#     elif psftype==6:
#         g,derivative = empirical(x,y,np.array([amp,xc,yc]),lookup,imshape,nderiv)
#     else:
#         print('psftype=',psftype,'not supported')
#         g = np.zeros(1,float)
#         derivative = np.zeros((1,1),float)

#     # Add lookup table portion
#     if psftype <= 5 and lookup.size > 1:
#         eg,ederivative = empirical(x,y,np.array([amp,xc,yc]),lookup,imshape,(nderiv>0))
#         g[:] += eg
#         # Make sure the model is positive everywhere
#         derivative[:,:] += ederivative
    
#     return g,derivative


# cpdef psf2dfit(im,err,x,y,psf,ampc,xc,yc,verbose=False):
#     """
#     Fit a PSF model to data.

#     Parameters
#     ----------
#     im : numpy array
#        Flux array.  Can be 1D or 2D array.
#     err : numpy array
#        Uncertainty array of im.  Same dimensions as im.
#     x : numpy array
#        Array of X-values for im.
#     y : numpy array
#        Array of Y-values for im.
#     psf : int
#        The PSF model description in a single 1D array.
#        If there is no lookup table, then
#        pars = [psftype, parameters[
#        where the types are: 1-gaussian, 2-moffat, 3-penny, 4-gausspow, 5-sersic, 6-empirical
#        if there's a lookup table, then there are extract parameters
#        [psftype, parameters, npsfx,npsfy,psforder, nxhalf, nyhalf, raveled lookup table values]
#        where nxhalf and nyhalf are the center of the entire image in pixels
#        that's only needed when psforder is greater than 0
#     ampc : float
#        Initial guess of amplitude.
#     xc : float
#        Initial guess of central X coordinate.
#     yc : float
#        Initial guess of central Y coordinate.
#     verbose : bool
#        Verbose output to the screen.

#     Returns
#     -------
#     pars : numpy array
#        Best fit pararmeters.
#     perror : numpy array
#        Uncertainties in pars.
#     pcov : numpy array
#        Covariance matrix.
#     flux : float
#        Best fit flux.
#     fluxerr : float
#        Uncertainty in flux.
    
#     Example
#     -------

#     pars,perror,cov,flux,fluxerr = psffit(im,err,x,y,psf,100.0,5.5,6.5,False)

#     """

#     # We are ONLY varying amplitude, xc, and yc

#     psftype = psf[0]
#     maxiter = 10
#     minpercdiff = 0.5

#     if im.ndim==2:
#         im1d = im.ravel()
#         err1d = err.ravel()
#         x1d = x.ravel()
#         y1d = y.ravel()
#     else:
#         im1d = im
#         err1d = err
#         x1d = x
#         y1d = y
#     wt1d = 1/err1d**2
#     npix = len(im1d)
    
#     # Initial values
#     bestpar = np.zeros(3,float)
#     bestpar[:] = [ampc,xc,yc]
#     bounds = np.zeros((3,2),float)
#     bounds[:,0] = [0.0, -10, -10]
#     bounds[:,1] = [1e30, 10,  10]

#     if verbose:
#         print('bestpar=',bestpar)
    
#     # Iteration loop
#     maxpercdiff = 1e10
#     niter = 0
#     while (niter<maxiter and maxpercdiff>minpercdiff):
#         model,deriv = psf2d(x1d,y1d,psf,bestpar[0],bestpar[1],bestpar[2],True)
#         resid = im1d-model
#         dbeta = utils.qr_jac_solve(deriv,resid,weight=wt1d)
        
#         if verbose:
#             print(niter,bestpar)
#             print(dbeta)
        
#         # Update parameters
#         last_bestpar = bestpar.copy()
#         # limit the steps to the maximum step sizes and boundaries
#         maxsteps = np.zeros(3,float)
#         maxsteps[:] = [0.2*bestpar[0],0.5,0.5]
#         bestpar = newlsqpars(bestpar,dbeta,bounds,maxsteps)
        
#         # Check differences and changes
#         diff = np.abs(bestpar-last_bestpar)
#         denom = np.maximum(np.abs(bestpar.copy()),0.0001)
#         percdiff = diff.copy()/denom*100  # percent differences
#         maxpercdiff = np.max(percdiff)
#         chisq = np.sum((im1d-model)**2 * wt1d)/npix
#         if verbose:
#             print('chisq=',chisq)
#         #if verbose:
#         #    print(niter,percdiff,chisq)
#         #    print()
#         last_dbeta = dbeta
#         niter += 1

#     model,deriv = psf2d(x1d,y1d,psf,bestpar[0],bestpar[1],bestpar[2],True)
#     resid = im1d-model
    
#     # Get covariance and errors
#     cov = utils.jac_covariance(deriv,resid,wt1d)
#     perror = np.sqrt(np.diag(cov))

#     # Now get the flux
#     flux = model2d_flux(psftype,bestpar)
#     fluxerr = perror[0]*(flux/bestpar[0]) 
    
#     return bestpar,perror,cov,flux,fluxerr,chisq


# #########################################################################
# # PSF Classes

# spec = [
#     ('ixmin', types.int32),
#     ('ixmax', types.int32),
#     ('iymin', types.int32),
#     ('iymax', types.int32),
# ]
# @jitclass(spec)
# class BoundingBox(object):

#     def __init__(self, ixmin, ixmax, iymin, iymax):
#         for value in (ixmin, ixmax, iymin, iymax):
#             if not isinstance(value, (int, np.integer)):
#                 raise TypeError('ixmin, ixmax, iymin, and iymax must all be '
#                                 'integers')

#         if ixmin > ixmax:
#             raise ValueError('ixmin must be <= ixmax')
#         if iymin > iymax:
#             raise ValueError('iymin must be <= iymax')

#         self.ixmin = ixmin
#         self.ixmax = ixmax
#         self.iymin = iymin
#         self.iymax = iymax

#     @property
#     def xrange(self):
#         return (self.ixmin,self.ixmax)

#     @property
#     def yrange(self):
#         return (self.iymin,self.iymax)

#     @property
#     def data(self):
#         return [(self.ixmin,self.ixmax),(self.iymin,self.iymax)]

#     def __getitem__(self,item):
#         return self.data[item]

#     def slice(self,array):
#         """ Return slice of array."""
#         return array[self.iymin:self.iymax+1,self.ixmin:self.ixmax+1]

#     def xy(self):
#         """ Return 2D X/Y arrays."""
#         return utils.meshgrid(np.arange(self.ixmin,self.ixmax+1),
#                               np.arange(self.iymin,self.iymax+1))
    
#     def reset(self):
#         """ Forget the original coordinates."""
#         self.ixmax -= self.ixmin
#         self.iymax -= self.iymin
#         self.ixmin = 0
#         self.iymin = 0
    

# # PSF Gaussian class

# spec = [
#     ('npix', types.int32),                # a simple scalar field
#     ('mpars', types.float64[:]),          # an array field
#     ('_params', types.float64[:]),
#     ('radius', types.int32),
#     ('verbose', types.boolean),
#     ('niter', types.int32),
#     ('_unitfootflux', types.float64),
#     ('lookup', types.float64[:,:,:]),
#     ('_bounds', types.float64[:,:]),
#     ('_steps', types.float64[:]),
#     #('labels', types.ListType(types.string)),
# ]

# @jitclass(spec)
# #class PSFGaussian(PSFBase):
# class PSFGaussian(object):    

#     # Initalize the object
#     def __init__(self,mpars=None,npix=51,verbose=False):
#         # MPARS are the model parameters
#         #  mpars = [xsigma, ysigma, theta]
#         if mpars is None:
#             mpars = np.array([1.0,1.0,0.0])
#         if len(mpars)!=3:
#             raise ValueError('3 parameters required')
#         # mpars = [xsigma, ysigma, theta]
#         if mpars[0]<=0 or mpars[1]<=0:
#             raise ValueError('sigma parameters must be >0')

#         # npix must be odd                                                                      
#         if npix%2==0: npix += 1
#         self._params = np.atleast_1d(mpars)
#         #self.binned = binned
#         self.npix = npix
#         self.radius = npix//2
#         self.verbose = verbose
#         self.niter = 0
#         self._unitfootflux = np.nan  # unit flux in footprint                                     
#         self.lookup = np.zeros((npix,npix,3),float)+np.nan
#         # Set the bounds
#         self._bounds = np.zeros((2,3),float)
#         self._bounds[0,:] = [0.0,0.0,-np.inf]
#         self._bounds[1,:] = [np.inf,np.inf,np.inf]
#         # Set step sizes
#         self._steps = np.array([0.5,0.5,0.2])
#         # Labels
#         #self.labels = ['xsigma','ysigma','theta']

#     @property
#     def params(self):
#         """ Return the PSF model parameters."""
#         return self._params

#     @params.setter
#     def params(self,value):
#         """ Set the PSF model parameters."""
#         self._params = value

#     @property
#     def haslookup(self):
#         """ Check if there is a lookup table."""
#         return (np.isfinite(self.lookup[0,0,0])==True)
        
#     def starbbox(self,coords,imshape,radius=np.nan):
#         """                                                                                     
#         Return the boundary box for a star given radius and image size.                         
                                                                                                
#         Parameters                                                                              
#         ----------                                                                              
#         coords: list or tuple                                                                   
#            Central coordinates (xcen,ycen) of star (*absolute* values).                         
#         imshape: list or tuple                                                                  
#             Image shape (ny,nx) values.  Python images are (Y,X).                               
#         radius: float, optional                                                                 
#             Radius in pixels.  Default is psf.npix//2.                                          
                                                                                                
#         Returns                                                                                 
#         -------                                                                                 
#         bbox : BoundingBox object                                                               
#           Bounding box of the x/y ranges.                                                       
#           Upper values are EXCLUSIVE following the python convention.                           
                                                                                                
#         """
#         if np.isfinite(radius)==False:
#             radius = self.npix//2
#         return starbbox(coords,imshape,radius)

#     def bbox2xy(self,bbox):
#         """                                                                                     
#         Convenience method to convert boundary box of X/Y limits to 2-D X and Y arrays.
#         The upper limits are EXCLUSIVE following the python convention.
#         """
#         return bbox2xy(bbox)

#     #def __str__(self):
#     #    """ String representation of the PSF."""
#     #    return 'PSFGaussian('+str(list(self.params))+',npix='+str(self.npix)+',lookup='+str(self.haslookup)+') FWHM='+str(self.fwhm())

#     @property
#     def unitfootflux(self):
#         """ Return the unit flux inside the footprint."""
#         if np.isfinite(self._unitfootflux)==False:
#             xx,yy = utils.meshgrid(np.arange(self.npix),np.arange(self.npix))
#             pars = np.zeros(6,float)
#             pars[0] = 1.0
#             pars[3:] = self.params
#             foot = self.evaluate(xx,yy,pars)
#             self._unitfootflux = np.sum(foot) # sum up footprint flux                         
#         return self._unitfootflux
    
#     def fwhm(self,pars=None):
#         """ Return the FWHM of the model."""
#         if pars is None:
#             pars = np.zeros(6,float)
#             pars[0] = 1.0
#             pars[3:] = self.params
#         return gaussian2d_fwhm(pars)

#     def flux(self,pars=np.array([1.0]),footprint=False):
#         """ Return the flux/volume of the model given the amp or parameters."""
#         if len(pars)==1:
#             amp = pars[0]
#             pars = np.zeros(6,float)
#             pars[0] = amp
#             pars[3:] = self.params
#         if footprint:
#             return self.unitfootflux*pars[0]
#         else:
#             return gaussian2d_flux(pars)        
    
#     def evaluate(self,x, y, pars, deriv=False, nderiv=0):
#         """Two dimensional Gaussian model function"""
#         # pars = [amplitude, x0, y0, xsigma, ysigma, theta]
#         g,_ = agaussian2d(x, y, pars, nderiv=nderiv)
#         return g
    
#     def deriv(self,x, y, pars, binned=None, nderiv=3):
#         """Two dimensional Gaussian model derivative with respect to parameters"""
#         g, derivative = agaussian2d(x, y, pars, nderiv=nderiv)
#         return derivative            


# # Generic PSF class

# spec = [
#     ('psftype', types.int32),
#     ('mpars', types.float64[:]),
#     ('npix', types.int32),
#     ('_params', types.float64[:]),
#     ('radius', types.int32),
#     ('verbose', types.boolean),
#     ('niter', types.int32),
#     ('_unitfootflux', types.float64),
#     ('lookup', types.float64[:,:,:]),
#     ('_bounds', types.float64[:,:]),
#     ('_steps', types.float64[:]),
#     ('coords', types.float64[:]),
#     ('imshape', types.int32[:]),
#     ('order', types.int32),
# ]

# @jitclass(spec)
# class PSF(object):    

#     # Initalize the object
#     def __init__(self,psftype,mpars,npix=51,imshape=np.array([0,0],np.int32),order=0,verbose=False):
#         # MPARS are the model parameters
#         self.psftype = psftype
#         # npix must be odd                                                                      
#         if npix%2==0: npix += 1
#         self._params = np.atleast_1d(mpars)
#         self.npix = npix
#         self.radius = npix//2
#         self.imshape = imshape
#         self.order = 0
#         self.verbose = verbose
#         self.niter = 0
#         self._unitfootflux = np.nan  # unit flux in footprint                                     
#         self.lookup = np.zeros((npix,npix,3),float)+np.nan
#         # Set the bounds
#         self._bounds = np.zeros((2,3),float)
#         self._bounds[0,:] = [0.0,0.0,-np.inf]
#         self._bounds[1,:] = [np.inf,np.inf,np.inf]
#         # Set step sizes
#         self._steps = np.array([0.5,0.5,0.2])
        
#     @property
#     def nparams(self):
#         numparams = [3,4,5,5,4]
#         return numparams[self.psftype-1]
        
#     @property
#     def params(self):
#         """ Return the PSF model parameters."""
#         return self._params

#     @property
#     def name(self):
#         """ Return the name of the PSF type. """
#         if self.psftype==1:
#             return "Gaussian"
#         elif self.psftype==2:
#             return "Moffat"
#         elif self.psftype==3:
#             return "Penny"
#         elif self.psftype==4:
#             return "Gausspow"
#         elif self.psftype==5:
#             return "Sersic"
#         elif self.psftype==6:
#             return "Empirical"
        
#     @params.setter
#     def params(self,value):
#         """ Set the PSF model parameters."""
#         self._params = value

#     @property
#     def haslookup(self):
#         """ Check if there is a lookup table."""
#         return (np.isfinite(self.lookup[0,0,0])==True)
        
#     def starbbox(self,coords,imshape,radius=np.nan):
#         """                                                                                     
#         Return the boundary box for a star given radius and image size.                         
                                                                                                
#         Parameters                                                                              
#         ----------                                                                              
#         coords: list or tuple                                                                   
#            Central coordinates (xcen,ycen) of star (*absolute* values).                         
#         imshape: list or tuple                                                                  
#             Image shape (ny,nx) values.  Python images are (Y,X).                               
#         radius: float, optional                                                                 
#             Radius in pixels.  Default is psf.npix//2.                                          
                                                                                                
#         Returns                                                                                 
#         -------                                                                                 
#         bbox : BoundingBox object                                                               
#           Bounding box of the x/y ranges.                                                       
#           Upper values are EXCLUSIVE following the python convention.                           
                                                                                                
#         """
#         if np.isfinite(radius)==False:
#             radius = self.npix//2
#         return starbbox(coords,imshape,radius)

#     def bbox2xy(self,bbox):
#         """                                                                                     
#         Convenience method to convert boundary box of X/Y limits to 2-D X and Y arrays.
#         The upper limits are EXCLUSIVE following the python convention.
#         """
#         return bbox2xy(bbox)

#     def __str__(self):
#         """ String representation of the PSF."""
#         return 'PSF('+self.name+',npix='+str(self.npix)+',lookup='+str(self.haslookup)+') FWHM='+str(self.fwhm())

#     @property
#     def unitfootflux(self):
#         """ Return the unit flux inside the footprint."""
#         if np.isfinite(self._unitfootflux)==False:
#             xx,yy = utils.meshgrid(np.arange(self.npix),np.arange(self.npix))
#             pars = np.zeros(3,float)
#             pars[0] = 1.0
#             foot = self.model(xx,yy,pars,deriv=False)
#             self._unitfootflux = np.sum(foot) # sum up footprint flux                         
#         return self._unitfootflux
    
#     def fwhm(self,pars=np.array([1.0])):
#         """ Return the FWHM of the model."""
#         tpars = np.zeros(3+self.nparams,float)
#         tpars[0] = pars[0]
#         tpars[3:] = self.params
#         if self.psftype <= 5:
#             fwhm = model2d_fwhm(self.psftype,tpars)
#         else:
#             xx,yy = utils.meshgrid(np.arange(self.npix),np.arange(self.npix))
#             tpars = np.zeros(3,float)
#             tpars[0] = pars[0]
#             foot = self.model(xx,yy,tpars)
#             fwhm = gaussfwhm(foot)
#         return fwhm

#     def flux(self,pars=np.array([1.0]),footprint=False):
#         """ Return the flux/volume of the model given the amp or parameters."""
#         if len(pars)<3:
#             amp = pars[0]
#             pars = np.zeros(3+self.nparams,float)
#             pars[0] = amp
#             pars[3:] = self.params
#         if footprint:
#             return self.unitfootflux*pars[0]
#         else:
#             if self.psftype <= 5:
#                 flux = model2d_flux(self.psftype,pars)
#             else:
#                 xx,yy = utils.meshgrid(np.arange(self.npix),np.arange(self.npix))
#                 tpars = np.zeros(3,float)
#                 tpars[0] = pars[0]
#                 foot = self.model(xx,yy,tpars)
#                 flux = np.sum(foot)
#             return flux   

#     def evaluate(self,x, y, pars):
#         """Two dimensional model function and derivatives"""
#         # Get the analytic portion
#         nderiv = 3
#         if len(pars) != 3:
#             raise Exception('pars must have 3 elements [amp,xc,yc]')
#         #amp,xc,yc = pars
#         if self.psftype <= 5:
#             # Add amp, xc, yc to the parameters
#             allpars = np.zeros(3+self.nparams,float)
#             allpars[:3] = pars
#             allpars[3:] = self.params
#             g,derivative = amodel2d(x,y,self.psftype,allpars,nderiv)
#         elif self.psftype == 6:
#             g,derivative = empirical(x,y,pars,self.lookup,self.imshape,deriv=True)

#         # Add lookup table portion
#         if self.psftype <= 5 and self.haslookup:
#             eg,ederivative = empirical(x,y,pars,self.lookup,self.imshape,deriv=True)
#             g[:] += eg
#             # Make sure the model is positive everywhere
#             derivative[:,:] += ederivative

#         return g,derivative

#     def model(self,x, y, pars, deriv=False):
#         """Two dimensional PSF model."""
#         g, _ = self.evaluate(x, y, pars)
#         return g
        
#     def deriv(self,x, y, pars):
#         """Two dimensional PSF derivative with respect to parameters"""
#         _, derivative = self.evaluate(x, y, pars)
#         return derivative

#     def packpsf(self):
#         """ Return the packed PSF array."""
#         if self.psftype <= 5:
#             if self.haslookup:
#                 return packpsf(self.psftype,self.params,self.lookup,self.imshape)
#             else:
#                 return packpsf(self.psftype,self.params)
#         else:
#             return packpsf(self.psftype,0,self.lookup,self.imshape)

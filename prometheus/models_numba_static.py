import os
import sys
import numpy as np
#from scipy.special import gamma, gammaincinv, gammainc
import scipy.special as sc
from scipy.optimize import root_scalar
#import numba_special  # The import generates Numba overloads for special
#from skimage import measure
from dlnpyutils import utils as dln
import numba
from numba import jit,njit,types
from numba.experimental import jitclass
#from . import leastsquares as lsq, utils_numba as utils
import utils_numba_static as utils
import ladfit_numba_static as ladfit

from numba.pycc import CC
cc = CC('_models_numba_static')

@njit
@cc.export('gaussfwhm', '(f8[:,:],)')
@cc.export('gaussfwhmi', '(i8[:,:],)')
def gaussfwhm(im):
    """
    Use the Gaussian equation Area
    Volume = A*2*pi*sigx*sigy
    to estimate the FWHM.
    """
    volume = np.sum(im)
    ht = np.max(im)
    sigma = np.sqrt(volume/(ht*2*np.pi))
    fwhm = 2.35*sigma
    return fwhm

@njit
@cc.export('hfluxrad', '(f8[:,:],)')
@cc.export('hfluxradi', '(i8[:,:],)')
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
    xx,yy = utils.meshgrid(np.arange(nx)-nx//2,np.arange(ny)-ny//2)
    rr = np.sqrt(xx**2+yy**2)
    si = np.argsort(rr.ravel())
    rsi = rr.ravel()[si]
    fsi = im.ravel()[si]
    totf = np.sum(fsi)
    cumf = np.cumsum(fsi)/totf
    hfluxrad = rsi[np.argmin(np.abs(cumf-0.5))]
    return hfluxrad

# @njit
# @cc.export('contourfwhm', '(f8[:,:],)')
# @cc.export('contourfwhmi', '(i8[:,:],)')
# def contourfwhm(im):
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
#     x = np.arange(nx)-nx//2
#     y = np.arange(ny)-ny//2
#     #xx,yy = utils.meshgridi(np.arange(nx)-nx//2,np.arange(ny)-ny//2)
#     xx,yy = utils.meshgrid(x,y)
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
#             inside = utils.isPointInPolygon(x1,y1,xcen,ycen)
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

@njit
@cc.export('imfwhm', '(f8[:,:],)')
@cc.export('imfwhmi', '(i8[:,:],)')
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
    xx,yy = utils.meshgrid(np.arange(nx)-nx//2,np.arange(ny)-ny//2)
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


@njit
@cc.export('numba_linearinterp', '(f8[:,:],f8[:,:],i4)')
@cc.export('numba_linearinterpi', '(i8[:,:],i8[:,:],i4)')
def numba_linearinterp(binim,fullim,binsize):
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
                    
    # do the edges
    #for i in range(binsize):
    #    fullim[i,:] = fullim[binsize,:]
    #    fullim[:,i] = fullim[:,binsize]
    #    fullim[-i,:] = fullim[-binsize,:]
    #    fullim[:,-i] = fullim[:,-binsize]
    # do the corners
    #fullim[:binsize,:binsize] = binsize[0,0]
    #fullim[-binsize:,:binsize] = binsize[-1,0]
    #fullim[:binsize,-binsize:] = binsize[0,-1]
    #fullim[-binsize:,-binsize:] = binsize[-1,-1]

    return fullim

@njit
@cc.export('checkbounds', 'i4[:](f8[:],f8[:,:])')
@cc.export('checkboundsi', 'i4[:](i8[:],i8[:,:])')
def checkbounds(pars,bounds):
    """ Check the parameters against the bounds."""
    # 0 means it's fine
    # 1 means it's beyond the lower bound
    # 2 means it's beyond the upper bound
    npars = len(pars)
    lbounds = bounds[:,0]
    ubounds = bounds[:,1]
    check = np.zeros(npars,np.int32)
    check[np.where(pars<=lbounds)] = 1
    check[np.where(pars>=ubounds)] = 2
    return check

@njit
@cc.export('limbounds', '(f8[:],f8[:,:])')
@cc.export('limboundsi', '(i8[:],i8[:,:])')
def limbounds(pars,bounds):
    """ Limit the parameters to the boundaries."""
    lbounds = bounds[:,0]
    ubounds = bounds[:,1]
    outpars = np.minimum(np.maximum(pars,lbounds),ubounds)
    return outpars

@njit
@cc.export('limsteps', '(f8[:],f8[:])')
@cc.export('limstepsi', '(i8[:],i8[:])')
def limsteps(steps,maxsteps):
    """ Limit the parameter steps to maximum step sizes."""
    signs = np.sign(steps)
    outsteps = np.minimum(np.abs(steps),maxsteps)
    outsteps *= signs
    return outsteps

@njit
@cc.export('newlsqpars', '(f8[:],f8[:],f8[:,:],f8[:])')
def newlsqpars(pars,steps,bounds,maxsteps):
    """ Return new parameters that fit the constraints."""
    # Limit the steps to maxsteps
    limited_steps = limsteps(steps,maxsteps)
        
    # Make sure that these don't cross the boundaries
    lbounds = bounds[:,0]
    ubounds = bounds[:,1]
    check = checkbounds(pars+limited_steps,bounds)
    # Reduce step size for any parameters to go beyond the boundaries
    badpars, = np.where(check!=0)
    # reduce the step sizes until they are within bounds
    newsteps = limited_steps.copy()
    count = 0
    maxiter = 2
    while (len(badpars)>0 and count<=maxiter):
        newsteps[badpars] = newsteps[badpars] / 2
        newcheck = checkbounds(pars+newsteps,bounds)
        badpars, = np.where(newcheck!=0)
        count += 1
            
    # Final parameters
    newparams = pars + newsteps
            
    # Make sure to limit them to the boundaries
    check = checkbounds(newparams,bounds)
    badpars, = np.where(check!=0)
    if len(badpars)>0:
        # add a tiny offset so it doesn't fit right on the boundary
        newparams = np.minimum(np.maximum(newparams,lbounds+1e-30),ubounds-1e-30)
    return newparams

@njit
@cc.export('newbestpars', '(f8[:],f8[:])')
@cc.export('newbestparsi', '(i8[:],i8[:])')
def newbestpars(bestpars,dbeta):
    """ Get new pars from offsets."""
    newpars = np.zeros(3,float)
    maxchange = 0.5
    # Amplitude
    ampmin = bestpars[0]-maxchange*np.abs(bestpars[0])
    ampmin = np.maximum(ampmin,0)
    ampmax = bestpars[0]+np.abs(maxchange*bestpars[0])
    newamp = utils.clip(bestpars[0]+dbeta[0],ampmin,ampmax)
    newpars[0] = newamp
    # Xc, maxchange in pixels
    xmin = bestpars[1]-maxchange
    xmax = bestpars[1]+maxchange
    newx = utils.clip(bestpars[1]+dbeta[1],xmin,xmax)
    newpars[1] = newx
    # Yc
    ymin = bestpars[2]-maxchange
    ymax = bestpars[2]+maxchange
    newy = utils.clip(bestpars[2]+dbeta[2],ymin,ymax)
    newpars[2] = newy
    return newpars

@njit
@cc.export('starbbox', '(UniTuple(f8,2),UniTuple(i8,2),f8)')
@cc.export('starbboxi', '(UniTuple(i8,2),UniTuple(i8,2),f8)')
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

    bb = np.zeros((2,2),np.int64)
    bb[0,:] = [xlo,xhi]
    bb[1,:] = [ylo,yhi]
    return bb
    #return BoundingBox(xlo,xhi,ylo,yhi)

@njit
@cc.export('bbox2xy', '(f8[:,:],)')
@cc.export('bbox2xyi', '(i8[:,:],)')
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
    #if isinstance(bbox,BoundingBox):
    #    x0,x1 = bbox.xrange
    #    y0,y1 = bbox.yrange
    #else:
    x0,x1 = bbox[0]
    y0,y1 = bbox[1]
    dx = np.arange(x0,x1)
    nxpix = len(dx)
    dy = np.arange(y0,y1)
    nypix = len(dy)
    # Python images are (Y,X)                                                                   
    x = dx.reshape(1,-1)+np.zeros(nypix,np.int64).reshape(-1,1)   # broadcasting is faster           
    y = dy.reshape(-1,1)+np.zeros(nxpix,np.int64)
    return x,y


# ###################################################################
# # Numba analytical PSF models

@njit
@cc.export('gauss_abt2cxy', '(f8,f8,f8)')
def gauss_abt2cxy(asemi,bsemi,theta):
    """ Convert asemi/bsemi/theta to cxx/cyy/cxy. """
    # theta in radians
    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    sintheta2 = sintheta**2
    costheta2 = costheta**2
    asemi2 = asemi**2
    bsemi2 = bsemi**2
    cxx = costheta2/asemi2 + sintheta2/bsemi2
    cyy = sintheta2/asemi2 + costheta2/bsemi2
    cxy = 2*costheta*sintheta*(1/asemi2-1/bsemi2)
    return cxx,cyy,cxy

@njit
@cc.export('gauss_cxy2abt', '(f8,f8,f8)')
def gauss_cxy2abt(cxx,cyy,cxy):
    """ Convert asemi/bsemi/theta to cxx/cyy/cxy. """

    # a+c = 1/xstd2 + 1/ystd2
    # b = sin2t * (1/xstd2 + 1/ystd2)
    # tan 2*theta = b/(a-c)
    if cxx==cyy or cxy==0:
        theta = 0.0
    else:
        theta = np.arctan2(cxy,cxx-cyy)/2.0

    if theta==0:
        # a = 1 / xstd2
        # b = 0        
        # c = 1 / ystd2
        xstd = 1/np.sqrt(cxx)
        ystd = 1/np.sqrt(cyy)
        return xstd,ystd,theta        
        
    sin2t = np.sin(2.0*theta)
    # b/sin2t + (a+c) = 2/xstd2
    # xstd2 = 2.0/(b/sin2t + (a+c))
    xstd = np.sqrt( 2.0/(cxy/sin2t + (cxx+cyy)) )

    # a+c = 1/xstd2 + 1/ystd2
    ystd = np.sqrt( 1/(cxx+cyy-1/xstd**2) )

    # theta in radians
    
    return xstd,ystd,theta

# ####### GAUSSIAN ########

@njit
@cc.export('gaussian2d_flux', '(f8[:],)')
def gaussian2d_flux(pars):
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
    # Volume is 2*pi*A*sigx*sigy
    amp = pars[0]
    xsig = pars[3]
    ysig = pars[4]    
    volume = 2*np.pi*amp*xsig*ysig
    return volume

@njit
@cc.export('gaussian2d_fwhm', '(f8[:],)')
def gaussian2d_fwhm(pars):
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
    
    # pars = [amplitude, x0, y0, xsig, ysig, theta]

    # xdiff = x-x0
    # ydiff = y-y0
    # f(x,y) = A*exp(-0.5 * (a*xdiff**2 + b*xdiff*ydiff + c*ydiff**2))

    xsig = pars[3]
    ysig = pars[4]

    # The mean radius of an ellipse is: (2a+b)/3
    sig_major = np.max(np.array([xsig,ysig]))
    sig_minor = np.min(np.array([xsig,ysig]))
    mnsig = (2.0*sig_major+sig_minor)/3.0
    # Convert sigma to FWHM
    # FWHM = 2*sqrt(2*ln(2))*sig ~ 2.35482*sig
    fwhm = mnsig*2.35482

    return fwhm

@njit
@cc.export('agaussian2d', '(f8[:],f8[:],f8[:],i4)')
@cc.export('agaussian2di', '(i8[:],i8[:],f8[:],i4)')
def agaussian2d(x,y,pars,nderiv):
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
    if len(pars)!=6 and len(pars)!=9:
        raise Exception('agaussian2d pars must have either 6 or 9 elements')
    
    if len(pars)==6:
        amp,xc,yc,asemi,bsemi,theta = pars
        cxx,cyy,cxy = gauss_abt2cxy(asemi,bsemi,theta)
        allpars = np.zeros(9,float)
        allpars[:6] = pars
        allpars[6:] = [cxx,cyy,cxy]
    else:
        amp,xc,yc,asemi,bsemi,theta,cxx,cyy,cxy = pars
        allpars = pars

    # Unravel 2D arrays
    if x.ndim==2:
        xx = x.ravel()
        yy = y.ravel()
    else:
        xx = x
        yy = y
    npix = len(xx)
    # Initialize output
    g = np.zeros(npix,float)
    if nderiv>0:
        deriv = np.zeros((npix,nderiv),float)
    else:
        deriv = np.zeros((1,1),float)
    # Loop over the points
    for i in range(npix):
        g1,deriv1 = gaussian2d(xx[i],yy[i],allpars,nderiv)
        g[i] = g1
        if nderiv>0:
            deriv[i,:] = deriv1
    return g,deriv
    
@njit
@cc.export('gaussian2d', '(f8,f8,f8[:],i8)')
@cc.export('gaussian2di', '(i8,i8,f8[:],i8)')
def gaussian2d(x,y,pars,nderiv):
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

    if len(pars)==6:
        amp,xc,yc,asemi,bsemi,theta = pars
        cxx,cyy,cxy = gauss_abt2cxy(asemi,bsemi,theta)
    else:
        amp,xc,yc,asemi,bsemi,theta,cxx,cyy,cxy = pars
    u = (x-xc)
    u2 = u**2
    v = (y-yc)
    v2 = v**2
    # amp = 1/(asemi*bsemi*2*np.pi)
    g = amp * np.exp(-0.5*(cxx*u**2 + cyy*v**2 + cxy*u*v))

    #  pars = [amplitude, x0, y0, xsigma, ysigma, theta]
    deriv = np.zeros(nderiv,float)    
    if nderiv>0:
        # amplitude
        dg_dA = g / amp
        deriv[0] = dg_dA
        # x0
        dg_dx_mean = g * 0.5*((2. * cxx * u) + (cxy * v))
        deriv[1] = dg_dx_mean
        # y0
        dg_dy_mean = g * 0.5*((cxy * u) + (2. * cyy * v))
        deriv[2] = dg_dy_mean
        if nderiv>3:
            sint = np.sin(theta)        
            cost = np.cos(theta)        
            sint2 = sint ** 2
            cost2 = cost ** 2
            sin2t = np.sin(2. * theta)
            # xsig
            asemi2 = asemi ** 2
            asemi3 = asemi ** 3
            da_dxsig = -cost2 / asemi3
            db_dxsig = -sin2t / asemi3
            dc_dxsig = -sint2 / asemi3
            dg_dxsig = g * (-(da_dxsig * u2 +
                              db_dxsig * u * v +
                              dc_dxsig * v2))
            deriv[3] = dg_dxsig
            # ysig
            bsemi2 = bsemi ** 2
            bsemi3 = bsemi ** 3
            da_dysig = -sint2 / bsemi3
            db_dysig = sin2t / bsemi3
            dc_dysig = -cost2 / bsemi3
            dg_dysig = g * (-(da_dysig * u2 +
                              db_dysig * u * v +
                              dc_dysig * v2))
            deriv[4] = dg_dysig
            # dtheta
            if asemi != bsemi:
                cos2t = np.cos(2.0*theta)
                da_dtheta = (sint * cost * ((1. / bsemi2) - (1. / asemi2)))
                db_dtheta = (cos2t / asemi2) - (cos2t / bsemi2)
                dc_dtheta = -da_dtheta
                dg_dtheta = g * (-(da_dtheta * u2 +
                                   db_dtheta * u * v +
                                   dc_dtheta * v2))
                deriv[5] = dg_dtheta

    return g,deriv

@njit
@cc.export('gaussian2dfit', '(f8[:,:],f8[:,:],f8,f8,f8,b1)')
def gaussian2dfit(im,err,ampc,xc,yc,verbose):
    """
    Fit a single Gaussian 2D model to data.

    Parameters
    ----------
    im : numpy array
       Flux array.  Can be 1D or 2D array.
    err : numpy array
       Uncertainty array of im.  Same dimensions as im.
    ampc : float
       Initial guess of amplitude.
    xc : float
       Initial guess of central X coordinate.
    yc : float
       Initial guess of central Y coordinate.
    verbose : bool
       Verbose output to the screen.

    Returns
    -------
    pars : numpy array
       Best fit pararmeters.
    perror : numpy array
       Uncertainties in pars.
    pcov : numpy array
       Covariance matrix.
    flux : float
       Best fit flux.
    fluxerr : float
       Uncertainty in flux.
    
    Example
    -------

    pars,perror,cov,flux,fluxerr = gaussian2dfit(im,err,1,100.0,5.5,6.5,False)

    """

    # xc/yc are with respect to the image origin (0,0)
    
    # Solve for x, y, amplitude and asemi/bsemi/theta

    maxiter = 10
    minpercdiff = 0.5
    
    ny,nx = im.shape
    im1d = im.ravel()

    x2d,y2d = utils.meshgrid(np.arange(nx),np.arange(ny))
    x1d = x2d.ravel()
    y1d = y2d.ravel()
    
    wt = 1/err**2
    wt1d = wt.ravel()

    asemi = 2.5
    bsemi = 2.4
    theta = 0.1

    # theta in radians
    
    # Initial values
    bestpar = np.zeros(6,float)
    bestpar[0] = ampc
    bestpar[1] = xc
    bestpar[2] = yc
    bestpar[3] = asemi
    bestpar[4] = bsemi
    bestpar[5] = theta
    
    # Iteration loop
    maxpercdiff = 1e10
    niter = 0
    while (niter<maxiter and maxpercdiff>minpercdiff):
        model,deriv = agaussian2d(x1d,y1d,bestpar,6)
        resid = im1d-model
        dbeta = utils.qr_jac_solve(deriv,resid,weight=wt1d)
        
        if verbose:
            print(niter,bestpar)
            print(dbeta)
        
        # Update parameters
        last_bestpar = bestpar.copy()
        # limit the steps to the maximum step sizes and boundaries
        #if bounds is not None or maxsteps is not None:
        #    bestpar = newpars(bestpar,dbeta,bounds,maxsteps)
        bounds = np.zeros((6,2),float)
        bounds[:,0] = [0.00, 0, 0, 0.1, 0.1, -180]
        bounds[:,1] = [1e30,nx,ny, nx//2, ny//2, 180]
        maxsteps = np.zeros(6,float)
        maxsteps[:] = [0.5*bestpar[0],0.5,0.5,0.5,0.5,2.0]
        bestpar = newlsqpars(bestpar,dbeta,bounds,maxsteps)
        
        # Check differences and changes
        diff = np.abs(bestpar-last_bestpar)
        denom = np.maximum(np.abs(bestpar.copy()),0.0001)
        percdiff = diff.copy()/denom*100  # percent differences
        maxpercdiff = np.max(percdiff)
        chisq = np.sum((im1d-model)**2 * wt1d)/(nx*ny)
        if verbose:
            print('chisq=',chisq)
        #if verbose:
        #    print(niter,percdiff,chisq)
        #    print()
        last_dbeta = dbeta
        niter += 1

    model,deriv = agaussian2d(x1d,y1d,bestpar,6)
    resid = im1d-model
    
    # Get covariance and errors
    cov = utils.jac_covariance(deriv,resid,wt1d)
    perror = np.sqrt(np.diag(cov))

    # Now get the flux, multiply by the volume of the Gaussian
    asemi,bsemi,theta = bestpar[3],bestpar[4],bestpar[5]
    gvolume = asemi*bsemi*2*np.pi
    flux = bestpar[0]*gvolume
    fluxerr = perror[0]*gvolume

    # USE GAUSSIAN_FLUX
    # pars = [amplitude, x0, y0, xsigma, ysigma, theta]
    flux = gaussian2d_flux(bestpar)
    fluxerr = perror[0]*(flux/bestpar[0]) 
    
    return bestpar,perror,cov,flux,fluxerr


####### MOFFAT ########

@njit
@cc.export('moffat2d_fwhm', '(f8[:],)')
def moffat2d_fwhm(pars):
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

@njit
@cc.export('moffat2d_flux', '(f8[:],)')
def moffat2d_flux(pars):
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


@njit
@cc.export('amoffat2d', '(f8[:],f8[:],f8[:],i4)')
@cc.export('amoffat2di', '(i8[:],i8[:],f8[:],i4)')
def amoffat2d(x,y,pars,nderiv):
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

    if len(pars)!=7 and len(pars)!=10:
        raise Exception('amoffat2d pars must have either 6 or 9 elements')
    
    allpars = np.zeros(10,float)
    if len(pars)==7:
        amp,xc,yc,asemi,bsemi,theta,beta = pars
        cxx,cyy,cxy = gauss_abt2cxy(asemi,bsemi,theta)
        allpars[:7] = pars
        allpars[7:] = [cxx,cyy,cxy]
    else:
        allpars[:] = pars

    # Unravel 2D arrays
    if x.ndim==2:
        xx = x.ravel()
        yy = y.ravel()
    else:
        xx = x
        yy = y
    npix = len(xx)
    # Initialize output
    g = np.zeros(npix,float)
    if nderiv>0:
        deriv = np.zeros((npix,nderiv),float)
    else:
        deriv = np.zeros((1,1),float)
    # Loop over the points
    for i in range(npix):
        g1,deriv1 = moffat2d(xx[i],yy[i],allpars,nderiv)
        g[i] = g1
        if nderiv>0:
            deriv[i,:] = deriv1
    return g,deriv

    
@njit
@cc.export('moffat2d', '(f8,f8,f8[:],i4)')
@cc.export('moffat2di', '(i8,i8,f8[:],i4)')
def moffat2d(x,y,pars,nderiv):
    """
    Two dimensional Moffat model function for a single point.

    Parameters
    ----------
    x : float
      Single X-value for which to compute the Moffat model.
    y : float
      Single Y-value for which to compute the Moffat model.
    pars : numpy array
       Parameter list. pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta]
         The cxx, cyy, cxy parameter can be added to the end so they don't
         have to be computed.
    nderiv : int
       The number of derivatives to return.

    Returns
    -------
    g : float
      The Moffat model for the input x/y values and parameters.
    derivative : numpy array
      Array of derivatives of g relative to the input parameters.

    Example
    -------

    g,derivative = moffat2d(x,y,pars,nderiv)

    """

    if len(pars)==7:
        amp,xc,yc,asemi,bsemi,theta,beta = pars
        cxx,cyy,cxy = gauss_abt2cxy(asemi,bsemi,theta)
    else:
        amp,xc,yc,asemi,bsemi,theta,beta,cxx,cyy,cxy = pars
        
    u = (x-xc)
    u2 = u**2
    v = (y-yc)
    v2 = v**2
    # amp = 1/(asemi*bsemi*2*np.pi)
    rr_gg = (cxx*u**2 + cyy*v**2 + cxy*u*v)
    g = amp * (1 + rr_gg) ** (-beta)
    
    #  pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta]
    deriv = np.zeros(nderiv,float)    
    if nderiv>0:
        # amplitude
        dg_dA = g / amp
        deriv[0] = dg_dA
        # x0
        dg_dx_mean = beta * g/(1+rr_gg) * ((2. * cxx * u) + (cxy * v))
        deriv[1] = dg_dx_mean
        # y0
        dg_dy_mean = beta * g/(1+rr_gg) * ((cxy * u) + (2. * cyy * v))
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
            dg_dxsig = (-beta)*g/(1+rr_gg) * 2*(da_dxsig * u2 +
                                                db_dxsig * u * v +
                                                dc_dxsig * v2)
            deriv[3] = dg_dxsig
            # bsemi/ysig
            bsemi2 = bsemi ** 2
            bsemi3 = bsemi ** 3
            da_dysig = -sint2 / bsemi3
            db_dysig = sin2t / bsemi3
            dc_dysig = -cost2 / bsemi3
            dg_dysig = (-beta)*g/(1+rr_gg) * 2*(da_dysig * u2 +
                                                db_dysig * u * v +
                                                dc_dysig * v2)
            deriv[4] = dg_dysig
            # dtheta
            if asemi != bsemi:
                cos2t = np.cos(2.0*theta)
                da_dtheta = (sint * cost * ((1. / bsemi2) - (1. / asemi2)))
                db_dtheta = (cos2t / asemi2) - (cos2t / bsemi2)
                dc_dtheta = -da_dtheta
                dg_dtheta = (-beta)*g/(1+rr_gg) * 2*(da_dtheta * u2 +
                                                     db_dtheta * u * v +
                                                     dc_dtheta * v2)
                deriv[5] = dg_dtheta
            # beta
            dg_dbeta = -g * np.log(1 + rr_gg)
            deriv[6] = dg_dbeta
                
    return g,deriv

@njit
@cc.export('moffat2dfit', '(f8[:,:],f8[:,:],f8,f8,f8,i4)')
def moffat2dfit(im,err,ampc,xc,yc,verbose):
    """
    Fit a single Moffat 2D model to data.

    Parameters
    ----------
    im : numpy array
       Flux array.  Can be 1D or 2D array.
    err : numpy array
       Uncertainty array of im.  Same dimensions as im.
    ampc : float
       Initial guess of amplitude.
    xc : float
       Initial guess of central X coordinate.
    yc : float
       Initial guess of central Y coordinate.
    verbose : bool
       Verbose output to the screen.

    Returns
    -------
    pars : numpy array
       Best fit pararmeters.
    perror : numpy array
       Uncertainties in pars.
    pcov : numpy array
       Covariance matrix.
    flux : float
       Best fit flux.
    fluxerr : float
       Uncertainty in flux.
    
    Example
    -------

    pars,perror,cov,flux,fluxerr = moffat2dfit(im,err,1,100.0,5.5,6.5,False)

    """

    # xc/yc are with respect to the image origin (0,0)
    
    # Solve for x, y, amplitude and asemi/bsemi/theta

    maxiter = 10
    minpercdiff = 0.5
    
    ny,nx = im.shape
    im1d = im.ravel()

    x2d,y2d = utils.meshgrid(np.arange(nx),np.arange(ny))
    x1d = x2d.ravel()
    y1d = y2d.ravel()
    
    wt = 1/err**2
    wt1d = wt.ravel()

    asemi = 2.5
    bsemi = 2.4
    theta = 0.1
    beta = 2.5

    # theta in radians
    
    # Initial values
    bestpar = np.zeros(7,float)
    bestpar[0] = ampc
    bestpar[1] = xc
    bestpar[2] = yc
    bestpar[3] = asemi
    bestpar[4] = bsemi
    bestpar[5] = theta
    bestpar[6] = beta
    
    # Iteration loop
    maxpercdiff = 1e10
    niter = 0
    while (niter<maxiter and maxpercdiff>minpercdiff):
        model,deriv = amoffat2d(x1d,y1d,bestpar,7)
        resid = im1d-model
        dbeta = utils.qr_jac_solve(deriv,resid,weight=wt1d)
        
        if verbose:
            print(niter,bestpar)
            print(dbeta)
        
        # Update parameters
        last_bestpar = bestpar.copy()
        # limit the steps to the maximum step sizes and boundaries
        #if bounds is not None or maxsteps is not None:
        #    bestpar = newpars(bestpar,dbeta,bounds,maxsteps)
        bounds = np.zeros((7,2),float)
        bounds[:,0] = [0.00, 0, 0, 0.1, 0.1, -180, 0.1]
        bounds[:,1] = [1e30,nx,ny, nx//2, ny//2, 180, 10]
        maxsteps = np.zeros(7,float)
        maxsteps[:] = [0.5*bestpar[0],0.5,0.5,0.5,0.5,2.0,0.5]
        bestpar = newlsqpars(bestpar,dbeta,bounds,maxsteps)
        
        # Check differences and changes
        diff = np.abs(bestpar-last_bestpar)
        denom = np.maximum(np.abs(bestpar.copy()),0.0001)
        percdiff = diff.copy()/denom*100  # percent differences
        maxpercdiff = np.max(percdiff)
        chisq = np.sum((im1d-model)**2 * wt1d)/(nx*ny)
        if verbose:
            print('chisq=',chisq)
        #if verbose:
        #    print(niter,percdiff,chisq)
        #    print()
        last_dbeta = dbeta
        niter += 1

    model,deriv = amoffat2d(x1d,y1d,bestpar,7)
    resid = im1d-model
    
    # Get covariance and errors
    cov = utils.jac_covariance(deriv,resid,wt1d)
    perror = np.sqrt(np.diag(cov))

    # Now get the flux, multiply by the volume of the Gaussian
    #asemi,bsemi,theta = bestpar[3],bestpar[4],bestpar[5]
    #gvolume = asemi*bsemi*2*np.pi
    #flux = bestpar[0]*gvolume
    #fluxerr = perror[0]*gvolume

    # USE MOFFAT_FLUX
    # pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta]
    flux = moffat2d_flux(bestpar)
    fluxerr = perror[0]*(flux/bestpar[0]) 
    
    return bestpar,perror,cov,flux,fluxerr


####### PENNY ########

@njit
@cc.export('penny2d_fwhm', '(f8[:],)')
def penny2d_fwhm(pars):
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

    # [amplitude, x0, y0, xsig, ysig, theta, relative amplitude, sigma]

    amp = pars[0]
    xsig = pars[3]
    ysig = pars[4]
    relamp = utils.clip(pars[6],0.0,1.0)  # 0<relamp<1
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
    x = np.arange( np.min(np.array([gfwhm,mfwhm]))/2.35/2,
                   np.max(np.array([gfwhm,mfwhm])), 0.5)
    f = (1-relamp)*np.exp(-0.5*(x/mnsig)**2) + relamp/(1+(x/sigma)**2)**beta
    hwhm = np.interp(0.5,f[::-1],x[::-1])
    fwhm = 2*hwhm
        
    return fwhm

@njit
@cc.export('penny2d_flux', '(f8[:],)')
def penny2d_flux(pars):
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

@njit
@cc.export('apenny2d', '(f8[:],f8[:],f8[:],i4)')
@cc.export('apenny2di', '(i8[:],i8[:],f8[:],i4)')
def apenny2d(x,y,pars,nderiv):
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

    if len(pars)!=8 and len(pars)!=11:
        raise Exception('apenny2d pars must have either 6 or 9 elements')
    
    allpars = np.zeros(11,float)
    if len(pars)==8:
        amp,xc,yc,asemi,bsemi,theta,relamp,sigma = pars
        cxx,cyy,cxy = gauss_abt2cxy(asemi,bsemi,theta)
        allpars[:8] = pars
        allpars[8:] = [cxx,cyy,cxy]
    else:
        allpars[:] = pars

    # Unravel 2D arrays
    if x.ndim==2:
        xx = x.ravel()
        yy = y.ravel()
    else:
        xx = x
        yy = y
    npix = len(xx)
    # Initialize output
    g = np.zeros(npix,float)
    if nderiv>0:
        deriv = np.zeros((npix,nderiv),float)
    else:
        deriv = np.zeros((1,1),float)
    # Loop over the points
    for i in range(npix):
        g1,deriv1 = penny2d(xx[i],yy[i],allpars,nderiv)
        g[i] = g1
        if nderiv>0:
            deriv[i,:] = deriv1
    return g,deriv

    
@njit
@cc.export('penny2d', '(f8,f8,f8[:],i4)')
@cc.export('penny2di', '(i8,i8,f8[:],i4)')
def penny2d(x,y,pars,nderiv):
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

    if len(pars)==8:
        amp,xc,yc,asemi,bsemi,theta,relamp,sigma = pars
        cxx,cyy,cxy = gauss_abt2cxy(asemi,bsemi,theta)
    else:
        amp,xc,yc,asemi,bsemi,theta,relamp,sigma,cxx,cyy,cxy = pars
        
    u = (x-xc)
    u2 = u**2
    v = (y-yc)
    v2 = v**2
    relamp = utils.clip(relamp,0.0,1.0)  # 0<relamp<1
    # Gaussian component
    g = amp * (1-relamp) * np.exp(-0.5*((cxx * u2) + (cxy * u*v) +
                                        (cyy * v2)))
    # Add Lorentzian/Moffat beta=1.2 wings
    sigma = np.maximum(sigma,0)
    rr_gg = (u2+v2) / sigma ** 2
    beta = 1.2
    l = amp * relamp / (1 + rr_gg)**(beta)
    # Sum of Gaussian + Lorentzian
    f = g + l
    
    #  pars = [amplitude, x0, y0, xsigma, ysigma, theta, relamp, sigma]
    deriv = np.zeros(nderiv,float)    
    if nderiv>0:
        # amplitude
        df_dA = f / amp
        deriv[0] = df_dA
        # x0
        df_dx_mean = ( g * 0.5*((2 * cxx * u) + (cxy * v)) +                           
                       2*beta*l*u/(sigma**2 * (1+rr_gg)) )  
        deriv[1] = df_dx_mean
        # y0
        df_dy_mean = ( g * 0.5*((2 * cyy * v) + (cxy * u)) +
                       2*beta*l*v/(sigma**2 * (1+rr_gg)) ) 
        deriv[2] = df_dy_mean
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
            df_dxsig = g * (-(da_dxsig * u2 +
                              db_dxsig * u * v +
                              dc_dxsig * v2))
            deriv[3] = df_dxsig
            # bsemi/ysig
            bsemi2 = bsemi ** 2
            bsemi3 = bsemi ** 3
            da_dysig = -sint2 / bsemi3
            db_dysig = sin2t / bsemi3
            dc_dysig = -cost2 / bsemi3
            df_dysig = g * (-(da_dysig * u2 +
                              db_dysig * u * v +
                              dc_dysig * v2))
            deriv[4] = df_dysig
            # dtheta
            if asemi != bsemi:
                cos2t = np.cos(2.0*theta)
                da_dtheta = (sint * cost * ((1. / bsemi2) - (1. / asemi2)))
                db_dtheta = (cos2t / asemi2) - (cos2t / bsemi2)
                dc_dtheta = -da_dtheta
                df_dtheta = g * (-(da_dtheta * u2 +
                                   db_dtheta * u * v +
                                   dc_dtheta * v2))
                deriv[5] = df_dtheta
            # relamp
            df_drelamp = -g/(1-relamp) + l/relamp
            deriv[6] = df_drelamp
            # sigma
            df_dsigma = beta*l/(1+rr_gg) * 2*(u2+v2)/sigma**3 
            deriv[7] = df_dsigma
            
    return f,deriv

@njit
@cc.export('penny2dfit', '(f8[:,:],f8[:,:],f8,f8,f8,b1)')
def penny2dfit(im,err,ampc,xc,yc,verbose):
    """
    Fit a single Penny 2D model to data.

    Parameters
    ----------
    im : numpy array
       Flux array.  Can be 1D or 2D array.
    err : numpy array
       Uncertainty array of im.  Same dimensions as im.
    ampc : float
       Initial guess of amplitude.
    xc : float
       Initial guess of central X coordinate.
    yc : float
       Initial guess of central Y coordinate.
    verbose : bool
       Verbose output to the screen.

    Returns
    -------
    pars : numpy array
       Best fit pararmeters.
    perror : numpy array
       Uncertainties in pars.
    pcov : numpy array
       Covariance matrix.
    flux : float
       Best fit flux.
    fluxerr : float
       Uncertainty in flux.
    
    Example
    -------

    pars,perror,cov,flux,fluxerr = penny2dfit(im,err,1,100.0,5.5,6.5,False)

    """
    # xc/yc are with respect to the image origin (0,0)
    
    # Solve for x, y, amplitude and asemi/bsemi/theta

    maxiter = 10
    minpercdiff = 0.5
    
    ny,nx = im.shape
    im1d = im.ravel()

    x2d,y2d = utils.meshgrid(np.arange(nx),np.arange(ny))
    x1d = x2d.ravel()
    y1d = y2d.ravel()
    
    wt = 1/err**2
    wt1d = wt.ravel()

    asemi = 2.5
    bsemi = 2.4
    theta = 0.1
    relamp = 0.2
    sigma = 2*asemi

    # theta in radians
    
    # Initial values
    bestpar = np.zeros(8,float)
    bestpar[0] = ampc
    bestpar[1] = xc
    bestpar[2] = yc
    bestpar[3] = asemi
    bestpar[4] = bsemi
    bestpar[5] = theta
    bestpar[6] = relamp
    bestpar[7] = sigma
    
    # Iteration loop
    maxpercdiff = 1e10
    niter = 0
    while (niter<maxiter and maxpercdiff>minpercdiff):
        model,deriv = apenny2d(x1d,y1d,bestpar,8)
        resid = im1d-model
        dbeta = utils.qr_jac_solve(deriv,resid,weight=wt1d)
        
        if verbose:
            print(niter,bestpar)
            print(dbeta)
        
        # Update parameters
        last_bestpar = bestpar.copy()
        # limit the steps to the maximum step sizes and boundaries
        #if bounds is not None or maxsteps is not None:
        #    bestpar = newpars(bestpar,dbeta,bounds,maxsteps)
        bounds = np.zeros((8,2),float)
        bounds[:,0] = [0.00, 0, 0, 0.1, 0.1, -180, 0.00, 0.1]
        bounds[:,1] = [1e30,nx,ny, nx//2, ny//2, 180, 1, 10]
        maxsteps = np.zeros(8,float)
        maxsteps[:] = [0.5*bestpar[0],0.5,0.5,0.5,0.5,2.0,0.02,0.5]
        bestpar = newlsqpars(bestpar,dbeta,bounds,maxsteps)
        
        # Check differences and changes
        diff = np.abs(bestpar-last_bestpar)
        denom = np.maximum(np.abs(bestpar.copy()),0.0001)
        percdiff = diff.copy()/denom*100  # percent differences
        maxpercdiff = np.max(percdiff)
        chisq = np.sum((im1d-model)**2 * wt1d)/(nx*ny)
        if verbose:
            print('chisq=',chisq)
        #if verbose:
        #    print(niter,percdiff,chisq)
        #    print()
        last_dbeta = dbeta
        niter += 1

    model,deriv = apenny2d(x1d,y1d,bestpar,8)
    resid = im1d-model
    
    # Get covariance and errors
    cov = utils.jac_covariance(deriv,resid,wt1d)
    perror = np.sqrt(np.diag(cov))

    # Now get the flux, multiply by the volume of the Gaussian
    #asemi,bsemi,theta = bestpar[3],bestpar[4],bestpar[5]
    #gvolume = asemi*bsemi*2*np.pi
    #flux = bestpar[0]*gvolume
    #fluxerr = perror[0]*gvolume

    # USE PENNY_FLUX
    # pars = [amplitude, x0, y0, xsigma, ysigma, theta, relamp, sigma]
    flux = penny2d_flux(bestpar)
    fluxerr = perror[0]*(flux/bestpar[0])    
    
    return bestpar,perror,cov,flux,fluxerr


# ####### GAUSSPOW ########

@njit
@cc.export('gausspow2d_fwhm', '(f8[:],)')
def gausspow2d_fwhm(pars):
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
    z2 = 0.5*(x/mnsig)**2
    gxy = (1+z2+0.5*beta4*z2**2+(1.0/6.0)*beta6*z2**3)
    f = amp / gxy

    hwhm = np.interp(0.5,f[::-1],x[::-1])
    fwhm = 2*hwhm
    
    return fwhm

@njit
@cc.export('gausspow2d_flux', '(f8[:],)')
def gausspow2d_flux(pars):
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


@njit
@cc.export('agausspow2d', '(f8[:],f8[:],f8[:],i4)')
@cc.export('agausspow2di', '(i8[:],i8[:],f8[:],i4)')
def agausspow2d(x,y,pars,nderiv):
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

    if len(pars)!=8 and len(pars)!=11:
        raise Exception('agausspow2d pars must have either 6 or 9 elements')
    
    allpars = np.zeros(11,float)
    if len(pars)==8:
        amp,xc,yc,asemi,bsemi,theta,beta4,beta6 = pars
        cxx,cyy,cxy = gauss_abt2cxy(asemi,bsemi,theta)
        allpars[:8] = pars
        allpars[8:] = [cxx,cyy,cxy]
    else:
        allpars[:] = pars

    # Unravel 2D arrays
    if x.ndim==2:
        xx = x.ravel()
        yy = y.ravel()
    else:
        xx = x
        yy = y
    npix = len(xx)
    # Initialize output
    g = np.zeros(npix,float)
    if nderiv>0:
        deriv = np.zeros((npix,nderiv),float)
    else:
        deriv = np.zeros((1,1),float)
    # Loop over the points
    for i in range(npix):
        g1,deriv1 = gausspow2d(xx[i],yy[i],allpars,nderiv)
        g[i] = g1
        if nderiv>0:
            deriv[i,:] = deriv1
    return g,deriv

    
@njit
@cc.export('gausspow2d', '(f8,f8,f8[:],i4)')
@cc.export('gausspow2di', '(i8,i8,f8[:],i4)')
def gausspow2d(x,y,pars,nderiv):
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
            
    return g,deriv

@njit
@cc.export('gausspowd2dfit', '(f8[:,:],f8[:,:],f8,f8,f8,i4)')
def gausspow2dfit(im,err,ampc,xc,yc,verbose):
    """
    Fit a single GaussPOW 2D model to data.

    Parameters
    ----------
    im : numpy array
       Flux array.  Can be 1D or 2D array.
    err : numpy array
       Uncertainty array of im.  Same dimensions as im.
    ampc : float
       Initial guess of amplitude.
    xc : float
       Initial guess of central X coordinate.
    yc : float
       Initial guess of central Y coordinate.
    verbose : bool
       Verbose output to the screen.

    Returns
    -------
    pars : numpy array
       Best fit pararmeters.
    perror : numpy array
       Uncertainties in pars.
    pcov : numpy array
       Covariance matrix.
    flux : float
       Best fit flux.
    fluxerr : float
       Uncertainty in flux.
    
    Example
    -------

    pars,perror,cov,flux,fluxerr = gausspow2dfit(im,err,1,100.0,5.5,6.5,False)

    """

    # xc/yc are with respect to the image origin (0,0)
    
    # Solve for x, y, amplitude and asemi/bsemi/theta

    maxiter = 10
    minpercdiff = 0.5
    
    ny,nx = im.shape
    im1d = im.ravel()

    x2d,y2d = utils.meshgrid(np.arange(nx),np.arange(ny))
    x1d = x2d.ravel()
    y1d = y2d.ravel()
    
    wt = 1/err**2
    wt1d = wt.ravel()

    asemi = 2.5
    bsemi = 2.4
    theta = 0.1
    beta4 = 3.5
    beta6 = 4.5

    # theta in radians
    
    # Initial values
    bestpar = np.zeros(8,float)
    bestpar[0] = ampc
    bestpar[1] = xc
    bestpar[2] = yc
    bestpar[3] = asemi
    bestpar[4] = bsemi
    bestpar[5] = theta
    bestpar[6] = beta4
    bestpar[7] = beta6
    
    # Iteration loop
    maxpercdiff = 1e10
    niter = 0
    while (niter<maxiter and maxpercdiff>minpercdiff):
        model,deriv = agausspow2d(x1d,y1d,bestpar,8)
        resid = im1d-model
        dbeta = utils.qr_jac_solve(deriv,resid,weight=wt1d)
        
        if verbose:
            print(niter,bestpar)
            print(dbeta)
        
        # Update parameters
        last_bestpar = bestpar.copy()
        # limit the steps to the maximum step sizes and boundaries
        #if bounds is not None or maxsteps is not None:
        #    bestpar = newpars(bestpar,dbeta,bounds,maxsteps)
        bounds = np.zeros((8,2),float)
        bounds[:,0] = [0.00, 0, 0, 0.1, 0.1, -180, 0.1, 0.1]
        bounds[:,1] = [1e30,nx,ny, nx//2, ny//2, 180, nx//2, nx//2]
        maxsteps = np.zeros(8,float)
        maxsteps[:] = [0.5*bestpar[0],0.5,0.5,0.5,0.5,2.0,0.5,0.5]
        bestpar = newlsqpars(bestpar,dbeta,bounds,maxsteps)
        
        # Check differences and changes
        diff = np.abs(bestpar-last_bestpar)
        denom = np.maximum(np.abs(bestpar.copy()),0.0001)
        percdiff = diff.copy()/denom*100  # percent differences
        maxpercdiff = np.max(percdiff)
        chisq = np.sum((im1d-model)**2 * wt1d)/(nx*ny)
        if verbose:
            print('chisq=',chisq)
        #if verbose:
        #    print(niter,percdiff,chisq)
        #    print()
        last_dbeta = dbeta
        niter += 1

    model,deriv = agausspow2d(x1d,y1d,bestpar,8)
    resid = im1d-model
    
    # Get covariance and errors
    cov = utils.jac_covariance(deriv,resid,wt1d)
    perror = np.sqrt(np.diag(cov))

    # Now get the flux, multiply by the volume of the Gaussian
    #asemi,bsemi,theta = bestpar[3],bestpar[4],bestpar[5]
    #gvolume = asemi*bsemi*2*np.pi
    #flux = bestpar[0]*gvolume
    #fluxerr = perror[0]*gvolume

    # USE GAUSSPOW_FLUX
    # pars = [amplitude, x0, y0, xsigma, ysigma, theta, relamp, sigma]
    flux = gausspow2d_flux(bestpar)
    fluxerr = perror[0]*(flux/bestpar[0])    
    
    return bestpar,perror,cov,flux,fluxerr


# ####### SERSIC ########

@njit
@cc.export('asersic2d', '(f8[:],f8[:],f8[:],i4)')
@cc.export('asersic2di', '(i8[:],i8[:],f8[:],i4)')
def asersic2d(x,y,pars,nderiv):
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

    if len(pars)!=7:
        raise Exception('aseric2d pars must have 7 elements')
    
    # Unravel 2D arrays
    if x.ndim==2:
        xx = x.ravel()
        yy = y.ravel()
    else:
        xx = x
        yy = y
    npix = len(xx)
    # Initialize output
    g = np.zeros(npix,float)
    if nderiv>0:
        deriv = np.zeros((npix,nderiv),float)
    else:
        deriv = np.zeros((1,1),float)
    # Loop over the points
    for i in range(npix):
        g1,deriv1 = sersic2d(xx[i],yy[i],pars,nderiv)
        g[i] = g1
        if nderiv>0:
            deriv[i,:] = deriv1
    return g,deriv

@njit
@cc.export('sersic2d', '(f8,f8,f8[:],i4)')
@cc.export('sersic2di', '(i8,i8,f8[:],i4)')
def sersic2d(x, y, pars, nderiv):
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
    xdiff = (x-xc)
    xdiff2 = xdiff**2
    ydiff = (y-yc)
    ydiff2 = ydiff**2
    # recc = b/c
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
            dg_dx_mean = g * du_drr * 0.5 * ((2 * a * xdiff) + (b * ydiff))
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
            dg_dy_mean = g * du_drr * 0.5 * ((2 * c * ydiff) + (b * xdiff))
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
            xdiff2 = xdiff ** 2
            ydiff2 = ydiff ** 2
            recc3 = recc**3
            da_drecc = -2*sint2 / recc3
            db_drecc =  2*sin2t / recc3            
            dc_drecc = -2*cost2 / recc3
            if rr==0:
                dg_drecc = 0.0
            else:
                dg_drecc = -g * du_drr * 0.5 * (da_drecc * xdiff2 +
                                                db_drecc * xdiff * ydiff +
                                                dc_drecc * ydiff2)
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
                dg_dtheta = -g * du_drr * (da_dtheta * xdiff2 +
                                           db_dtheta * xdiff * ydiff +
                                           dc_dtheta * ydiff2)
            deriv[6] = dg_dtheta

    return g,deriv

@njit
@cc.export('sersic2d_fwhm', '(f8[:],)')
def sersic2d_fwhm(pars):
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

@njit
@cc.export('sersic_b', '(i4,)')
def sersic_b(n):
    # Normalisation constant
    # bn ~ 2n-1/3 for n>8
    # https://gist.github.com/bamford/b657e3a14c9c567afc4598b1fd10a459
    # n is always positive
    #return gammaincinv(2*n, 0.5)
    return utils.gammaincinv05(2*n)

# #@njit
# @cc.export('agausspow2df', '(f8[:],f8[:],f8[:],i4)')
# def create_sersic_function(Ie, re, n):
#     # Not required for integrals - provided for reference
#     # This returns a "closure" function, which is fast to call repeatedly with different radii
#     neg_bn = -b(n)
#     reciprocal_n = 1.0/n
#     f = neg_bn/re**reciprocal_n
#     def sersic_wrapper(r):
#         return Ie * np.exp(f * r ** reciprocal_n - neg_bn)
#     return sersic_wrapper

@njit
@cc.export('sersic_lum', '(f8,f8,i4)')
def sersic_lum(Ie, re, n):
    # total luminosity (integrated to infinity)
    bn = sersic_b(n)
    g2n = utils.gamma(2*n)
    return Ie * re**2 * 2*np.pi*n * np.exp(bn)/(bn**(2*n)) * g2n

@njit
@cc.export('sersic_full2half', '(f8,f8,f8)')
def sersic_full2half(I0,kserc,alpha):
    # Convert Io and k to Ie and Re
    # Ie = Io * exp(-bn)
    # Re = (bn/k)**n
    n = 1/alpha
    bn = sersic_b(n)
    Ie = I0 * np.exp(-bn)
    Re = (bn/kserc)**n
    return Ie,Re

@njit
@cc.export('sersic_half2full', '(f8,f8,f8)')
def sersic_half2full(Ie,Re,alpha):
    # Convert Ie and Re to Io and k
    # Ie = Io * exp(-bn)
    # Re = (bn/k)**n
    n = 1/alpha
    bn = sersic_b(n)
    I0 = Ie * np.exp(bn)
    kserc = bn/Re**alpha
    return I0,kserc

@njit
@cc.export('sersic2d_flux', '(f8[:],)')
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

# #@njit
# def sersic2d_estimates(pars):
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
#     # flux =  recc * Ie * Re**2 * 2*np.pi*n * np.exp(bn)/(bn**(2*n)) * g2n
#     # Re = np.sqrt(flux/(recc * Ie * 2*np.pi*n * np.exp(bn)/(bn**(2*n)) * g2n))
#     # kserc = bn/Re**alpha    
#     # kserc = bn * ((recc * Ie * 2*np.pi*n * np.exp(bn)/(bn**(2*n)) * g2n)/flux)**(alpha/2)

#     # Setting the two equal and then putting everything to one side
#     # 0 = np.log(0.5)/rhalf**alpha + bn * ((recc * Ie * 2*np.pi*n * np.exp(bn)/(bn**(2*n)) * g2n)/flux)**(alpha/2)
#     def alphafunc(alpha):
#         # rhalf, recc, flux are defined above
#         n = 1/alpha
#         bn = sersic_b(n)
#         g2n = utils.gamma(2*n)
#         Ie,_ = sersic_full2half(peak,1.0,alpha)
#         return np.log(0.5)/rhalf**alpha + bn * ((recc * Ie * 2*np.pi*n * np.exp(bn)/(bn**(2*n)) * g2n)/flux)**(alpha/2)
    
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

@njit
@cc.export('model2d', '(f8,f8,i4,f8[:],i4)')
@cc.export('model2di', '(i8,i8,i4,f8[:],i4)')
def model2d(x,y,psftype,pars,nderiv):
    """
    Two dimensional model function.
    
    Parameters
    ----------
    x : float
      Single X-value for which to compute the 2D model.
    y : float
      Single Y-value of points for which to compute the 2D model.
    psftype : int
      Type of PSF model: 1-gaussian, 2-moffat, 3-penny, 4-gausspow, 5-sersic.
    pars : numpy array
       Parameter list.
    nderiv : int
       The number of derivatives to return.

    Returns
    -------
    g : float
      The 2D model for the input x/y values and parameters (same
        shape as x/y).
    derivative : numpy array
      Array of derivatives of g relative to the input parameters.

    Example
    -------

    g,derivative = model2d(x,y,1,pars,nderiv)

    """
    # Gaussian
    if psftype==1:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta]
        return gaussian2d(x,y,pars,nderiv)
    # Moffat
    elif psftype==2:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta]
        return moffat2d(x,y,pars,nderiv)
    # Penny
    elif psftype==3:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta, relamp, sigma]
        return penny2d(x,y,pars,nderiv)
    # Gausspow
    elif psftype==4:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta4, beta6]
        return gausspow2d(x,y,pars,nderiv)
    # Sersic
    elif psftype==5:
        # pars = [amplitude, x0, y0, kserc, alpha, recc, theta]
        return sersic2d(x,y,pars,nderiv)
    else:
        print('psftype=',psftype,'not supported')
        return

@njit
@cc.export('amodel2d', '(f8[:],f8[:],i4,f8[:],i4)')
@cc.export('amodel2di', '(i8[:],i8[:],i4,f8[:],i4)')
def amodel2d(x,y,psftype,pars,nderiv):
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
        return agaussian2d(x,y,pars,nderiv)
    # Moffat
    elif psftype==2:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta]
        return amoffat2d(x,y,pars,nderiv)
    # Penny
    elif psftype==3:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta, relamp, sigma]
        return apenny2d(x,y,pars,nderiv)
    # Gausspow
    elif psftype==4:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta4, beta6]
        return agausspow2d(x,y,pars,nderiv)
    # Sersic
    elif psftype==5:
        # pars = [amplitude, x0, y0, kserc, alpha, recc, theta]
        return asersic2d(x,y,pars,nderiv)
    else:
        print('psftype=',psftype,'not supported')
        return

@njit
@cc.export('model2d_flux', '(i4,f8[:])')
def model2d_flux(psftype,pars):
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
        return

@njit
@cc.export('model2d_fwhm', '(i4,f8[:])')
def model2d_fwhm(psftype,pars):
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

    return fwhm


@njit
@cc.export('model2d_estimates', '(i4,f8,f8,f8)')
def model2d_estimates(psftype,ampc,xc,yc):
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
    # Gaussian
    if psftype==1:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta]
        initpars = np.zeros(6,float)
        initpars[:3] = [ampc,xc,yc]
        initpars[3:] = [3.5,3.0,0.2]
        return initpars
    # Moffat
    elif psftype==2:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta]
        initpars = np.zeros(7,float)
        initpars[:3] = [ampc,xc,yc]
        initpars[3:] = [3.5,3.0,0.2,2.5]
        return initpars        
    # Penny
    elif psftype==3:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta, relamp, sigma]
        initpars = np.zeros(8,float)
        initpars[:3] = [ampc,xc,yc]
        initpars[3:] = [3.5,3.0,0.2,0.1,5.0]
        return initpars        
    # Gausspow
    elif psftype==4:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta4, beta6]
        initpars = np.zeros(8,float)
        initpars[:3] = [ampc,xc,yc]
        initpars[3:] = [3.5,3.0,0.2,4.0,6.0]
        return initpars        
    # Sersic
    elif psftype==5:
        # pars = [amplitude, x0, y0, kserc, alpha, recc, theta]
        initpars = np.zeros(7,float)
        initpars[:3] = [ampc,xc,yc]
        initpars[3:] = [0.3,0.7,0.2,0.2]
        return initpars        
    else:
        print('psftype=',psftype,'not supported')
        return

@njit
@cc.export('model2d_bounds', '(i4,)')
def model2d_bounds(psftype):
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
    # Gaussian
    if psftype==1:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta]
        bounds = np.zeros((6,2),float)
        bounds[:,0] = [0.00, 0.0, 0.0, 0.1, 0.1, -np.pi]
        bounds[:,1] = [1e30, 1e4, 1e4,  50,  50,  np.pi]
        return bounds
    # Moffat
    elif psftype==2:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta]
        bounds = np.zeros((7,2),float)
        bounds[:,0] = [0.00, 0.0, 0.0, 0.1, 0.1, -np.pi, 0.1]
        bounds[:,1] = [1e30, 1e4, 1e4,  50,  50,  np.pi, 10]
        return bounds
    # Penny
    elif psftype==3:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta, relamp, sigma]
        bounds = np.zeros((8,2),float)
        bounds[:,0] = [0.00, 0.0, 0.0, 0.1, 0.1, -np.pi, 0.0, 0.1]
        bounds[:,1] = [1e30, 1e4, 1e4,  50,  50,  np.pi, 1.0,  50]
        return bounds
    # Gausspow
    elif psftype==4:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta4, beta6]
        bounds = np.zeros((8,2),float)
        bounds[:,0] = [0.00, 0.0, 0.0, 0.1, 0.1, -np.pi, 0.1, 0.1]
        bounds[:,1] = [1e30, 1e4, 1e4,  50,  50,  np.pi,  50,  50]
        return bounds
    # Sersic
    elif psftype==5:
        # pars = [amplitude, x0, y0, kserc, alpha, recc, theta]
        bounds = np.zeros((7,2),float)
        bounds[:,0] = [0.00, 0.0, 0.0, 0.01, 0.02, 0.0, -np.pi]
        bounds[:,1] = [1e30, 1e4, 1e4,   20,  100, 1.0,  np.pi]
        return bounds
    else:
        print('psftype=',psftype,'not supported')
        return
    
@njit
@cc.export('model2d_maxsteps', '(i4,f8[:])')
def model2d_maxsteps(psftype,pars):
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
    # Gaussian
    if psftype==1:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta]
        maxsteps = np.zeros(6,float)
        maxsteps[:] = [0.5*pars[0],0.5,0.5,0.5,0.5,0.05]
        return maxsteps
    # Moffat
    elif psftype==2:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta]
        maxsteps = np.zeros(7,float)
        maxsteps[:] = [0.5*pars[0],0.5,0.5,0.5,0.5,0.05,0.03]
        return maxsteps
    # Penny
    elif psftype==3:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta, relamp, sigma]
        maxsteps = np.zeros(8,float)
        maxsteps[:] = [0.5*pars[0],0.5,0.5,0.5,0.5,0.05,0.01,0.5]
        return maxsteps
    # Gausspow
    elif psftype==4:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta4, beta6]
        maxsteps = np.zeros(8,float)
        maxsteps[:] = [0.5*pars[0],0.5,0.5,0.5,0.5,0.05,0.5,0.5]
        return maxsteps
    # Sersic
    elif psftype==5:
        # pars = [amplitude, x0, y0, kserc, alpha, recc, theta]
        maxsteps = np.zeros(7,float)
        maxsteps[:] = [0.5*pars[0],0.5,0.5,0.05,0.1,0.05,0.05]
        return maxsteps
    else:
        print('psftype=',psftype,'not supported')
        return

@njit
@cc.export('model2dfit', '(f8[:],f8[:],f8[:],f8[:],i8,f8,f8,f8,b1)')
def model2dfit(im,err,x,y,psftype,ampc,xc,yc,verbose=False):
    """
    Fit all parameters of a single 2D model to data.

    Parameters
    ----------
    im : numpy array
       Flux array.  Must be 1D array.
    err : numpy array
       Uncertainty array of im.  Same dimensions as im.
    x : numpy array
       1D array of X-values for im.
    y : numpy array
       1D array of Y-values for im.
    psftype : int
       Type of PSF model: 1-gaussian, 2-moffat, 3-penny, 4-gausspow, 5-sersic.
    ampc : float
       Initial guess of amplitude.
    xc : float
       Initial guess of central X coordinate.
    yc : float
       Initial guess of central Y coordinate.
    verbose : bool
       Verbose output to the screen.

    Returns
    -------
    pars : numpy array
       Best fit pararmeters.
    perror : numpy array
       Uncertainties in pars.
    pcov : numpy array
       Covariance matrix.
    flux : float
       Best fit flux.
    fluxerr : float
       Uncertainty in flux.
    chisq : float
       Reduced chi-squared of the best-fit.
    
    Example
    -------

    pars,perror,cov,flux,fluxerr,chisq = model2dfit(im,err,x,y,1,100.0,5.5,6.5,False)

    """

    maxiter = 10
    minpercdiff = 0.5

    im1d = im.ravel()
    err1d = err.ravel()
    x1d = x.ravel()
    y1d = y.ravel()
    wt1d = 1/err1d**2
    npix = len(im1d)
    
    # Initial values
    bestpar = model2d_estimates(psftype,ampc,xc,yc)
    nparams = len(bestpar)
    nderiv = nparams
    bounds = model2d_bounds(psftype)

    if verbose:
        print('bestpar=',bestpar)
        print('nderiv=',nderiv)
    
    # Iteration loop
    maxpercdiff = 1e10
    niter = 0
    while (niter<maxiter and maxpercdiff>minpercdiff):
        model,deriv = amodel2d(x1d,y1d,psftype,bestpar,nderiv)
        resid = im1d-model
        dbeta = utils.qr_jac_solve(deriv,resid,weight=wt1d)
        
        if verbose:
            print(niter,bestpar)
            print(dbeta)
        
        # Update parameters
        last_bestpar = bestpar.copy()
        # limit the steps to the maximum step sizes and boundaries
        maxsteps = model2d_maxsteps(psftype,bestpar)
        bestpar = newlsqpars(bestpar,dbeta,bounds,maxsteps)
        
        # Check differences and changes
        diff = np.abs(bestpar-last_bestpar)
        denom = np.maximum(np.abs(bestpar.copy()),0.0001)
        percdiff = diff.copy()/denom*100  # percent differences
        maxpercdiff = np.max(percdiff)
        chisq = np.sum((im1d-model)**2 * wt1d)/npix
        if verbose:
            print('chisq=',chisq)
        #if verbose:
        #    print(niter,percdiff,chisq)
        #    print()
        last_dbeta = dbeta
        niter += 1

    model,deriv = amodel2d(x1d,y1d,psftype,bestpar,nderiv)
    resid = im1d-model
    
    # Get covariance and errors
    cov = utils.jac_covariance(deriv,resid,wt1d)
    perror = np.sqrt(np.diag(cov))

    # Now get the flux
    flux = model2d_flux(psftype,bestpar)
    fluxerr = perror[0]*(flux/bestpar[0]) 
    
    return bestpar,perror,cov,flux,fluxerr,chisq


# #########################################################################
# # Empirical PSF

@njit
@cc.export('relcoord', '(f8[:],f8[:],UniTuple(i8,2))')
@cc.export('relcoordi', '(i8[:],i8[:],UniTuple(i8,2))')
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

    midpt = np.array([shape[0]//2,shape[1]//2])
    relx = (x-midpt[1])/shape[1]*2
    rely = (y-midpt[0])/shape[0]*2
    return relx,rely

@njit
@cc.export('empirical', '(f8[:],f8[:],f8[:],f8[:,:,:],UniTuple(i8,2),b1)')
@cc.export('empiricali', '(i8[:],i8[:],f8[:],f8[:,:,:],UniTuple(i8,2),b1)')
def empirical(x, y, pars, data, imshape=None, deriv=False):
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
    data : numpy array
       The empirical PSF information.  This must be in the proper 2D (Ny,Nx)
         or 3D shape (Ny,Nx,psforder+1).
    imshape : numpy array
       The (ny,nx) shape of the full image.  This is needed if the PSF is
         spatially varying.
    deriv : boolean, optional
       Return the derivatives as well.

    Returns
    -------
    g : numpy array
      The empirical model for the input x/y values and parameters (same
        shape as x/y).
    derivative : numpy array
      Array of derivatives of g relative to the input parameters.
        This is only returned if deriv=True.

    Example
    -------

    g = empirical(x,y,pars,data)

    or

    g,derivative = empirical(x,y,pars,data,deriv=True)

    """


    #psftype,pars,npsfx,npsfy,psforder,nxhalf,nyhalf,lookup = unpackpsf(psf)    

    if data.ndim != 2 and data.ndim != 3:
        raise Exception('data must be 2D or 3D')
        
    # Reshape the lookup table
    if data.ndim == 2:
        data3d = data.reshape((data.shape[0],data.shape[1],1))
    else:
        data3d = data
    npsfy,npsfx,npsforder = data3d.shape
    if npsfy % 2 == 0 or npsfx % 2 ==0:
        raise Exception('Empirical PSF dimensions must be odd')
    npsfyc = npsfy // 2
    npsfxc = npsfx // 2
    
    # Parameters for the profile
    amp = pars[0]
    xc = pars[1]
    yc = pars[2]
    # xc/yc are positions within the large image

    if x.ndim==2:
        x1d = x.ravel()
        y1d = y.ravel()
    else:
        x1d = x
        y1d = y
    npix = len(x1d)
    
    ## Relative positions
    #  npsfyc/nsfpxc are the pixel coordinates at the center of
    #  the lookup table
    dx = x1d - xc + npsfxc
    dy = y1d - yc + npsfyc
    
    # Higher-order X/Y terms
    if npsforder>1:
        relx,rely = relcoord(xc,yc,imshape)
        coeff = np.array([1.0, relx, rely, relx*rely])
    else:
        coeff = np.array([1.0])

    # Perform the interpolation
    g = np.zeros(npix,float)
    # We must find the derivative with x0/y0 empirically
    if deriv:
        gxplus = np.zeros(npix,float)
        gyplus = np.zeros(npix,float)        
        xoff = 0.01
        yoff = 0.01
    for i in range(npsforder):
        # spline is initialized with x,y, z(Nx,Ny)
        # and evaluated with f(x,y)
        # since we are using im(Ny,Nx), we have to evalute with f(y,x)
        g[:] += utils.alinearinterp(data3d[:,:,i],dx,dy) * coeff[i]
        #g += farr[i](dy,dx,grid=False) * coeff[i]
        if deriv:
            gxplus[:] += utils.alinearinterp(data3d[:,:,i],dx-xoff,dy) * coeff[i]
            gyplus[:] += utils.alinearinterp(data3d[:,:,i],dx,dy-yoff) * coeff[i]
            #gxplus += farr[i](dy,dx-xoff,grid=False) * coeff[i]
            #gyplus += farr[i](dy-yoff,dx,grid=False) * coeff[i]
    g *= amp
    if deriv:
        gxplus *= amp
        gyplus *= amp        

    if deriv is True:
        # We cannot use np.gradient() because the input x/y values
        # might not be a regular grid
        derivative = np.zeros((npix,3),float)
        derivative[:,0] = g/amp
        derivative[:,1] = (gxplus-g)/xoff
        derivative[:,2] = (gyplus-g)/yoff
    else:
        derivative = np.zeros((1,1),float)
    
    return g,derivative


#########################################################################
# Generic PSF


@njit
@cc.export('unpackpsf', '(f8[:],)')
def unpackpsf(psf):
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
    npsf = len(psf)
    psftype = int(psf[0])
    imshape = np.zeros(2,np.int32)
    if psftype <= 5:
        nparsarr = [3,4,5,5,4]
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
        lookup = psf[npars+1+5:]
        imshape[:] = [nimy,nimx]
        # Reshape the lookup table
        lookup = np.ascontiguousarray(lookup)
        lookup = lookup.reshape((npsfy,npsfx,psforder+1))
    else:
        lookup = np.zeros((1,1,1),float)
        
    return psftype,pars,lookup,imshape

@njit
@cc.export('packpsf', '(i4,f8[:],f8[:,:,:],UniTuple(i8,2))')
def packpsf(psftype,pars,lookup=np.zeros((1,1,1),float),imshape=(0,0)):
    """ Put all of the PSF information into a 1D array."""
    # Figure out how many elements we need in the array
    if lookup.size>1:
        npsfy,npsfx,norder = lookup.shape
        psforder = norder-1
    else:
        npsfy,npsfx,psforder = 0,0,0
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
    if lookup.size>1:
        if lookup.ndim==2:
            npsfy,npsfx = lookup.shape
            psforder = 0
        else:
            npsfy,npsfx,norder = lookup.shape
            psforder = norder-1
        if imshape[0] != 0:
            nimy,nimx = imshape
        else:
            nimy,nimx = 0,0
        psf[count:count+5] = [npsfx,npsfy,psforder,nimx,nimy]
        psf[count+5:] = lookup.ravel()
    return psf

@njit
@cc.export('psfinfo', '(f8[:],)')
def psfinfo(psf):
    """ Print out information about the PSF."""
    psftype,pars,lookup,imshape = unpackpsf(psf)
    names = ['Gaussian','Moffat','Penny','Gausspow','Sersic','Empirical']
    print('PSF type =',psftype,' ('+names[psftype-1]+')')
    if psftype <=5:
        print('PARS =',pars)
    if lookup.size > 1:
        lshape = lookup.shape
        npsfy,npsfx = lshape[:2]
        if len(lshape)==2:
            psforder = 0
        else:
            psforder = lshape[2]-1
        print('Lookup dims = [',npsfx,npsfy,']')
        print('PSF order =',psforder)
        nimy,nimx = imshape
        if psforder > 0:
            print('Image size = [',nimx,nimy,']')

@njit
@cc.export('psf2d_fwhm', '(f8[:],)')
def psf2d_fwhm(psf):
    """
    Return the FWHM of the PSF
    """
    psftype,psfpars,lookup,imshape = unpackpsf(psf)
    tpars = np.zeros(3+len(psfpars),float)
    tpars[0] = 1.0
    tpars[3:] = psfpars
    fwhm = model2d_fwhm(psftype,tpars)
    # Need to make the PSF if it is empirical or has a lookup table
    
    return fwhm

@njit
@cc.export('psf2d_flux', '(f8[:],f8,f8,f8)')
def psf2d_flux(psf,amp,xc,yc):
    """
    Return the flux of the PSF
    """
    psftype,psfpars,lookup,imshape = unpackpsf(psf)
    tpars = np.zeros(3+len(psfpars),float)
    tpars[:3] = [amp,xc,yc]
    tpars[3:] = psfpars
    flux = model2d_flux(psftype,tpars)
    # Need to make the PSF if it is empirical or has a lookup table
    
    return flux

@njit
@cc.export('psf', '(f8[:],f8[:],f8[:],i4,f8[:],f8[:,:,:],UniTuple(i8,2),b1,b1)')
def psf(x,y,pars,psftype,psfparams,lookup,imshape,deriv=False,verbose=False):
    """
    Return a PSF model.

    Parameters
    ----------
    x : numpy array
       Array of X-values at which to evaluate the PSF model.
    y : numpy array
       Array of Y-values at which to evaluate the PSF model.
    pars : numpy array
       Amplitude, xcen, ycen of the model.
    psftype : int
       PSF type.
    psfparams : numpy array
       Analytical model parameters.
    lookup : numpy array
       Empirical lookup table.
    imshape : numpy array
       Shape of the whole image.
    deriv : bool
       Return derivatives as well.
    verbose : bool
       Verbose output to the screen.

    Returns
    -------
    model : numpy array
       PSF model.
    derivatives : numpy array
       Array of partial derivatives.
    
    Examples
    --------

    model,derivatives = psf2d(x,y,pars,1,psfparams,lookup,imshape)

    """
    
    # Unpack psf parameters
    #psftype,psfpars,lookup,imshape = unpackpsf(psf)

    # Get the analytic portion
    if deriv==True:
        nderiv = 3
    else:
        nderiv = 0

    if psftype <= 5:
        nparsarr = np.zeros(5,np.int32)
        nparsarr[:] = [6,7,8,8,7]
        npars = nparsarr[psftype-1]
        # Add amp, xc, yc to the parameters
        allpars = np.zeros(npars,float)
        allpars[:3] = pars
        allpars[3:] = psfparams
        
    # Gaussian
    if psftype==1:
        g,derivative = agaussian2d(x,y,allpars,nderiv)
    # Moffat
    elif psftype==2:
        g,derivative = amoffat2d(x,y,allpars,nderiv)
    # Penny
    elif psftype==3:
        g,derivative = apenny2d(x,y,allpars,nderiv)
    # Gausspow
    elif psftype==4:
        g,derivative = agausspow2d(x,y,allpars,nderiv)
    # Sersic
    elif psftype==5:
        g,derivative = asersic2d(x,y,allpars,nderiv)
    # Empirical
    elif psftype==6:
        g,derivative = empirical(x,y,pars,lookup,imshape,nderiv)
    else:
        print('psftype=',psftype,'not supported')
        g = np.zeros(1,float)
        derivative = np.zeros((1,1),float)

    # Add lookup table portion
    #if psftype <= 5 and lookup.size > 1:
    #    eg,ederivative = empirical(x,y,pars,lookup,imshape,(nderiv>0))
    #    g[:] += eg
    #    # Make sure the model is positive everywhere
    #    derivative[:,:] += ederivative
    
    return g,derivative

@njit
@cc.export('psffit', '(f8[:],f8[:],f8[:],f8[:],f8[:],i4,f8[:],f8[:,:,:],UniTuple(i8,2),b1)')
def psffit(im,err,x,y,pars,psftype,psfparams,lookup,imshape=None,verbose=False):
    """
    Fit a PSF model to data.

    Parameters
    ----------
    im : numpy array
       Flux array.  Must be 1D array.
    err : numpy array
       Uncertainty array of im.  Same dimensions as im.
    x : numpy array
       1D array of X-values for im.
    y : numpy array
       1D array of Y-values for im.
    pars : numpy array
       Initial guess of amplitude, xcen, ycen of the model.
    psftype : int
       PSF type.
    psfparams : numpy array
       Analytical model parameters.
    lookup : numpy array
       Empirical lookup table.
    imshape : numpy array
       Shape of the whole image.
    verbose : bool
       Verbose output to the screen.

    Returns
    -------
    pars : numpy array
       Best fit pararmeters.
    perror : numpy array
       Uncertainties in pars.
    pcov : numpy array
       Covariance matrix.
    flux : float
       Best fit flux.
    fluxerr : float
       Uncertainty in flux.
    
    Example
    -------

    pars,perror,cov,flux,fluxerr = psffit(im,err,x,y,psf,100.0,5.5,6.5,False)

    """

    # We are ONLY varying amplitude, xc, and yc

    maxiter = 10
    minpercdiff = 0.5

    im1d = im
    err1d = err
    x1d = x
    y1d = y
    wt1d = 1/err1d**2
    npix = len(im1d)
    
    # Initial values
    bestpar = np.zeros(3,float)
    bestpar[:] = pars
    bounds = np.zeros((3,2),float)
    bounds[0,:] = [0.0,1e30]
    bounds[1,:] = [np.maximum(pars[1]-10,np.min(x)),
                   np.minimum(pars[1]+10,np.max(x))]
    bounds[2,:] = [np.maximum(pars[2]-10,np.min(y)),
                   np.minimum(pars[2]+10,np.max(y))]
    
    if verbose:
        print('bestpar=',bestpar)
        print('bounds=',bounds)
        
    # Iteration loop
    maxpercdiff = 1e10
    niter = 0
    while (niter<maxiter and maxpercdiff>minpercdiff):
        # psf(x,y,pars,psftype,psfparams,lookup,imshape
        model,deriv = psf(x1d,y1d,bestpar[:3],psftype,psfparams,lookup,imshape,True)
        resid = im1d-model
        dbeta = utils.qr_jac_solve(deriv,resid,weight=wt1d)
        
        if verbose:
            print(niter,bestpar)
            print(dbeta)
        
        # Update parameters
        last_bestpar = bestpar.copy()
        # limit the steps to the maximum step sizes and boundaries
        maxsteps = np.zeros(3,float)
        maxsteps[:] = [0.2*bestpar[0],0.5,0.5]
        bestpar = newlsqpars(bestpar,dbeta,bounds,maxsteps)
        
        # Check differences and changes
        diff = np.abs(bestpar-last_bestpar)
        denom = np.maximum(np.abs(bestpar.copy()),0.0001)
        percdiff = diff.copy()/denom*100  # percent differences
        maxpercdiff = np.max(percdiff)
        chisq = np.sum((im1d-model)**2 * wt1d)/npix
        if verbose:
            print('chisq=',chisq)
        #if verbose:
        #    print(niter,percdiff,chisq)
        #    print()
        last_dbeta = dbeta
        niter += 1

    model,deriv = psf(x1d,y1d,bestpar[:3],psftype,psfparams,lookup,imshape,True)
    #model,deriv = psf2d(x1d,y1d,psf,bestpar[0],bestpar[1],bestpar[2],True)
    resid = im1d-model
    
    # Get covariance and errors
    cov = utils.jac_covariance(deriv,resid,wt1d)
    perror = np.sqrt(np.diag(cov))

    # Now get the flux
    flux = model2d_flux(psftype,bestpar)
    fluxerr = perror[0]*(flux/bestpar[0]) 
    
    return bestpar,perror,cov,flux,fluxerr,chisq

@njit
@cc.export('psf2d', '(f8[:],f8[:],f8[:],f8,f8,f8,b1,b1)')
@cc.export('psf2di', '(i8[:],i8[:],f8[:],f8,f8,f8,b1,b1)')
def psf2d(x,y,psf,amp,xc,yc,deriv=False,verbose=False):
    """
    Return a PSF model.

    Parameters
    ----------
    x : numpy array
       Array of X-values at which to evaluate the PSF model.
    y : numpy array
       Array of Y-values at which to evaluate the PSF model.
    psf : numpy array
       The PSF model description in a single 1D array.
       If there is no lookup table, then
       pars = [psftype, parameters[
       where the types are: 1-gaussian, 2-moffat, 3-penny, 4-gausspow, 5-sersic, 6-empirical
       if there's a lookup table, then there are extract parameters
       [psftype, parameters, npsfx,npsfy,psforder, nxhalf, nyhalf, raveled lookup table values]
       where nxhalf and nyhalf are the center of the entire image in pixels
       that's only needed when psforder is greater than 0
    amp : float
       PSF Amplitude.
    xc : float
       PSF central X coordinate.
    yc : float
       PSF central Y coordinate.
    deriv : bool
       Return derivatives as well.
    verbose : bool
       Verbose output to the screen.

    Returns
    -------
    model : numpy array
       PSF model.
    derivatives : numpy array
       Array of partial derivatives.
    
    Examples
    --------

    model,derivatives = psf2d(x,y,psf,100.0,5.5,6.5,True,False)

    """
    
    # Unpack psf parameters
    psftype,psfpars,lookup,imshape = unpackpsf(psf)

    # Get the analytic portion
    if deriv==True:
        nderiv = 3
    else:
        nderiv = 0

    if psftype <= 5:
        nparsarr = [6,7,8,8,7]
        npars = nparsarr[psftype-1]
        # Add amp, xc, yc to the parameters
        pars = np.zeros(npars,float)
        pars[:3] = [amp,xc,yc]
        pars[3:] = psfpars
        
    # Gaussian
    if psftype==1:
        g,derivative = agaussian2d(x,y,pars,nderiv)
    # Moffat
    elif psftype==2:
        g,derivative = amoffat2d(x,y,pars,nderiv)
    # Penny
    elif psftype==3:
        g,derivative = apenny2d(x,y,pars,nderiv)
    # Gausspow
    elif psftype==4:
        g,derivative = agausspow2d(x,y,pars,nderiv)
    # Sersic
    elif psftype==5:
        g,derivative = asersic2d(x,y,pars,nderiv)
    # Empirical
    elif psftype==6:
        g,derivative = empirical(x,y,np.array([amp,xc,yc]),lookup,imshape,nderiv)
    else:
        print('psftype=',psftype,'not supported')
        g = np.zeros(1,float)
        derivative = np.zeros((1,1),float)

    # Add lookup table portion
    if psftype <= 5 and lookup.size > 1:
        eg,ederivative = empirical(x,y,np.array([amp,xc,yc]),lookup,imshape,(nderiv>0))
        g[:] += eg
        # Make sure the model is positive everywhere
        derivative[:,:] += ederivative
    
    return g,derivative

@njit
@cc.export('psf2dfit', '(f8[:],f8[:],f8[:],f8[:],f8[:],f8,f8,f8,b1)')
def psf2dfit(im,err,x,y,psf,ampc,xc,yc,verbose=False):
    """
    Fit a PSF model to data.

    Parameters
    ----------
    im : numpy array
       Flux array.  Must be 1D array.
    err : numpy array
       Uncertainty array of im.  Same dimensions as im.
    x : numpy array
       1D array of X-values for im.
    y : numpy array
       1D array of Y-values for im.
    psf : numpy array
       The PSF model description in a single 1D array.
       If there is no lookup table, then
       pars = [psftype, parameters[
       where the types are: 1-gaussian, 2-moffat, 3-penny, 4-gausspow, 5-sersic, 6-empirical
       if there's a lookup table, then there are extract parameters
       [psftype, parameters, npsfx,npsfy,psforder, nxhalf, nyhalf, raveled lookup table values]
       where nxhalf and nyhalf are the center of the entire image in pixels
       that's only needed when psforder is greater than 0
    ampc : float
       Initial guess of amplitude.
    xc : float
       Initial guess of central X coordinate.
    yc : float
       Initial guess of central Y coordinate.
    verbose : bool
       Verbose output to the screen.

    Returns
    -------
    pars : numpy array
       Best fit pararmeters.
    perror : numpy array
       Uncertainties in pars.
    pcov : numpy array
       Covariance matrix.
    flux : float
       Best fit flux.
    fluxerr : float
       Uncertainty in flux.
    
    Example
    -------

    pars,perror,cov,flux,fluxerr = psf2dfit(im,err,x,y,psf,100.0,5.5,6.5,False)

    """

    # We are ONLY varying amplitude, xc, and yc

    psftype = psf[0]
    maxiter = 10
    minpercdiff = 0.5

    if im.ndim==2:
        im1d = im.ravel()
        err1d = err.ravel()
        x1d = x.ravel()
        y1d = y.ravel()
    else:
        im1d = im
        err1d = err
        x1d = x
        y1d = y
    wt1d = 1/err1d**2
    npix = len(im1d)
    
    # Initial values
    bestpar = np.zeros(3,float)
    bestpar[:] = [ampc,xc,yc]
    bounds = np.zeros((3,2),float)
    bounds[:,0] = [0.0, -10, -10]
    bounds[:,1] = [1e30, 10,  10]

    if verbose:
        print('bestpar=',bestpar)
    
    # Iteration loop
    maxpercdiff = 1e10
    niter = 0
    while (niter<maxiter and maxpercdiff>minpercdiff):
        model,deriv = psf2d(x1d,y1d,psf,bestpar[0],bestpar[1],bestpar[2],True)
        resid = im1d-model
        dbeta = utils.qr_jac_solve(deriv,resid,weight=wt1d)
        
        if verbose:
            print(niter,bestpar)
            print(dbeta)
        
        # Update parameters
        last_bestpar = bestpar.copy()
        # limit the steps to the maximum step sizes and boundaries
        maxsteps = np.zeros(3,float)
        maxsteps[:] = [0.2*bestpar[0],0.5,0.5]
        bestpar = newlsqpars(bestpar,dbeta,bounds,maxsteps)
        
        # Check differences and changes
        diff = np.abs(bestpar-last_bestpar)
        denom = np.maximum(np.abs(bestpar.copy()),0.0001)
        percdiff = diff.copy()/denom*100  # percent differences
        maxpercdiff = np.max(percdiff)
        chisq = np.sum((im1d-model)**2 * wt1d)/npix
        if verbose:
            print('chisq=',chisq)
        #if verbose:
        #    print(niter,percdiff,chisq)
        #    print()
        last_dbeta = dbeta
        niter += 1

    model,deriv = psf2d(x1d,y1d,psf,bestpar[0],bestpar[1],bestpar[2],True)
    resid = im1d-model
    
    # Get covariance and errors
    cov = utils.jac_covariance(deriv,resid,wt1d)
    perror = np.sqrt(np.diag(cov))

    # Now get the flux
    flux = model2d_flux(psftype,bestpar)
    fluxerr = perror[0]*(flux/bestpar[0]) 
    
    return bestpar,perror,cov,flux,fluxerr,chisq


#########################################################################
# PSF Classes

spec = [
    ('ixmin', types.int32),
    ('ixmax', types.int32),
    ('iymin', types.int32),
    ('iymax', types.int32),
]
@jitclass(spec)
class BoundingBox(object):

    def __init__(self, ixmin, ixmax, iymin, iymax):
        for value in (ixmin, ixmax, iymin, iymax):
            if not isinstance(value, (int, np.integer)):
                raise TypeError('ixmin, ixmax, iymin, and iymax must all be '
                                'integers')

        if ixmin > ixmax:
            raise ValueError('ixmin must be <= ixmax')
        if iymin > iymax:
            raise ValueError('iymin must be <= iymax')

        self.ixmin = ixmin
        self.ixmax = ixmax
        self.iymin = iymin
        self.iymax = iymax

    @property
    def xrange(self):
        return (self.ixmin,self.ixmax)

    @property
    def yrange(self):
        return (self.iymin,self.iymax)

    @property
    def data(self):
        return [(self.ixmin,self.ixmax),(self.iymin,self.iymax)]

    def __getitem__(self,item):
        return self.data[item]

    def slice(self,array):
        """ Return slice of array."""
        return array[self.iymin:self.iymax+1,self.ixmin:self.ixmax+1]

    def xy(self):
        """ Return 2D X/Y arrays."""
        return utils.meshgrid(np.arange(self.ixmin,self.ixmax+1),
                              np.arange(self.iymin,self.iymax+1))
    
    def reset(self):
        """ Forget the original coordinates."""
        self.ixmax -= self.ixmin
        self.iymax -= self.iymin
        self.ixmin = 0
        self.iymin = 0
    

# PSF Gaussian class
spec = [
    ('npix', types.int32),                # a simple scalar field
    ('mpars', types.float64[:]),          # an array field
    ('_params', types.float64[:]),
    ('radius', types.int32),
    ('verbose', types.boolean),
    ('niter', types.int32),
    ('_unitfootflux', types.float64),
    ('lookup', types.float64[:,:,:]),
    ('_bounds', types.float64[:,:]),
    ('_steps', types.float64[:]),
    #('labels', types.ListType(types.string)),
]

@jitclass(spec)
#class PSFGaussian(PSFBase):
class PSFGaussian(object):    

    # Initalize the object
    def __init__(self,mpars=None,npix=51,verbose=False):
        # MPARS are the model parameters
        #  mpars = [xsigma, ysigma, theta]
        if mpars is None:
            mpars = np.array([1.0,1.0,0.0])
        if len(mpars)!=3:
            raise ValueError('3 parameters required')
        # mpars = [xsigma, ysigma, theta]
        if mpars[0]<=0 or mpars[1]<=0:
            raise ValueError('sigma parameters must be >0')

        # npix must be odd                                                                      
        if npix%2==0: npix += 1
        self._params = np.atleast_1d(mpars)
        #self.binned = binned
        self.npix = npix
        self.radius = npix//2
        self.verbose = verbose
        self.niter = 0
        self._unitfootflux = np.nan  # unit flux in footprint                                     
        self.lookup = np.zeros((npix,npix,3),float)+np.nan
        # Set the bounds
        self._bounds = np.zeros((2,3),float)
        self._bounds[0,:] = [0.0,0.0,-np.inf]
        self._bounds[1,:] = [np.inf,np.inf,np.inf]
        # Set step sizes
        self._steps = np.array([0.5,0.5,0.2])
        # Labels
        #self.labels = ['xsigma','ysigma','theta']

    @property
    def params(self):
        """ Return the PSF model parameters."""
        return self._params

    @params.setter
    def params(self,value):
        """ Set the PSF model parameters."""
        self._params = value

    @property
    def haslookup(self):
        """ Check if there is a lookup table."""
        return (np.isfinite(self.lookup[0,0,0])==True)
        
    def starbbox(self,coords,imshape,radius=np.nan):
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
        if np.isfinite(radius)==False:
            radius = self.npix//2
        return starbbox(coords,imshape,radius)

    def bbox2xy(self,bbox):
        """                                                                                     
        Convenience method to convert boundary box of X/Y limits to 2-D X and Y arrays.
        The upper limits are EXCLUSIVE following the python convention.
        """
        return bbox2xy(bbox)

    #def __str__(self):
    #    """ String representation of the PSF."""
    #    return 'PSFGaussian('+str(list(self.params))+',npix='+str(self.npix)+',lookup='+str(self.haslookup)+') FWHM='+str(self.fwhm())

    @property
    def unitfootflux(self):
        """ Return the unit flux inside the footprint."""
        if np.isfinite(self._unitfootflux)==False:
            xx,yy = utils.meshgrid(np.arange(self.npix),np.arange(self.npix))
            pars = np.zeros(6,float)
            pars[0] = 1.0
            pars[3:] = self.params
            foot = self.evaluate(xx,yy,pars)
            self._unitfootflux = np.sum(foot) # sum up footprint flux                         
        return self._unitfootflux
    
    def fwhm(self,pars=None):
        """ Return the FWHM of the model."""
        if pars is None:
            pars = np.zeros(6,float)
            pars[0] = 1.0
            pars[3:] = self.params
        return gaussian2d_fwhm(pars)

    def flux(self,pars=np.array([1.0]),footprint=False):
        """ Return the flux/volume of the model given the amp or parameters."""
        if len(pars)==1:
            amp = pars[0]
            pars = np.zeros(6,float)
            pars[0] = amp
            pars[3:] = self.params
        if footprint:
            return self.unitfootflux*pars[0]
        else:
            return gaussian2d_flux(pars)        
    
    def evaluate(self,x, y, pars, deriv=False, nderiv=0):
        """Two dimensional Gaussian model function"""
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta]
        g,_ = agaussian2d(x, y, pars, nderiv=nderiv)
        return g
    
    def deriv(self,x, y, pars, binned=None, nderiv=3):
        """Two dimensional Gaussian model derivative with respect to parameters"""
        g, derivative = agaussian2d(x, y, pars, nderiv=nderiv)
        return derivative            


# Generic PSF class
spec = [
    ('psftype', types.int32),
    ('mpars', types.float64[:]),
    ('npix', types.int32),
    ('_params', types.float64[:]),
    ('radius', types.int32),
    ('verbose', types.boolean),
    ('niter', types.int32),
    ('_unitfootflux', types.float64),
    ('lookup', types.float64[:,:,:]),
    ('_bounds', types.float64[:,:]),
    ('_steps', types.float64[:]),
    ('coords', types.float64[:]),
    ('imshape', types.int32[:]),
    ('order', types.int32),
]

@jitclass(spec)
class PSF(object):    

    # Initalize the object
    def __init__(self,psftype,mpars,npix=51,imshape=np.array([0,0],np.int32),order=0,verbose=False):
        # MPARS are the model parameters
        self.psftype = psftype
        # npix must be odd                                                                      
        if npix%2==0: npix += 1
        self._params = np.atleast_1d(mpars)
        self.npix = npix
        self.radius = npix//2
        self.imshape = imshape
        self.order = 0
        self.verbose = verbose
        self.niter = 0
        self._unitfootflux = np.nan  # unit flux in footprint                                     
        self.lookup = np.zeros((npix,npix,3),float)+np.nan
        # Set the bounds
        self._bounds = np.zeros((2,3),float)
        self._bounds[0,:] = [0.0,0.0,-np.inf]
        self._bounds[1,:] = [np.inf,np.inf,np.inf]
        # Set step sizes
        self._steps = np.array([0.5,0.5,0.2])
        
    @property
    def nparams(self):
        numparams = [3,4,5,5,4]
        return numparams[self.psftype-1]
        
    @property
    def params(self):
        """ Return the PSF model parameters."""
        return self._params

    @property
    def name(self):
        """ Return the name of the PSF type. """
        if self.psftype==1:
            return "Gaussian"
        elif self.psftype==2:
            return "Moffat"
        elif self.psftype==3:
            return "Penny"
        elif self.psftype==4:
            return "Gausspow"
        elif self.psftype==5:
            return "Sersic"
        elif self.psftype==6:
            return "Empirical"
        
    @params.setter
    def params(self,value):
        """ Set the PSF model parameters."""
        self._params = value

    @property
    def haslookup(self):
        """ Check if there is a lookup table."""
        return (np.isfinite(self.lookup[0,0,0])==True)
        
    def starbbox(self,coords,imshape,radius=np.nan):
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
        if np.isfinite(radius)==False:
            radius = self.npix//2
        return starbbox(coords,imshape,radius)

    def bbox2xy(self,bbox):
        """                                                                                     
        Convenience method to convert boundary box of X/Y limits to 2-D X and Y arrays.
        The upper limits are EXCLUSIVE following the python convention.
        """
        return bbox2xy(bbox)

    def __str__(self):
        """ String representation of the PSF."""
        return 'PSF('+self.name+',npix='+str(self.npix)+',lookup='+str(self.haslookup)+') FWHM='+str(self.fwhm())

    @property
    def unitfootflux(self):
        """ Return the unit flux inside the footprint."""
        if np.isfinite(self._unitfootflux)==False:
            xx,yy = utils.meshgrid(np.arange(self.npix),np.arange(self.npix))
            pars = np.zeros(3,float)
            pars[0] = 1.0
            foot = self.model(xx,yy,pars,deriv=False)
            self._unitfootflux = np.sum(foot) # sum up footprint flux                         
        return self._unitfootflux
    
    def fwhm(self,pars=np.array([1.0])):
        """ Return the FWHM of the model."""
        tpars = np.zeros(3+self.nparams,float)
        tpars[0] = pars[0]
        tpars[3:] = self.params
        if self.psftype <= 5:
            fwhm = model2d_fwhm(self.psftype,tpars)
        else:
            xx,yy = utils.meshgrid(np.arange(self.npix),np.arange(self.npix))
            tpars = np.zeros(3,float)
            tpars[0] = pars[0]
            foot = self.model(xx,yy,tpars)
            fwhm = gaussfwhm(foot)
        return fwhm

    def flux(self,pars=np.array([1.0]),footprint=False):
        """ Return the flux/volume of the model given the amp or parameters."""
        if len(pars)<3:
            amp = pars[0]
            pars = np.zeros(3+self.nparams,float)
            pars[0] = amp
            pars[3:] = self.params
        if footprint:
            return self.unitfootflux*pars[0]
        else:
            if self.psftype <= 5:
                flux = model2d_flux(self.psftype,pars)
            else:
                xx,yy = utils.meshgrid(np.arange(self.npix),np.arange(self.npix))
                tpars = np.zeros(3,float)
                tpars[0] = pars[0]
                foot = self.model(xx,yy,tpars)
                flux = np.sum(foot)
            return flux   

    def evaluate(self,x, y, pars):
        """Two dimensional model function and derivatives"""
        # Get the analytic portion
        nderiv = 3
        if len(pars) != 3:
            raise Exception('pars must have 3 elements [amp,xc,yc]')
        #amp,xc,yc = pars
        if self.psftype <= 5:
            # Add amp, xc, yc to the parameters
            allpars = np.zeros(3+self.nparams,float)
            allpars[:3] = pars
            allpars[3:] = self.params
            g,derivative = amodel2d(x,y,self.psftype,allpars,nderiv)
        elif self.psftype == 6:
            g,derivative = empirical(x,y,pars,self.lookup,self.imshape,deriv=True)

        # Add lookup table portion
        if self.psftype <= 5 and self.haslookup:
            eg,ederivative = empirical(x,y,pars,self.lookup,self.imshape,deriv=True)
            g[:] += eg
            # Make sure the model is positive everywhere
            derivative[:,:] += ederivative

        return g,derivative

    def model(self,x, y, pars, deriv=False):
        """Two dimensional PSF model."""
        g, _ = self.evaluate(x, y, pars)
        return g
        
    def deriv(self,x, y, pars):
        """Two dimensional PSF derivative with respect to parameters"""
        _, derivative = self.evaluate(x, y, pars)
        return derivative

    def packpsf(self):
        """ Return the packed PSF array."""
        if self.psftype <= 5:
            if self.haslookup:
                return packpsf(self.psftype,self.params,self.lookup,self.imshape)
            else:
                return packpsf(self.psftype,self.params)
        else:
            return packpsf(self.psftype,0,self.lookup,self.imshape)

if __name__ == "__main__":
    cc.compile()

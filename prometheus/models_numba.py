import os
import sys
import numpy as np
#from scipy.special import gamma, gammaincinv, gammainc
import scipy.special as sc
#import numba_special  # The import generates Numba overloads for special
from dlnpyutils import utils as dln
import numba
from numba import njit,types
from . import leastsquares as lsq

PI = 3.141592653589793

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



###################################################################
# Numba analytical PSF models

@njit
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

####### GAUSSIAN ########

@njit
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



#@njit
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

    x2d,y2d = np.meshgrid(np.arange(nx),np.arange(ny))
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
        dbeta = qr_jac_solve(deriv,resid,weight=wt1d)
        
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
    cov = jac_covariance(deriv,resid,wt1d)
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

#@njit
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

    x2d,y2d = np.meshgrid(np.arange(nx),np.arange(ny))
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
        dbeta = qr_jac_solve(deriv,resid,weight=wt1d)
        
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
    cov = jac_covariance(deriv,resid,wt1d)
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
    relamp = clip(pars[6],0.0,1.0)  # 0<relamp<1
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
    relamp = clip(relamp,0.0,1.0)  # 0<relamp<1
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

#@njit
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

    x2d,y2d = np.meshgrid(np.arange(nx),np.arange(ny))
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
        dbeta = qr_jac_solve(deriv,resid,weight=wt1d)
        
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
    cov = jac_covariance(deriv,resid,wt1d)
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


####### GAUSSPOW ########

@njit
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

#@njit
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

    x2d,y2d = np.meshgrid(np.arange(nx),np.arange(ny))
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
        dbeta = qr_jac_solve(deriv,resid,weight=wt1d)
        
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
    cov = jac_covariance(deriv,resid,wt1d)
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


####### SERSIC ########

@njit
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
        raise Exception('aseric2d pars must have either 6 or 9 elements')
    
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
def sersic_b(n):
    # Normalisation constant
    # bn ~ 2n-1/3 for n>8
    # https://gist.github.com/bamford/b657e3a14c9c567afc4598b1fd10a459
    # n is always positive
    #return gammaincinv(2*n, 0.5)
    return gammaincinv05(2*n)

#@njit
def create_sersic_function(Ie, re, n):
    # Not required for integrals - provided for reference
    # This returns a "closure" function, which is fast to call repeatedly with different radii
    neg_bn = -b(n)
    reciprocal_n = 1.0/n
    f = neg_bn/re**reciprocal_n
    def sersic_wrapper(r):
        return Ie * np.exp(f * r ** reciprocal_n - neg_bn)
    return sersic_wrapper

@njit
def sersic_lum(Ie, re, n):
    # total luminosity (integrated to infinity)
    bn = sersic_b(n)
    g2n = gamma(2*n)
    return Ie * re**2 * 2*np.pi*n * np.exp(bn)/(bn**(2*n)) * g2n

@njit
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

#@njit
def sersic2d_estimates(pars):
    # calculate estimates for the Sersic parameters using
    # peak, x0, y0, flux, asemi, bsemi, theta
    # Sersic Parameters are [amp,x0,y0,k,alpha,recc,theta]
    peak = pars[0]
    x0 = pars[1]
    y0 = pars[2]
    flux = pars[3]
    asemi = pars[4]
    bsemi = pars[5]
    theta = pars[6]
    recc = bsemi/asemi
    
    # Calculate FWHM
    # The mean radius of an ellipse is: (2a+b)/3
    mnsig = (2.0*asemi+bsemi)/3.0
    # Convert sigma to FWHM
    # FWHM = 2*sqrt(2*ln(2))*sig ~ 2.35482*sig
    fwhm = mnsig*2.35482
    rhalf = 0.5*fwhm
    
    # Solve half-max radius equation for kserc
    # I(R) = I0 * exp(-k*R**alpha) 
    # 0.5*I0 = I0 * exp(-k*R**alpha)
    # 0.5 = exp(-k*R**alpha)
    # ln(0.5) = -k*R**alpha
    # R = (-ln(0.5)/k)**(1/alpha)
    # rhalf = (-np.log(0.5)/kserc)**(1/alpha)
    # kserc = -np.log(0.5)/rhalf**alpha
   
    # Solve flux equation for kserc
    # bn = sersic_b(n)
    # g2n = gamma(2*n)
    # flux =  recc * Ie * Re**2 * 2*np.pi*n * np.exp(bn)/(bn**(2*n)) * g2n
    # Re = np.sqrt(flux/(recc * Ie * 2*np.pi*n * np.exp(bn)/(bn**(2*n)) * g2n))
    # kserc = bn/Re**alpha    
    # kserc = bn * ((recc * Ie * 2*np.pi*n * np.exp(bn)/(bn**(2*n)) * g2n)/flux)**(alpha/2)

    # Setting the two equal and then putting everything to one side
    # 0 = np.log(0.5)/rhalf**alpha + bn * ((recc * Ie * 2*np.pi*n * np.exp(bn)/(bn**(2*n)) * g2n)/flux)**(alpha/2)
    def alphafunc(alpha):
        # rhalf, recc, flux are defined above
        n = 1/alpha
        bn = sersic_b(n)
        g2n = gamma(2*n)
        Ie,_ = sersic_full2half(peak,1.0,alpha)
        return np.log(0.5)/rhalf**alpha + bn * ((recc * Ie * 2*np.pi*n * np.exp(bn)/(bn**(2*n)) * g2n)/flux)**(alpha/2)
    
    
    # Solve for the roots
    res = root_scalar(alphafunc,x0=1.0,x1=0.5)
    if res.converged:
        alpha = res.root
    else:
        alphas = np.arange(0.1,2.0,0.05)
        vals = np.zeros(len(alphas),float)
        for i in range(len(alphas)):
            vals[i] = alphafunc(alphas[i])
        bestind = np.argmin(np.abs(vals))
        alpha = alphas[bestind]
                            
    # Now solve for ksersic
    # rhalf = (-np.log(0.5)/kserc)**(1/alpha)
    kserc = -np.log(0.5)/rhalf**alpha
    
    # Put all the parameters together
    spars = [peak,x0,y0,kserc,alpha,recc,theta]
    
    return spars


#---------------------------------------------------

# Generic model routines

@njit
def psf2d(x,y,psftype,pars,nderiv):
    """
    Two dimensional Gaussian model function.
    
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

    g,derivative = psf2d(x,y,1,pars,nderiv)

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
def apsf2d(x,y,psftype,pars,nderiv):
    """
    Two dimensional Gaussian model function with x/y array inputs.
    
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

    g,derivative = apsf2d(x,y,1,pars,3)

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
def psf2d_flux(psftype,pars):
    """
    Return the flux of a 2D PSF model.

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

    flux = psf2d_flux(pars)

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
def psf2d_fwhm(psftype,pars):
    """
    Return the fwhm of a 2D PSF model.

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

    fwhm = psf2d_fwhm(pars)

    """
    # Gaussian
    if psftype==1:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta]
        g,deriv = gaussian2d_fwhm(pars)
    # Moffat
    elif psftype==2:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta]
        g,deriv = moffat2d_fwhm(pars)
    # Penny
    elif psftype==3:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta, relamp, sigma]
        g,deriv = penny2d_fwhm(pars)
    # Gausspow
    elif psftype==4:
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta4, beta6]
        g,deriv = gausspow2d_fwhm(pars)
    # Sersic
    elif psftype==5:
        # pars = [amplitude, x0, y0, kserc, alpha, recc, theta]
        g,deriv = sersic2d_fwhm(pars)
    else:
        print('psftype=',psftype,'not supported')

    return g,deriv


@njit
def psf2d_estimates(psftype,ampc,xc,yc):
    """
    Get initial estimates for parameters

    Parameters
    ----------

    Returns
    -------

    Examples
    --------

    
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
def psf2d_bounds(psftype):
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

    bounds = psf2d_bounds(2)
    
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
def psf2d_maxsteps(psftype,pars):
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

    maxsteps = psf2d_maxsteps(2,pars)
    
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
def psf2dfit(im,err,x,y,psftype,ampc,xc,yc,verbose=False):
    """
    Fit a single model to data.

    Parameters
    ----------
    im : numpy array
       Flux array.  Can be 1D or 2D array.
    err : numpy array
       Uncertainty array of im.  Same dimensions as im.
    x : numpy array
       Array of X-values for im.
    y : numpy array
       Array of Y-values for im.
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

    pars,perror,cov,flux,fluxerr,chisq = psf2dfit(im,err,x,y,1,100.0,5.5,6.5,False)

    """

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
    bestpar = psf2d_estimates(psftype,ampc,xc,yc)
    nparams = len(bestpar)
    nderiv = nparams
    bounds = psf2d_bounds(psftype)

    if verbose:
        print('bestpar=',bestpar)
        print('nderiv=',nderiv)
    
    # Iteration loop
    maxpercdiff = 1e10
    niter = 0
    while (niter<maxiter and maxpercdiff>minpercdiff):
        model,deriv = apsf2d(x1d,y1d,psftype,bestpar,nderiv)
        resid = im1d-model
        dbeta = qr_jac_solve(deriv,resid,weight=wt1d)
        
        if verbose:
            print(niter,bestpar)
            print(dbeta)
        
        # Update parameters
        last_bestpar = bestpar.copy()
        # limit the steps to the maximum step sizes and boundaries
        maxsteps = psf2d_maxsteps(psftype,bestpar)
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

    model,deriv = apsf2d(x1d,y1d,psftype,bestpar,nderiv)
    resid = im1d-model
    
    # Get covariance and errors
    cov = jac_covariance(deriv,resid,wt1d)
    perror = np.sqrt(np.diag(cov))

    # Now get the flux
    flux = psf2d_flux(psftype,bestpar)
    fluxerr = perror[0]*(flux/bestpar[0]) 
    
    return bestpar,perror,cov,flux,fluxerr,chisq

    
@njit
def psf2dfit2(im,err,psftype,ampc,xc,yc,verbose):
    """
    Fit a single model to data.

    Parameters
    ----------
    im : numpy array
       Flux array.  Can be 1D or 2D array.
    err : numpy array
       Uncertainty array of im.  Same dimensions as im.
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
    
    Example
    -------

    pars,perror,cov,flux,fluxerr = psf2dfit(im,err,1,100.0,5.5,6.5,False)

    """
    # Gaussian
    if psftype==1:
        return gaussian2dfit(im,err,ampc,xc,yc,verbose)
    # Moffat
    elif psftype==2:
        return moffat2d_fit(im,err,ampc,xc,yc,verbose)
    # Penny
    elif psftype==3:
        return penny2d_fit(im,err,ampc,xc,yc,verbose)
    # Gausspow
    elif psftype==4:
        return gausspow2d_fit(im,err,ampc,xc,yc,verbose)
    # Sersic
    elif psftype==5:
        return sersic2d_fit(im,err,ampc,xc,yc,verbose)
    else:
        print('psftype=',psftype,'not supported')
        return
    
####################################################




# Left to add:
# -empirical


####---------------------------------------


@njit
def gaussderiv(nx,ny,asemi,bsemi,theta,amp,xc,yc,nderiv):
    """ Generate Gaussians PSF model and derivative."""
    cxx,cyy,cxy = gauss_abt2cxy(asemi,bsemi,theta)
    deriv = np.zeros((ny,nx,nderiv),float)
    for i in range(nx):
        for j in range(ny):
            deriv1 = gaussvalderiv(i,j,amp,xc,yc,asemi,bsemi,theta,cxx,cyy,cxy,nderiv)
            #deriv1 = gaussvalderiv(i,j,amp,xc,yc,cxx,cyy,cxy,nderiv)
            deriv[j,i,:] = deriv1
    model = amp * deriv[:,:,0]
    return model,deriv



#def gaussfit(im,xc,yc,asemi,bsemi,theta,bbx0,bbx1,bby0,bby1,thresh=1e-3):
def gaussfit(im,tab,thresh=1e-3):
    """ Fit elliptical Gaussian profile to a source. """

    gflux,apflux,npix = numba_gaussfit(im,tab['mnx'],tab['mny'],tab['asemi'],
                                       tab['bsemi'],tab['theta'],tab['bbx0'].astype(float),
                                       tab['bbx1'].astype(float),tab['bby0'].astype(float),
                                       tab['bby1'].astype(float),thresh)

    return gflux,apflux,npix
    
@njit
def numba_gaussfit(im,xc,yc,asemi,bsemi,theta,bbx0,bbx1,bby0,bby1,thresh):
    """ Fit elliptical Gaussian profile to a source. """

    nsource = len(xc)
    gflux = np.zeros(nsource,float)
    apflux = np.zeros(nsource,float)
    npix = np.zeros(nsource,dtype=np.int64)
    for i in range(nsource):
        if asemi[i]>0 and bsemi[i]>0:
            gflux1,apflux1,npix1 = numba_gfit(im,xc[i],yc[i],asemi[i],bsemi[i],theta[i],
                                              bbx0[i],bbx1[i],bby0[i],bby1[i],thresh)
            gflux[i] = gflux1
            apflux[i] = apflux1
            npix[i] = npix1
        
    return gflux,apflux,npix
    
@njit
def numba_gfit(im,xc,yc,asemi,bsemi,theta,bbx0,bbx1,bby0,bby1,thresh):
    """ Fit elliptical Gaussian profile to a source. """

    ny,nx = im.shape
    nyb = int(bby1-bby0+1)
    nxb = int(bbx1-bbx0+1)

    if bsemi==0:
        bsemi = 0.1
    if asemi==0:
        asemi = 0.1
        
    # Calculate sigx, sigy, cxx, cyy, cxy
    cxx,cyy,cyy = gauss_abt2cxy(asemi,bsemi,theta)
    #thetarad = np.deg2rad(theta)
    #sintheta = np.sin(thetarad)
    #costheta = np.cos(thetarad)
    #sintheta2 = sintheta**2
    #costheta2 = costheta**2
    #asemi2 = asemi**2
    #bsemi2 = bsemi**2
    #cxx = costheta2/asemi2 + sintheta2/bsemi2
    #cyy = sintheta2/asemi2 + costheta2/bsemi2
    #cxy = 2*costheta*sintheta*(1/asemi2-1/bsemi2)
    
    # Simple linear regression
    # https://en.wikipedia.org/wiki/Simple_linear_regression
    # y = alpha + beta*x
    # beta = Sum( (xi-xmn)*(yi-ymn) ) / Sum( (xi-xmn)**2 )
    # alpha = ymn - beta*xmn
    #
    # without the intercept term
    # y = beta*x
    # beta = Sum(xi*yi ) / Sum(xi**2)

    #model = np.zeros((nyb,nxb),float)

    # thresh has to change with the size of the source
    # if the source is larger then the value of the normalized
    # Gaussian will be smaller
    # We are now using a unit-amplitude Gaussian instead
    
    apflux = 0.0    # elliptical aperture flux
    sumxy = 0.0
    sumx2 = 0.0
    npix = 0
    for i in range(nxb):
        x = i+int(bbx0)
        for j in range(nyb):
            y = j+int(bby0)
            imval = im[y,x]
            # Unit amplitude Gaussian value
            gval = gaussval(x,y,xc,yc,cxx,cyy,cxy)
            #model[j,i] = gval
            if gval>thresh:
                npix += 1
                # Add to aperture flux
                if imval>0:
                    apflux += imval
                # Calculate fit
                # x is the model, y is the data
                # beta is the flux amplitude
                sumxy += gval*imval
                sumx2 += gval**2

    if sumx2 <= 0:
        sumx2 = 1
    beta = sumxy / sumx2

    # Now multiply by the volume of the Gaussian
    amp = asemi*bsemi*2*np.pi
    gflux = amp * beta
    
    return gflux,apflux,npix

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
def checkbounds(pars,bounds):
    """ Check the parameters against the bounds."""
    # 0 means it's fine
    # 1 means it's beyond the lower bound
    # 2 means it's beyond the upper bound
    npars = len(pars)
    lbounds = bounds[:,0]
    ubounds = bounds[:,1]
    check = np.zeros(npars,np.int32)
    check[pars<=lbounds] = 1
    check[pars>=ubounds] = 2
    return check

@njit
def limbounds(pars,bounds):
    """ Limit the parameters to the boundaries."""
    lbounds = bounds[:,0]
    ubounds = bounds[:,1]
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
def newlsqpars(pars,steps,bounds,maxsteps):
    """ Return new parameters that fit the constraints."""
    # Limit the steps to maxsteps
    limited_steps = limsteps(steps,maxsteps)
        
    # Make sure that these don't cross the boundaries
    lbounds = bounds[:,0]
    ubounds = bounds[:,1]
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
    newparams = pars + newsteps
            
    # Make sure to limit them to the boundaries
    check = checkbounds(newparams,bounds)
    badpars = (check!=0)
    if np.sum(badpars)>0:
        # add a tiny offset so it doesn't fit right on the boundary
        newparams = np.minimum(np.maximum(newparams,lbounds+1e-30),ubounds-1e-30)
    return newparams


# Fit analytic Gaussian profile first
# x/y/amp


@njit
def newbestpars(bestpars,dbeta):
    """ Get new pars from offsets."""
    newpars = np.zeros(3,float)
    maxchange = 0.5
    # Amplitude
    ampmin = bestpars[0]-maxchange*np.abs(bestpars[0])
    ampmin = np.maximum(ampmin,0)
    ampmax = bestpars[0]+np.abs(maxchange*bestpars[0])
    newamp = clip(bestpars[0]+dbeta[0],ampmin,ampmax)
    newpars[0] = newamp
    # Xc, maxchange in pixels
    xmin = bestpars[1]-maxchange
    xmax = bestpars[1]+maxchange
    newx = clip(bestpars[1]+dbeta[1],xmin,xmax)
    newpars[1] = newx
    # Yc
    ymin = bestpars[2]-maxchange
    ymax = bestpars[2]+maxchange
    newy = clip(bestpars[2]+dbeta[2],ymin,ymax)
    newpars[2] = newy
    return newpars

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


@njit
def gausspsffit(im,err,gpars,ampc,xc,yc,verbose):
    """ Fit a PSF to a source."""
    # xc/yc are with respect to the image origin (0,0)
    
    # Solve for x, y, amplitude

    maxiter = 10
    minpercdiff = 0.5
    
    ny,nx = im.shape
    #nyp,nxp = psf.shape

    wt = 1/err**2
    #wt /= np.sum(wt)
    
    # Gaussian parameters
    asemi,bsemi,theta = gpars
    cxx,cyy,cxy = gauss_abt2cxy(asemi,bsemi,theta)
    #volume = asemi*bsemi*2*np.pi

    # Initial values
    bestpar = np.zeros(3,float)
    bestpar[0] = ampc
    bestpar[1] = xc
    bestpar[2] = yc
    
    # Iteration loop
    maxpercdiff = 1e10
    niter = 0
    while (niter<maxiter and maxpercdiff>minpercdiff):
        model,deriv = gaussderiv(nx,ny,asemi,bsemi,theta,
                                 bestpar[0],bestpar[1],bestpar[2],3)
        resid = im-model
        dbeta = qr_jac_solve(deriv.reshape(ny*nx,3),resid.ravel(),weight=wt.ravel())

        if verbose:
            print(niter,bestpar)
            print(dbeta)
        
        # Update parameters
        last_bestpar = bestpar.copy()
        # limit the steps to the maximum step sizes and boundaries
        #if bounds is not None or maxsteps is not None:
        #    bestpar = newpars(bestpar,dbeta,bounds,maxsteps)
        bounds = np.zeros((3,2),float)
        bounds[:,0] = [0.00,0,0]
        bounds[:,1] = [1e30,nx,ny]
        maxsteps = np.zeros(3,float)
        maxsteps[:] = [0.5*bestpar[0],0.5,0.5]
        bestpar = newlsqpars(bestpar,dbeta,bounds,maxsteps)
        #bestpar = newbestpars(bestpar,dbeta)
        #else:
        #bestpar += 0.5*dbeta
        
        # Check differences and changes
        diff = np.abs(bestpar-last_bestpar)
        denom = np.maximum(np.abs(bestpar.copy()),0.0001)
        percdiff = diff.copy()/denom*100  # percent differences
        maxpercdiff = np.max(percdiff)
        chisq = np.sum((im-model)**2/err**2)/(nx*ny)
        if verbose:
            print(niter,percdiff,chisq)
            print()
        last_dbeta = dbeta
        niter += 1

    model,deriv = gaussderiv(nx,ny,asemi,bsemi,theta,
                             bestpar[0],bestpar[1],bestpar[2],3)        
    resid = im-model
    
    # Get covariance and errors
    cov = jac_covariance(deriv.reshape(ny*nx,3),resid.ravel(),wt.ravel())
    perror = np.sqrt(np.diag(cov))

    # Now get the flux, multiply by the volume of the Gaussian
    gvolume = asemi*bsemi*2*np.pi
    flux = bestpar[0]*gvolume
    fluxerr = perror[0]*gvolume
    
    return bestpar,perror,cov,flux,fluxerr


@njit
def psffit(im,err,ampc,xc,yc,verbose):
    """ Fit all parameters of the Gaussian."""
    # xc/yc are with respect to the image origin (0,0)
    
    # Solve for x, y, amplitude and asemi/bsemi/theta

    maxiter = 10
    minpercdiff = 0.5
    
    ny,nx = im.shape

    wt = 1/err**2

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
        #model,deriv = gaussderiv(nx,ny,bestpar[3],bestpar[4],bestpar[5],
        #                         bestpar[0],bestpar[1],bestpar[2],6)
        model,deriv = gaussderiv(xx,yy,bestpars,6)
        resid = im-model
        dbeta = qr_jac_solve(deriv.reshape(ny*nx,6),resid.ravel(),weight=wt.ravel())
        
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
        chisq = np.sum((im-model)**2/err**2)/(nx*ny)
        if verbose:
            print('chisq=',chisq)
        #if verbose:
        #    print(niter,percdiff,chisq)
        #    print()
        last_dbeta = dbeta
        niter += 1

    model,deriv = gaussderiv(nx,ny,bestpar[3],bestpar[4],bestpar[5],
                             bestpar[0],bestpar[1],bestpar[2],6)
    resid = im-model
    
    # Get covariance and errors
    cov = jac_covariance(deriv.reshape(ny*nx,6),resid.ravel(),wt.ravel())
    perror = np.sqrt(np.diag(cov))

    # Now get the flux, multiply by the volume of the Gaussian
    asemi,bsemi,theta = bestpar[3],bestpar[4],bestpar[5]
    gvolume = asemi*bsemi*2*np.pi
    flux = bestpar[0]*gvolume
    fluxerr = perror[0]*gvolume
    
    return bestpar,perror,cov,flux,fluxerr


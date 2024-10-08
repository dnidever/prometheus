import os
import sys
import numpy as np
from dlnpyutils import utils as dln
from numba import njit
from . import leastsquares as lsq

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
    
def sky(im,tot=False,med=False):
    tot = 0
    if tot:
        tot = 1
    if med:
        med = 1
        tot = 0
    return numba_sky(im,tot=tot,med=med)

@njit
def numba_sky(im,tot=1,med=0):
    """ Estimate the background."""
    binsize = 200
    ny,nx = im.shape
    ny2 = ny // binsize
    nx2 = nx // binsize
    bgim = np.zeros((ny2,nx2),float)
    sample = np.random.randint(0,binsize*binsize-1,1000)
    #xsample = np.random.randint(0,nbin-1,500)
    #ysample = np.random.randint(0,nbin-1,500)
    for i in range(ny2):
        for j in range(nx2):
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
    #import pdb; pdb.set_trace()

    return bgim

def sky2(im,binsize=200,tot=False,med=True):
    tot = 0
    if tot:
        tot = 1
    if med:
        med = 1
        tot = 0
    bgimbin = numba_sky2(im,binsize,tot,med)

    # Linearly interpolate
    bgim = np.zeros(im.shape,float)+np.median(bgimbin)
    bgim = numba_linearinterp(bgimbin,bgim,binsize)

    # do the edges
    #bgim[:binsize,:] = bgim[binsize,:].reshape(1,-1)
    #bgim[:,:binsize] = bgim[:,binsize].reshape(-1,1)
    #bgim[-binsize:,:] = bgim[-binsize,:].reshape(1,-1)
    #bgim[:,-binsize:] = bgim[:,-binsize].reshape(-1,1)    
    
    return bgim
    
@njit
def numba_sky2(im,binsize,tot,med):
    """ Estimate the background."""
    ny,nx = im.shape
    ny2 = ny // binsize
    nx2 = nx // binsize
    bgim = np.zeros((ny2,nx2),float)
    #sample = np.random.randint(0,binsize*binsize-1,1000)
    xsample = np.random.randint(0,binsize-1,1000)
    ysample = np.random.randint(0,binsize-1,1000)
    data = np.zeros(1000,float)
    for i in range(ny2):
        for j in range(nx2):
            x1 = i*binsize
            x2 = x1+binsize
            if x2 > nx: x2=nx
            y1 = j*binsize
            y2 = y1+binsize
            if y2 > ny: y2=ny
            for k in range(1000):
                data[k] = im[y1+ysample[k],x1+xsample[k]]
            if tot==1:
                bgim[j,i] = np.sum(data)
            elif med==1:
                bgim[j,i] = np.median(data)
            else:
                bgim[j,i] = np.mean(data)
    #import pdb; pdb.set_trace()

    return bgim

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

def sigma(im):
    """ Determine sigma."""
    sample = np.random.randint(0,im.size-1,10000)
    sig = dln.mad(im.ravel()[sample])
    return sig
    
def detection(im,nsig=10):
    """  Detect peaks """

    # just looping over the 9K x 9K array
    # takes 1.3 sec

    # bin 2x2 as a crude initial smoothing
    imbin = dln.rebin(im,binsize=(2,2))
    
    sig = sigma(imbin)
    xpeak,ypeak,count = numba_detection(imbin,sig,nsig)
    xpeak = xpeak[:count]*2
    ypeak = ypeak[:count]*2
    
    return xpeak,ypeak

@njit
def numba_detection(im,sig,nsig):
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
def numba_detection2(im,sig,nsig):
    """ Detect peaks"""
    # input sky subtracted image
    ny,nx = im.shape
    lines = np.zeros((ny,3),float)    
    nbin = 5
    nhbin = nbin//2
    
    # Start the mean lines
    for j in range(ny):
        lines[j,0] = np.mean(im[j,:nbin])
        lines[j,1] = np.mean(im[j,1:nbin+1])
        lines[j,2] = np.mean(im[j,2:nbin+2])
        
    count = 0
    xpeak = np.zeros(100000,float)
    ypeak = np.zeros(100000,float)
    grid = np.zeros((3,3),float)
    for i in np.arange(nhbin+1,nx-nhbin-2):
        # Find peaks
        #lval = np.mean(line[:nbin])
        #mval = np.mean(line[1:nbin+1])
        #rval = np.mean(line[2:nbin+2])
        for j in range(nhbin+1,ny-nhbin-2):
            # 3x3 grid
            grid = lines[j-1:j+2,:]
            mval = grid[1,1]
            if mval>nsig*sig:
                if (mval>grid[0,0] and mval>grid[0,1] and mval>grid[0,2] and
                    mval>grid[1,0] and mval>grid[1,2] and mval>grid[2,0] and
                    mval>grid[2,1] and mval>grid[2,2]):
                    xpeak[count] = i
                    ypeak[count] = j
                    count = count + 1
            ## increment
            #lval += line[j+nhbin]/nbin - line[j-nhbin-1]/nbin
            #mval += line[j+nhbin+1]/nbin - line[j-nhbin]/nbin
            #rval += line[j+nhbin+2]/nbin - line[j-nhbin+1]/nbin
                
        # Add/Subtract line
        for j in range(ny):
            lines[j,0] = lines[j,1]  # shift
            lines[j,1] = lines[j,2]  # shift
            lines[j,2] += im[j,i+nhbin+2]/nbin - im[j,i-nhbin+1]/nbin

    return xpeak,ypeak,count

@njit
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
    #print(leftxp,rightxp,downyp,upyp)

    return leftxp,rightxp,downyp,upyp

@njit
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

def morphology(im,xpeak,ypeak,thresh=0.2,bmax=100):
    """ Measure morphology for the peaks."""
    # thresh - fraction amplitude threshold
    # bmax - maximum half-width of the bounding box
    
    ny,nx = im.shape
    nbin = 3
    nhbin = nbin//2   

    out = numba_morphology(im,xpeak,ypeak,thresh,bmax)

    dt = [('xp',int),('yp',int),('bbx0',int),('bbx1',int),
          ('bby0',int),('bby1',int),('area',int),
          ('flux',float),('mnx',float),
          ('mny',float),('sigx',float),('sigy',float),
          ('sigxy',float),('fwhm',float),('asemi',float),
          ('bsemi',float),('theta',float)]
    tab = np.rec.fromarrays(out.T, dtype=dt)
    
    return tab
    
@njit
def numba_morphology(im,xpeak,ypeak,thresh,bmax):
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


# models


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




    
@njit
def gaussval(x,y,xc,yc,cxx,cyy,cxy):
    """ Evaluate an elliptical unit-amplitude Gaussian at a point."""
    u = (x-xc)
    v = (y-yc)
    # amp = 1/(asemi*bsemi*2*np.pi)
    val = np.exp(-0.5*(cxx*u**2 + cyy*v**2 + cxy*u*v))
    return val

@njit
def gaussvalderiv(x,y,amp,xc,yc,asemi,bsemi,theta,cxx,cyy,cxy,nderiv):
    """ Evaluate an elliptical unit-amplitude Gaussian at a point and deriv."""
    u = (x-xc)
    u2 = u**2
    v = (y-yc)
    v2 = v**2
    # amp = 1/(asemi*bsemi*2*np.pi)
    g = amp * np.exp(-0.5*(cxx*u**2 + cyy*v**2 + cxy*u*v))

    #  pars = [amplitude, x0, y0, xsigma, ysigma, theta]
    deriv = np.zeros(nderiv,float)
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
        
    return deriv

@njit
def gaussvalderiv_cxy(x,y,amp,xc,yc,cxx,cyy,cxy,nderiv):
    """ Evaluate an elliptical unit-amplitude Gaussian at a point and deriv."""
    u = (x-xc)
    u2 = u**2
    v = (y-yc)
    v2 = v**2
    # amp = 1/(asemi*bsemi*2*np.pi)
    g = amp * np.exp(-0.5*(cxx*u**2 + cyy*v**2 + cxy*u*v))

    #  pars = [amplitude, x0, y0, xsigma, ysigma, theta]
    deriv = np.zeros(nderiv,float)
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
        # cxx
        dg_cxx = -g * 0.5*u**2
        deriv[3] = dg_cxx
        # cyy
        dg_cyy = -g * 0.5*v**2
        deriv[4] = dg_cyy
        # cxy
        dg_cxy = -g * 0.5*u*v
        deriv[5] = dg_cxy
        
    return deriv

###################################################################
# Numba analytical PSF models

####### GAUSSIAN ########

#@njit
def gaussian2d_flux(pars):
    """
    Return the total flux (or volume) of a 2D Gaussian.

    Parameters
    ----------
    pars : numpy array or list
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

#@njit
def gaussian2d_fwhm(pars):
    """
    Return the FWHM of a 2D Gaussian.

    Parameters
    ----------
    pars : numpy array or list
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
    sig_major = np.max([xsig,ysig])
    sig_minor = np.min([xsig,ysig])
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
    pars : numpy array or list
       Parameter list. pars = [amplitude, x0, y0, xsigma, ysigma, theta].
         Or can include cxx, cyy, cxy at the end so they don't have to be
         computed.
    deriv : boolean, optional
       Return the derivatives as well.
    nderiv : int, optional
       The number of derivatives to return.  The default is None
        which means that all are returned if deriv=True.

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
    x : numpy array
      Single X-value for which to compute the Gaussian model.
    y : numpy array
      Single Y-value of points for which to compute the Gaussian model.
    pars : numpy array or list
       Parameter list. pars = [amplitude, x0, y0, xsigma, ysigma, theta]
    deriv : boolean, optional
       Return the derivatives as well.
    nderiv : int, optional
       The number of derivatives to return.  The default is None
        which means that all are returned if deriv=True.

    Returns
    -------
    g : numpy array
      The Gaussian model for the input x/y values and parameters (same
        shape as x/y).
    derivative : list
      List of derivatives of g relative to the input parameters.
        This is only returned if deriv=True.

    Example
    -------

    g = gaussian2d(x,y,pars)

    or

    g,derivative = gaussian2d(x,y,pars,deriv=True)

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
    """ Fit all parameters of the Gaussian."""
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

#@njit
def moffat2d_fwhm(pars):
    """
    Return the FWHM of a 2D Moffat function.

    Parameters
    ----------
    pars : numpy array or list
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
    sig_major = np.max([xsig,ysig])
    sig_minor = np.min([xsig,ysig])
    mnsig = (2.0*sig_major+sig_minor)/3.0
    
    return 2.0 * np.abs(mnsig) * np.sqrt(2.0 ** (1.0/beta) - 1.0)

#@njit
def moffat2d_flux(pars):
    """
    Return the total Flux of a 2D Moffat.

    Parameters
    ----------
    pars : numpy array or list
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
    pars : numpy array or list
       Parameter list. pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta].
    nderiv : int, optional
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
    x : numpy array
      Array of X-values of points for which to compute the Moffat model.
    y : numpy array
      Array of Y-values of points for which to compute the Moffat model.
    pars : numpy array or list
       Parameter list. pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta]
         The cxx, cyy, cxy parameter can be added to the end so they don't
         have to be computed.
    nderiv : int, optional
       The number of derivatives to return.

    Returns
    -------
    g : numpy array
      The Moffat model for the input x/y values and parameters.
    derivative : list
      Array of derivatives of g relative to the input parameters.

    Example
    -------

    g,derivative = moffat2d(x,y,pars,deriv=True)

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
    """ Fit all parameters of the Moffat."""
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

#@njit
def penny2d_fwhm(pars):
    """
    Return the FWHM of a 2D Penny function.

    Parameters
    ----------
    pars : numpy array or list
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
    relamp = np.minimum(np.maximum(pars[6],0),1.0)  # 0<relamp<1
    sigma = np.maximum(pars[7],0)
    beta = 1.2   # Moffat

    if np.sum(~np.isfinite(np.array(pars)))>0:
        raise ValueError('PARS cannot be inf or nan')
    
    # The mean radius of an ellipse is: (2a+b)/3
    sig_major = np.max([xsig,ysig])
    sig_minor = np.min([xsig,ysig])
    mnsig = (2.0*sig_major+sig_minor)/3.0
    # Convert sigma to FWHM
    # FWHM = 2*sqrt(2*ln(2))*sig ~ 2.35482*sig
    gfwhm = mnsig*2.35482
    if relamp==0:
        return gfwhm
    
    # Moffat beta=1.2 FWHM
    mfwhm = 2.0 * np.abs(sigma) * np.sqrt(2.0 ** (1.0/beta) - 1.0)

    # Generate a small profile
    x = np.arange( np.min([gfwhm,mfwhm])/2.35/2, np.max([gfwhm,mfwhm]), 0.5)
    f = (1-relamp)*np.exp(-0.5*(x/mnsig)**2) + relamp/(1+(x/sigma)**2)**beta
    hwhm = np.interp(0.5,f[::-1],x[::-1])
    fwhm = 2*hwhm
        
    return fwhm

#@njit
def penny2d_flux(pars):
    """
    Return the total Flux of a 2D Penny function.

    Parameters
    ----------
    pars : numpy array or list
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
    pars : numpy array or list
       Parameter list. pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta].
    nderiv : int, optional
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
    x : numpy array
      Array of X-values of points for which to compute the Penny model.
    y : numpy array
      Array of Y-values of points for which to compute the Penny model.
    pars : numpy array or list
       Parameter list. pars = [amplitude, x0, y0, xsigma, ysigma, theta,
                               relamp, sigma]
         The cxx, cyy, cxy parameter can be added to the end so they don't
         have to be computed.
    nderiv : int, optional
       The number of derivatives to return.

    Returns
    -------
    g : numpy array
      The Penny model for the input x/y values and parameters.
    derivative : list
      Array of derivatives of g relative to the input parameters.

    Example
    -------

    g,derivative = penny2d(x,y,pars,deriv=True)

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
        dg_dA = f / amp
        deriv[0] = dg_dA
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
    """ Fit all parameters of the Penny."""
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

#@njit
def gausspow2d_fwhm(pars):
    """
    Return the FWHM of a 2D DoPHOT Gausspow function.

    Parameters
    ----------
    pars : numpy array or list
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
    sig_major = np.max([xsig,ysig])
    sig_minor = np.min([xsig,ysig])
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

def gausspow2d_flux(pars):
    """
    Return the flux of a 2D DoPHOT Gausspow function.

    Parameters
    ----------
    pars : numpy array or list
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


#@njit
def agausspow2d(x,y,pars,nderiv):
    """
    Two dimensional Gausspow model function with x/y array inputs.
    
    Parameters
    ----------
    x : numpy array
      Array of X-values of points for which to compute the Gausspow model
    y : numpy array
      Array of Y-values of points for which to compute the Gausspow model.
    pars : numpy array or list
       Parameter list. pars = [amplitude, x0, y0, xsigma, ysigma, theta, beta].
    nderiv : int, optional
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

    g,derivative = agausspow2d(x,y,pars,3)

    """

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
        g1,deriv1 = gausspow2d(xx[i],yy[i],allpars,nderiv)
        g[i] = g1
        if nderiv>0:
            deriv[i,:] = deriv1
    return g,deriv

    
#@njit
def gausspow2d(x,y,pars,nderiv):
    """
    Two dimensional Gausspow model function for a single point.
    Gaussian core and Lorentzian-like wings, only Gaussian is tilted.

    Parameters
    ----------
    x : numpy array
      Array of X-values of points for which to compute the Gausspow model.
    y : numpy array
      Array of Y-values of points for which to compute the Gausspow model.
    pars : numpy array or list
       Parameter list. pars = [amplitude, x0, y0, xsigma, ysigma, theta,
                               relamp, sigma]
         The cxx, cyy, cxy parameter can be added to the end so they don't
         have to be computed.
    nderiv : int, optional
       The number of derivatives to return.

    Returns
    -------
    g : numpy array
      The Gausspow model for the input x/y values and parameters.
    derivative : list
      Array of derivatives of g relative to the input parameters.

    Example
    -------

    g,derivative = gausspow2d(x,y,pars,deriv=True)

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
        dg_dA = f / amp
        deriv[0] = dg_dA
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
def gausspow2dfit(im,err,ampc,xc,yc,verbose):
    """ Fit all parameters of the Gausspow."""
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

#@njit  
def sersic2d(x, y, pars, deriv=False, nderiv=None):
    """
    Sersic profile and can be elliptical and rotated.

    Parameters
    ----------
    x : numpy array
      Array of X-values of points for which to compute the Sersic model.
    y : numpy array
      Array of Y-values of points for which to compute the Sersic model.
    pars : numpy array or list
       Parameter list.
        pars = [amp,x0,y0,k,alpha,recc,theta]
    deriv : boolean, optional
       Return the derivatives as well.
    nderiv : int, optional
       The number of derivatives to return.  The default is None
        which means that all are returned if deriv=True.

    Returns
    -------
    g : numpy array
      The Sersic model for the input x/y values and parameters (same
        shape as x/y).
    derivative : list
      List of derivatives of g relative to the input parameters.
        This is only returned if deriv=True.

    Example
    -------

    g = sersic2d(x,y,pars)

    or

    g,derivative = sersic2d(x,y,pars,deriv=True)

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
    
    xdiff = x - pars[1]
    ydiff = y - pars[2]
    amp = pars[0]
    kserc = pars[3]
    alpha = pars[4]
    recc = pars[5]               # b/a
    theta = pars[6]    
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
   
    # Compute derivative as well
    if deriv is True:

        # How many derivative terms to return
        if nderiv is not None:
            if nderiv <=0:
                nderiv = 7
        else:
            nderiv = 7
        
        derivative = []
        if nderiv>=1:
            dg_dA = g / amp
            derivative.append(dg_dA)
        if nderiv>=2:        
            dg_dx_mean = g * (kserc*alpha)*(rr**(alpha-2))*0.5*((2 * a * xdiff) + (b * ydiff))
            dg_dx_mean[rr==0] = 0
            derivative.append(dg_dx_mean)
        if nderiv>=3:
            dg_dy_mean = g * (kserc*alpha)*(rr**(alpha-2))*0.5*((2 * c * ydiff) + (b * xdiff))
            dg_dx_mean[rr==0] = 0           
            derivative.append(dg_dy_mean)
        if nderiv>=4:
            dg_dk = -g * rr**alpha
            derivative.append(dg_dk)
        if nderiv>=5:
            dg_dalpha = -g * kserc*np.log(rr) * rr**alpha
            dg_dalpha[rr==0] = 0
            derivative.append(dg_dalpha)
        if nderiv>=6:
            xdiff2 = xdiff ** 2
            ydiff2 = ydiff ** 2
            recc3 = recc**3
            da_drecc = -2*sint2 / recc3
            db_drecc =  2*sin2t / recc3            
            dc_drecc = -2*cost2 / recc3            
            dg_drecc = -g*(kserc*alpha)*(rr**(alpha-2))*0.5*(da_drecc * xdiff2 +
                                                             db_drecc * xdiff * ydiff +
                                                             dc_drecc * ydiff2)
            dg_drecc[rr==0] = 0
            derivative.append(dg_drecc)
        if nderiv>=7:
            sint = np.sin(theta)
            cost = np.cos(theta)
            cos2t = np.cos(2.0*theta)
            da_dtheta = (sint * cost * ((1. / ysig2) - (1. / xsig2)))
            db_dtheta = (cos2t / xsig2) - (cos2t / ysig2)            
            dc_dtheta = -da_dtheta            
            dg_dtheta = -g*(kserc*alpha)*(rr**(alpha-2))*(da_dtheta * xdiff2 +
                                                          db_dtheta * xdiff * ydiff +
                                                          dc_dtheta * ydiff2)
            dg_dtheta[rr==0] = 0
            derivative.append(dg_dtheta)

        # special case if alpha=2???
            
        return g,derivative
            
    # No derivative
    else:        
        return g

#@njit
def sersic2d_fwhm(pars):
    """
    Return the FWHM of a 2D Sersic function.

    Parameters
    ----------
    pars : numpy array or list
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

    if np.sum(~np.isfinite(np.array(pars)))>0:
        raise ValueError('PARS cannot be inf or nan')

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

#@njit
def sersic_b(n):
    # Normalisation constant
    # bn ~ 2n-1/3 for n>8
    # https://gist.github.com/bamford/b657e3a14c9c567afc4598b1fd10a459    
    return gammaincinv(2*n, 0.5)

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

#@njit
def sersic_lum(Ie, re, n):
    # total luminosity (integrated to infinity)
    bn = sersic_b(n)
    g2n = gamma(2*n)
    return Ie * re**2 * 2*np.pi*n * np.exp(bn)/(bn**(2*n)) * g2n

#@njit
def sersic_full2half(I0,kserc,alpha):
    # Convert Io and k to Ie and Re
    # Ie = Io * exp(-bn)
    # Re = (bn/k)**n
    n = 1/alpha
    bn = sersic_b(n)
    Ie = I0 * np.exp(-bn)
    Re = (bn/kserc)**n
    return Ie,Re

#@njit
def sersic_half2full(Ie,Re,alpha):
    # Convert Ie and Re to Io and k
    # Ie = Io * exp(-bn)
    # Re = (bn/k)**n
    n = 1/alpha
    bn = sersic_b(n)
    I0 = Ie * np.exp(bn)
    kserc = bn/Re**alpha
    return I0,kserc

#@njit
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

    





# Left to add:
# -empirical
# -sersic


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


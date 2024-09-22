import os
import numpy as np
from dlnpyutils import utils as dln
from numba import njit

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
    theta = np.rad2deg(0.5*np.arctan2(2*sigxy,sigx2-sigy2))

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

@njit
def gaussval(x,y,xc,yc,asemi,bsemi,cxx,cyy,cxy):
    """ Evaluate an elliptical unit-amplitude Gaussian at a point."""
    u = (x-xc)
    v = (y-yc)
    # amp = 1/(asemi*bsemi*2*np.pi)
    val = np.exp(-0.5*(cxx*u**2 + cyy*v**2 + cxy*u*v))
    return val

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
    thetarad = np.deg2rad(theta)
    sintheta = np.sin(thetarad)
    costheta = np.cos(thetarad)
    sintheta2 = sintheta**2
    costheta2 = costheta**2
    asemi2 = asemi**2
    bsemi2 = bsemi**2
    cxx = costheta2/asemi2 + sintheta2/bsemi2
    cyy = sintheta2/asemi2 + costheta2/bsemi2
    cxy = 2*costheta*sintheta*(1/asemi2-1/bsemi2)
    
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
            gval = gaussval(x,y,xc,yc,asemi,bsemi,cxx,cyy,cxy)
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

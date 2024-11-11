#!/usr/bin/env python

"""ALLFIT_NUMBA.PY - Fit PSF to all stars in an image

"""

__authors__ = 'David Nidever <dnidever@montana.edu?'
__version__ = '20241013'  # yyyymmdd

import os
import numpy as np
import time
from numba import njit,types,from_dtype
from numba.experimental import jitclass
from . import utils_numba as utils, groupfit_numba as gfit, models_numba as mnb
from .clock_numba import clock

#@njit(cache=True)
@njit
def getstar(imshape,mask,xcen,ycen,hpsfnpix,fitradius,skyradius):
    """ Return a star's full footprint and fitted pixels data."""
    # always return the same size
    # a distance of 6.2 pixels spans 6 full pixels but you could have
    # 0.1 left on one side and 0.1 left on the other side
    # that's why we have to add 2 pixels
    maxpix = int(3.14*skyradius**2)
    nskypix = int(2*skyradius+1)
    skybbox = utils.starbbox((xcen,ycen),imshape,int(skyradius))
    nsx = skybbox[1]-skybbox[0]
    nsy = skybbox[3]-skybbox[2]
    skyravelindex = np.zeros(maxpix,np.int64)-1
    nfpix = hpsfnpix*2+1
    #fbbox = utils.starbbox((xcen,ycen),imshape,hpsfnpix)
    # extra buffer is ALWAYS at the end of each dimension    
    fravelindex = np.zeros(nfpix*nfpix,np.int64)-1
    # Fitting pixels
    npix = int(np.floor(2*fitradius))+2
    bbox = utils.starbbox((xcen,ycen),imshape,fitradius)
    nx = bbox[1]-bbox[0]
    ny = bbox[3]-bbox[2]
    ravelindex = np.zeros(npix*npix,np.int64)-1
    fcount = 0
    count = 0
    skycount = 0
    for j in range(nskypix):
        y = j + skybbox[2]
        for i in range(nskypix):
            x = i + skybbox[0]
            if mask[y,x]:   # exclude bad pixels
                continue
            r = np.sqrt((x-xcen)**2 + (y-ycen)**2)
            if x>=skybbox[0] and x<=skybbox[1]-1 and y>=skybbox[2] and y<=skybbox[3]-1:
                #if x>=fbbox[0] and x<=fbbox[1]-1 and y>=fbbox[2] and y<=fbbox[3]-1:
                if r <= 1.0*skyradius and r >= 0.7*skyradius:
                    skymulti_index = (np.array([y]),np.array([x]))
                    skyravelindex[skycount] = utils.ravel_multi_index(skymulti_index,imshape)[0]
                    skycount += 1
                if r <= 1.0*hpsfnpix:
                    fmulti_index = (np.array([y]),np.array([x]))
                    fravelindex[fcount] = utils.ravel_multi_index(fmulti_index,imshape)[0]
                    fcount += 1
                if r <= fitradius:
                    multi_index = (np.array([y]),np.array([x]))
                    ravelindex[count] = utils.ravel_multi_index(multi_index,imshape)[0]
                    count += 1
    return (fravelindex,fcount,ravelindex,count,skyravelindex,skycount)

#@njit(cache=True)
@njit
def collatestars(imshape,mask,starx,stary,hpsfnpix,fitradius,skyradius):
    """ Get full footprint and fitted pixels data for all stars."""
    nstars = len(starx)
    nfpix = 2*hpsfnpix+1
    npix = int(np.floor(2*fitradius))+2
    # Full footprint arrays
    fravelindex = np.zeros((nstars,nfpix*nfpix),np.int64)
    fndata = np.zeros(nstars,np.int32)
    # Fitting pixel arrays
    ravelindex = np.zeros((nstars,npix*npix),np.int64)
    ndata = np.zeros(nstars,np.int32)
    skyravelindex = np.zeros((nstars,int(3.14*skyradius**2)),np.int64)
    skyndata = np.zeros(nstars,np.int32)
    for i in range(nstars):
        out = getstar(imshape,mask,starx[i],stary[i],hpsfnpix,fitradius,skyradius)
        # full footprint information
        fravelindex1,fn1,ravelindex1,n1,skyravelindex1,skyn1 = out
        fravelindex[i,:] = fravelindex1
        fndata[i] = fn1
        # fitting pixel information
        ravelindex[i,:] = ravelindex1
        ndata[i] = n1
        # sky pixel information
        skyravelindex[i,:] = skyravelindex1
        skyndata[i] = skyn1
    # Trim arrays
    maxfn = np.max(fndata)
    fravelindex = fravelindex[:,:maxfn]
    maxn = np.max(ndata)
    ravelindex = ravelindex[:,:maxn]
    maxskyn = np.max(skyndata)
    skyravelindex = skyravelindex[:,:maxskyn]
    return (fravelindex,fndata,ravelindex,ndata,skyravelindex,skyndata)

#@njit(cache=True)
@njit
def getstarsky(skyravelindex,resid,error):
    """ Calculate the local sky for a single star."""
    res = resid[skyravelindex]
    err = error.ravel()[skyravelindex]
    sky = utils.skyval(res,err)
    return sky

#@njit(cache=True)
@njit
def starcov(psftype,psfparams,psflookup,imshape,pars,xind,yind,ravelindex,resid,error):
    """ Determine the covariance matrix for a single star."""

    # https://stats.stackexchange.com/questions/93316/parameter-uncertainty-after-non-linear-least-squares-estimation
    # more background here, too: http://ceres-solver.org/nnls_covariance.html

    res = resid[ravelindex]
    err = error.ravel()[ravelindex]
    wt = 1/err**2    # weights
    n = len(xind)
    # Hessian = J.T * T, Hessian Matrix
    #  higher order terms are assumed to be small
    # https://www8.cs.umu.se/kurser/5DA001/HT07/lectures/lsq-handouts.pdf
    m,j = mnb.psf(xind,yind,pars,psftype,psfparams,psflookup,
                  imshape,deriv=True,verbose=False)
    # Weights
    #   If weighted least-squares then
    #   J.T * W * J
    #   where W = I/sig_i**2
    hess = j.T @ (np.diag(wt) @ j)
    #hess = mjac.T @ mjac  # not weighted
    # cov = H-1, covariance matrix is inverse of Hessian matrix
    cov_orig = utils.inverse(hess)
    # Rescale to get an unbiased estimate
    # cov_scaled = cov * (RSS/(m-n)), where m=number of measurements, n=number of parameters
    # RSS = residual sum of squares
    #  using rss gives values consistent with what curve_fit returns
    #cov = cov_orig * (np.sum(resid**2)/(self.ntotpix-len(self.pars)))
    # Use chi-squared, since we are doing the weighted least-squares and weighted Hessian
    chisq = np.sum(res**2/err**2)
    cov = cov_orig * (chisq/(n-3))  # what MPFITFUN suggests, but very small

    # cov = lqr.jac_covariange(mjac,resid,wt)
        
    return cov

#@njit(cache=True)
@njit
def starfit(psftype,psfparams,psflookup,imshape,
            pars,xind,yind,ravelindex,resid,error,sky):
    """
    Fit a single star
    single iteration of the non-linear least-squares loop
    """
    
    res = resid[ravelindex]
    err = error.ravel()[ravelindex]
    wt = 1/err**2    # weights

    # Subtract sky from the residuals
    res -= sky
    
    # Jacobian solvers
    # Get the Jacobian and model
    #  only for pixels that are affected by the "free" parameters
    model0,j = mnb.psf(xind,yind,pars,psftype,psfparams,psflookup,
                       imshape,deriv=True,verbose=False)
    # Solve Jacobian
    dbeta = utils.qr_jac_solve(j,res,weight=wt)
    dbeta[~np.isfinite(dbeta)] = 0.0  # deal with NaNs, shouldn't happen
    # Perform line search
    # This has the previous best-fit model subtracted
    #  add it back in
    data = res + model0
    # Models
    model1,_ = mnb.psf(xind,yind,pars+0.5*dbeta,psftype,psfparams,psflookup,
                       imshape,deriv=False,verbose=False)
    model2,_ = mnb.psf(xind,yind,pars+dbeta,psftype,psfparams,psflookup,
                       imshape,deriv=False,verbose=False)
    chisq0 = np.sum((data-model0)**2/err**2)
    chisq1 = np.sum((data-model1)**2/err**2)
    chisq2 = np.sum((data-model2)**2/err**2)
    #if verbose:
    #    print('linesearch:',chisq0,chisq1,chisq2)
    alpha = utils.quadratic_bisector(np.array([0.0,0.5,1.0]),
                                     np.array([chisq0,chisq1,chisq2]))
    if alpha <= 0 and np.min(np.array([chisq0,chisq1,chisq2]))==chisq2:
        # the bisector can be negative if the chisq shape is concave
        alpha = 1.0
    alpha = np.minimum(np.maximum(alpha,0.0),1.0)  # 0<alpha<1
    if np.isfinite(alpha)==False:
        alpha = 1.0
    pars_new = pars + alpha * dbeta
    new_dbeta = alpha * dbeta
    
    # Update parameters
    bounds = utils.mkbounds(pars,imshape)
    maxsteps = utils.steps(pars)
    bestpars = utils.newpars(pars,new_dbeta,bounds,maxsteps)

    # Calculate chisq with updated resid array
    #bestchisq = np.sum(res**2/err**2)
    #rms = np.sqrt(np.mean((res/bestpars[0])**2))

    #return bestpars,bestchisq,rms
    return bestpars

#@njit(cache=True)
@njit
def chisq(ravelindex,resid,error):
    """ Compute total chi-square of the current best-fit solution for all
        fitting pixels."""
    chisq = np.sum(resid[ravelindex]**2/error.ravel()[ravelindex]**2)
    return chisq

#@njit(cache=True)
@njit
def dofreeze(oldpars,newpars,minpercdiff):
    """ Check if we should freeze this star."""
    # Check differences and changes
    diff = np.abs(newpars-oldpars)
    percdiff = diff.copy()*0
    percdiff[0] = diff[0]/np.maximum(oldpars[0],0.0001)*100  # amp
    percdiff[1] = diff[1]*100               # x
    percdiff[2] = diff[2]*100               # y
    # Freeze parameters/stars that converged
    if np.sum(percdiff<=minpercdiff)==3:
        result = True
    else:
        result = False
    return result
 

#@njit(cache=True)
@njit
def allfit(psftype,psfparams,psfnpix,psflookup,psfflux,
           image,error,mask,tab,fitradius,maxiter=10,
           minpercdiff=0.5,reskyiter=2,verbose=False,
           nofreeze=False):
    """
    Fit PSF to all stars iteratively

    Parameters
    ----------
    psftype : int
       PSF type.
    psfparams: numpy array
       PSF analytical parameters.
    psfnpix : int
       Number of pixels in the PSF footprint.
    psflookup : numpy array
       PSF lookup table.  Must be 3D.
    psfflux : float
       PSF flux.
    image : numpy array
       The image to fit.
    error : numpy array
       Uncertainty for image.
    mask : numpy array
       Boolean mask for image.  True - bad, False - good.
    tab : numpy array
       Table of initial guesses for each star.
       id, amp, xcen, ycen
    maxiter : int, optional
       Maximum number of iterations to allow.  Only for methods "cholesky", "qr" or "svd".
       Default is 10.
    minpercdiff : float, optional
       Minimum percent change in the parameters to allow until the solution is
       considered converged and the iteration loop is stopped.  Only for methods
       "cholesky", "qr" and "svd".  Default is 0.5.
    reskyiter : int, optional
       After how many iterations to re-calculate the sky background. Default is 2.
    nofreeze : boolean, optional
       Do not freeze any parameters even if they have converged.  Default is False.
    verbose : boolean, optional
       Verbose output.

    Returns
    -------
    out : table
       Table of best-fitting parameters for each star.
       id, amp, amp_error, x, x_err, y, y_err, sky
    model : numpy array
       Best-fitting model of the stars and sky background.
    sky : numpy array
       Best-fitting sky image.

    Examples
    --------

    out,model,sky = allfit(psf,image,error,mask,tab)
    
    """
    
    nstars = len(tab)
    _,_,npsforder = psflookup.shape
    psforder = 0
    if npsforder==4:
        psforder = 1
    if psflookup.ndim != 3:
        raise Exception('psflookup must have 3 dimensions')
    psforder = psflookup.shape[2]
    if psforder>1:
        psforder = 1
    ny,nx = image.shape                             # save image dimensions, python images are (Y,X)
    nx = nx
    ny = ny
    imshape = np.array([ny,nx])
    im = image.copy().astype(np.float64)
    err = error.copy().astype(np.float64)
    msk = mask.copy().astype(np.bool_)
    xx,yy = utils.meshgrid(np.arange(nx),np.arange(ny))
    xx = xx.flatten()
    yy = yy.flatten()
    # Order stars by flux, brightest first
    si = np.argsort(tab[:,1])[::-1]       # largest amp first
    startab = tab[si]                     # ID, amp, xcen, ycen
    initpars = startab[:,1:4]             # amp, xcen, ycen
    nfitpix = int(np.ceil(fitradius))     # +/- nfitpix
    skyradius = psfnpix//2 + 10
    # Initialize the parameter array
    initpars = startab[:,1:4] 
    pars = np.zeros(nstars*3,float) # amp, xcen, ycen
    pars[0::3] = initpars[:,0]
    pars[1::3] = initpars[:,1]
    pars[2::3] = initpars[:,2]
    
    # Get information for all the stars
    xcen = pars[1::3]
    ycen = pars[2::3]
    hpsfnpix = psfnpix//2
    out = collatestars(imshape,msk,xcen,ycen,hpsfnpix,fitradius,skyradius)
    starravelindex,starndata,starfitravelindex,starfitndata,skyravelindex,skyndata = out
    
    # Put indices of all the unique fitted pixels into one array
    ntotpix = np.sum(starfitndata)
    allravelindex = np.zeros(ntotpix,np.int64)
    count = 0
    for i in range(nstars):
        n1 = starfitndata[i]
        ravelind1 = starfitravelindex[i,:n1]
        allravelindex[count:count+n1] = ravelind1
        count += n1
    allravelindex = np.unique(allravelindex)
    ravelindex = allravelindex
    ntotpix = len(allravelindex)
    
    # Create initial smooth sky image
    skyim = utils.sky(im).flatten()
    # Subtract the initial models from the residual array
    resid = np.zeros(imshape[0]*imshape[1],np.float64)
    resid[:] = im.copy().astype(np.float64).flatten()   # flatten makes it easier to modify
    resid[:] -= skyim   # subtract smooth sky
    modelim = np.zeros(imshape[0]*imshape[1],np.float64)
    for i in range(nstars):
        pars1 = pars[3*i:3*i+3]
        n1 = starndata[i]
        ravelind1 = starravelindex[i]
        xind1 = xx[ravelind1]
        yind1 = yy[ravelind1]
        m,_ = mnb.psf(xind1,yind1,pars1,psftype,psfparams,psflookup,
                      imshape,deriv=False,verbose=False)
        resid[ravelind1] -= m
        modelim[ravelind1] += m
        
    # Sky and Niter arrays for the stars
    starsky = np.zeros(nstars,np.float64)
    starniter = np.zeros(nstars,np.int64)
    starchisq = np.zeros(nstars,np.float64)
    starrms = np.zeros(nstars,np.float64)
    freezestars = np.zeros(nstars,np.bool_)
    
    # While loop
    niter = 1
    maxpercdiff = 1e10
    nfreestars = nstars
    while (niter<maxiter and nfreestars>0):
        start0 = clock()
            
        # Star loop
        for i in range(nstars):
            # Fit the single star (if not frozen)
            if freezestars[i]==True:
                continue
            
            # Get fitting information for this star
            pars1 = pars[3*i:3*i+3]
            fn1 = starfitndata[i]
            fravelindex1 = starfitravelindex[i,:fn1]
            fxind1 = xx[fravelindex1]
            fyind1 = yy[fravelindex1]
            
            # Get local sky
            sn1 = skyndata[i]
            skyravelindex1 = skyravelindex[i,:sn1]
            sky1 = getstarsky(skyravelindex1,resid,err)
            starsky[i] = sky1
            
            # Get new best parameters
            newpars1 = starfit(psftype,psfparams,psflookup,imshape,
                               pars1,fxind1,fyind1,fravelindex1,resid,err,sky1)
            freezestars[i] = dofreeze(pars1,newpars1,minpercdiff)

            # Update the residuals for a star's full footprint
            #  add in the previous full footprint model
            #  and subtract the new full footprint model
            n1 = starndata[i]
            ravelindex1 = starravelindex[i,:n1]
            xind1 = xx[ravelindex1]
            yind1 = yy[ravelindex1]
            prevmodel,_ = mnb.psf(xind1,yind1,pars1,psftype,psfparams,psflookup,
                                  imshape,deriv=False,verbose=False)
            newmodel,_ = mnb.psf(xind1,yind1,newpars1,psftype,psfparams,psflookup,
                                 imshape,deriv=False,verbose=False)
            resid[ravelindex1] += prevmodel
            resid[ravelindex1] -= newmodel
            # Update the model image
            modelim[ravelindex1] += prevmodel
            modelim[ravelindex1] -= newmodel
            
            # Calculate chisq with updated resid array
            chisq1 = np.sum(resid[fravelindex1]**2/error.ravel()[fravelindex1]**2)
            rms1 = np.sqrt(np.mean((resid[fravelindex1]/newpars1[0])**2))
            
            # Save new values
            pars[3*i:3*i+3] = newpars1
            starniter[i] = niter
            starchisq[i] = chisq1
            starrms[i] = rms1
    
            if verbose:
                print('Iter = ',niter)
                print('Pars = ',newpars1)
                print('chisq = ',chisq1)
                
        # Re-estimate the sky
        if niter % reskyiter == 0:
            if verbose:
                print('Re-estimating the sky')
            prevsky = skyim.copy()
            # Remove the current best-fit model
            tresid = im - modelim.copy().reshape(imshape[0],imshape[1])    # remove model
            skyim = utils.sky(tresid).flatten()
            # Update resid
            resid[:] += prevsky
            resid[:] -= skyim
            
        if verbose:
            print('iter dt =',(clock()-start0)/1e9,'sec.')

        nfreezestars = np.sum(freezestars)
        nfreestars = nstars-nfreezestars
            
        niter += 1     # increment counter

        if verbose:
            print('niter=',niter)
            
    # Check that all starniter are set properly
    #  if we stopped "prematurely" then not all stars were frozen
    #  and didn't have starniter set
    starniter[np.where(starniter==0)] = niter
    
    # Calculate parameter uncertainties
    # estimate uncertainties
    # Calculate covariance matrix
    perror = np.zeros(len(pars),np.float64)
    for i in range(nstars):
        # Get fitting information for this star
        pars1 = pars[3*i:3*i+3]
        fn1 = starfitndata[i]
        fravelindex1 = starfitravelindex[i,:fn1]
        fxind1 = xx[fravelindex1]
        fyind1 = yy[fravelindex1]
        cov1 = starcov(psftype,psfparams,psflookup,imshape,
                       pars1,fxind1,fyind1,fravelindex1,resid,err)
        perror1 = np.sqrt(np.diag(cov1))
        perror[3*i:3*i+3] = perror1
    perror[:] = perror
    
    # Put in catalog
    outtab = np.zeros((nstars,15),np.float64)
    outtab[:,0] = np.arange(nstars)+1                # id
    outtab[:,1] = pars[0:-1:3]                       # amp
    outtab[:,2] = perror[0:-1:3]                     # amp_error
    outtab[:,3] = pars[1::3]                         # x
    outtab[:,4] = perror[1::3]                       # x_error
    outtab[:,5] = pars[2::3]                         # y
    outtab[:,6] = perror[2::3]                       # y_error
    outtab[:,7] = starsky                            # sky
    outtab[:,8] = outtab[:,1]*psfflux                # flux
    outtab[:,9] = outtab[:,2]*psfflux                # flux_error
    outtab[:,10] = -2.5*np.log10(np.maximum(outtab[:,8],1e-10))+25.0   # mag
    outtab[:,11] = (2.5/np.log(10))*outtab[:,9]/outtab[:,8]            # mag_error
    outtab[:,12] = starrms
    outtab[:,13] = starchisq
    outtab[:,14] = starniter                      # niter, what iteration it converged on
    
    return outtab,modelim,skyim
    

def pallfit(psf,image,tab,fitradius=0.0,maxiter=10,minpercdiff=0.5,
           reskyiter=2,nofreeze=False,verbose=False):
    """
    Fit PSF to all stars in an image iteratively.

    Parameters
    ----------
    psf : PSF object
       PSF object to use for the fitting.
    image : CCDData object
       Image to use to fit PSF model to stars.
    tab : table
       Catalog with initial amp/x/y values for the stars to use to fit the PSF.
       To pre-group the stars, add a "group_id" in the catalog.
    fitradius : float, optional
       The fitting radius in pixels.  By default the PSF FWHM is used.
           reskyiter=2,nofreeze=False,skyfit=True,verbose=False):
    maxiter : int, optional
       Maximum number of iterations to allow.  Only for methods "cholesky", "qr" or "svd".
       Default is 10.
    minpercdiff : float, optional
       Minimum percent change in the parameters to allow until the solution is
       considered converged and the iteration loop is stopped.  Only for methods
       "cholesky", "qr" and "svd".  Default is 0.5.
    reskyiter : int, optional
       After how many iterations to re-calculate the sky background. Default is 2.
    nofreeze : boolean, optional
       Do not freeze any parameters even if they have converged.  Default is False.
    verbose : boolean, optional
       Verbose output.

    Returns
    -------
    out : table
       Table of best-fitting parameters for each star.
       id, amp, amp_error, x, x_err, y, y_err, sky
    model : numpy array
       Best-fitting model of the stars and sky background.
    sky : numpy array
       Best-fitting sky image.

    Example
    -------

    outtab,model,sky = gf.fit()

    """

    t0 = time.time()

    # if skyfit==False:
    #     self.freezepars[-1] = True  # freeze sky value
        
    # # Centroids fixed
    # if recenter==False:
    #     self.freezepars[1::3] = True  # freeze X values
    #     self.freezepars[2::3] = True  # freeze Y values

    # Create AllFitter object
    fitradius = 3.0
    if psf.haslookup:
        psflookup = psf.lookup.copy().astype(np.float64)
    else:
        psflookup = np.zeros((1,1,1),np.float64)
    psfflux = psf.flux()
    out = numba_allfit(psf.psftype,psf.params,psf.npix,psflookup,psfflux,
                       image.data,image.error,image.mask,tab,fitradius,
                       maxiter,minpercdiff,reskyiter,verbose,nofreeze)
    outtab,model,skyim = out
    model = model.reshape(image.shape)
    skyim = skyim.reshape(image.shape)
    return outtab,model,skyim
        
    #af = AllFitter(psf.psftype,psf.params,psf.npix,psflookup,
    #               image.data,image.error,image.mask,tab,
    #               fitradius,verbose,nofreeze)
    ## Fit the stars iteratively
    #af.fit()
    
    # Make final model
    # self.unfreeze()
    model = af.modelim.copy().reshape(af.imshape)
    skyim = af.skyim.copy().reshape(af.imshape)

    # Parameters and uncertainties
    pars = af.pars
    perror = af.perror

    # Put in catalog
    outtab = np.zeros((af.nstars,15),np.float64)
    outtab[:,0] = np.arange(af.nstars)+1           # id
    outtab[:,1] = pars[0:-1:3]                       # amp
    outtab[:,2] = perror[0:-1:3]                     # amp_error
    outtab[:,3] = pars[1::3]                         # x
    outtab[:,4] = perror[1::3]                       # x_error
    outtab[:,5] = pars[2::3]                         # y
    outtab[:,6] = perror[2::3]                       # y_error
    outtab[:,7] = af._starsky                        # sky
    psfflux = psf.flux()
    outtab[:,8] = outtab[:,1]*psfflux                # flux
    outtab[:,9] = outtab[:,2]*psfflux                # flux_error
    outtab[:,10] = -2.5*np.log10(np.maximum(outtab[:,8],1e-10))+25.0   # mag
    outtab[:,11] = (2.5/np.log(10))*outtab[:,9]/outtab[:,8]            # mag_error
    outtab[:,12] = af.starrms
    outtab[:,13] = af.starchisq
    outtab[:,14] = af.starniter                      # niter, what iteration it converged on

    if verbose:
        print('dt =',(clock()-start)/1e9,'sec.')

    return outtab,model,skyim

        

#@njit
def fit(psf,image,tab,fitradius=0.0,recenter=True,maxiter=10,minpercdiff=0.5,
        reskyiter=2,nofreeze=False,skyfit=True,verbose=False):
    """
    Fit PSF to all stars in an image.

    To pre-group the stars, add a "group_id" in the input catalog.

    Parameters
    ----------
    psf : PSF object
       PSF object with initial parameters to use.
    image : CCDData object
       Image to use to fit PSF model to stars.
    tab : table
       Catalog with initial amp/x/y values for the stars to use to fit the PSF.
       To pre-group the stars, add a "group_id" in the catalog.
    fitradius : float, optional
       The fitting radius in pixels.  By default the PSF FWHM is used.
    recenter : boolean, optional
       Allow the centroids to be fit.  Default is True.
    maxiter : int, optional
       Maximum number of iterations to allow.  Only for methods "qr" or "svd".
       Default is 10.
    minpercdiff : float, optional
       Minimum percent change in the parameters to allow until the solution is
       considered converged and the iteration loop is stopped.  Only for methods
       "qr" and "svd".  Default is 0.5.
    reskyiter : int, optional
       After how many iterations to re-calculate the sky background. Default is 2.
    nofreeze : boolean, optional
       Do not freeze any parameters even if they have converged.  Default is False.
    skyfit : boolean, optional
       Fit a constant sky offset with the stellar parameters.  Default is True.
    verbose : boolean, optional
       Verbose output.

    Returns
    -------
    results : table
       Table of best-fitting parameters for each star.
       id, amp, amp_error, x, x_err, y, y_err, sky
    model : numpy array
       Best-fitting model of the stars and sky background.

    Example
    -------

    results,model = fit(psf,image,tab,groups)

    """

    start = clock()
    nstars = len(tab)
    ny,nx = image.shape

    # Groups
    if 'group_id' not in tab.keys():
        daogroup = DAOGroup(crit_separation=2.5*psf.fwhm())
        starlist = tab.copy()
        starlist['x_0'] = tab['x']
        starlist['y_0'] = tab['y']
        # THIS TAKES ~4 SECONDS!!!!!! WAY TOO LONG!!!!
        star_groups = daogroup(starlist)
        tab['group_id'] = star_groups['group_id']

    # Star index
    starindex = utils.index(np.array(tab['group_id']))
    #starindex = dln.create_index(np.array(tab['group_id'])) 
    groups = starindex['value']
    ngroups = len(groups)
    if verbose:
        print(ngroups,'star groups')

    # Initialize catalog
    #dt = np.dtype([('id',int),('amp',float),('amp_error',float),('x',float),
    #               ('x_error',float),('y',float),('y_error',float),('sky',float),
    #               ('flux',float),('flux_error',float),('mag',float),('mag_error',float),
    #               ('niter',int),('group_id',int),('ngroup',int),('rms',float),('chisq',float)])
    outtab = np.zeros((nstars,17),dtype=dt)
    outtab[:,0] = tab[:,0]  # copy ID


    # Group Loop
    #---------------
    resid = image.copy()
    outmodel = np.zeros(image.shape,np.float64)
    outsky = np.zeros(image.shape,np.float64)
    for g,grp in enumerate(groups):
        ind = starindex['index'][starindex['lo'][g]:starindex['hi'][g]+1]
        nind = len(ind)
        inptab = tab[ind].copy()
        if 'amp' not in inptab.columns:
            # Estimate amp from flux and fwhm
            # area under 2D Gaussian is 2*pi*A*sigx*sigy
            if 'fwhm' in inptab.columns:
                amp = inptab['flux']/(2*np.pi*(inptab['fwhm']/2.35)**2)
            else:
                amp = inptab['flux']/(2*np.pi*(psf.fwhm()/2.35)**2)                
            staramp = np.maximum(amp,0)   # make sure it's positive
            inptab['amp'] = staramp
        
        if verbose:
            print('-- Group '+str(grp)+'/'+str(len(groups))+' : '+str(nind)+' star(s) --')

        # Single Star
        if nind==1:
            inptab = np.array([inptab[1],inptab[2],inptab[3]])
            out,model = psf.fit(resid,inptab,niter=3,verbose=verbose,retfullmodel=True,recenter=recenter)
            model.data -= out['sky']   # remove sky
            outmodel.data[model.bbox.slices] += model.data
            outsky.data[model.bbox.slices] = out['sky']

        # Group
        else:
            bbox = cutoutbbox(image,psf,inptab)
            out,model,sky = gfit.fit(psf,resid[bbox.slices],inptab,fitradius=fitradius,
                                     recenter=recenter,maxiter=maxiter,minpercdiff=minpercdiff,
                                     reskyiter=reskyiter,nofreeze=nofreeze,verbose=verbose,
                                     skyfit=skyfit,absolute=True)
            outmodel.data[model.bbox.slices] += model.data
            outsky.data[model.bbox.slices] = sky
            
        # Subtract the best model for the group/star
        resid[model.bbox.slices].data -= model.data

        # Put in catalog
        cols = ['amp','amp_error','x','x_error','y','y_error',
                'sky','flux','flux_error','mag','mag_error','niter','rms','chisq']
        for c in cols:
            outtab[c][ind] = out[c]
        outtab['group_id'][ind] = grp
        outtab['ngroup'][ind] = nind
        outtab = Table(outtab)
        
    if verbose:
        print('dt = {:.2f} sec.'.format(time.time()-t0))
    
    return outtab,outmodel,outsky
        
            

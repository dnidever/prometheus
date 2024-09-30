import os
import numpy as np
from numba import njit,types
from . import models_numba as mnb

# Fit a PSF model to multiple stars in an image

#@njit
def starcube(tab,image,npix=51,fillvalue=np.nan):
    """
    Produce a cube of cutouts of stars.

    Parameters
    ----------
    tab : table
       The catalog of stars to use.  This should have "x" and "y" columns and
         preferably also "amp".
    image : CCDData object
       The image to use to generate the stellar images.
    fillvalue : float, optional
       The fill value to use for pixels that are bad are off the image.
            Default is np.nan.

    Returns
    -------
    cube : numpy array
       Two-dimensional cube (Npix,Npix,Nstars) of the star images.

    Example
    -------

    cube = starcube(tab,image)

    """

    # Get the residuals data
    nstars = len(tab)
    nhpix = npix//2
    cube = np.zeros((npix,npix,nstars),float)
    xx,yy = np.meshgrid(np.arange(npix)-nhpix,np.arange(npix)-nhpix)
    rr = np.sqrt(xx**2+yy**2)        
    x = xx[0,:]
    y = yy[:,0]
    for i in range(nstars):
        xcen = tab['x'][i]            
        ycen = tab['y'][i]
        bbox = mnb.starbbox((xcen,ycen),image.shape,nhpix)
        im = image[bbox.slices]
        flux = image.data[bbox.slices]-image.sky[bbox.slices]
        err = image.error[bbox.slices]
        if 'amp' in tab.columns:
            amp = tab['amp'][i]
        elif 'peak' in tab.columns:
            amp = tab['peak'][i]
        else:
            amp = flux[int(np.round(ycen)),int(np.round(xcen))]
        xim,yim = np.meshgrid(im.x,im.y)
        xim = xim.astype(float)-xcen
        yim = yim.astype(float)-ycen
        # We need to interpolate this onto the grid
        f = RectBivariateSpline(yim[:,0],xim[0,:],flux/amp)
        im2 = np.zeros((npix,npix),float)+np.nan
        xcover = (x>=bbox.ixmin-xcen) & (x<=bbox.ixmax-1-xcen)
        xmin,xmax = dln.minmax(np.where(xcover)[0])
        ycover = (y>=bbox.iymin-ycen) & (y<=bbox.iymax-1-ycen)
        ymin,ymax = dln.minmax(np.where(ycover)[0])            
        im2[ymin:ymax+1,xmin:xmax+1] = f(y[ycover],x[xcover],grid=True)
        # Stuff it into 3D array
        cube[:,:,i] = im2
    return cube


#@njit
def mkempirical(cube,order=0,coords=None,shape=None,lookup=False):
    """
    Take a star cube and collapse it to make an empirical PSF using median
    and outlier rejection.

    Parameters
    ----------
    cube : numpy array
      Three-dimensional cube of star images (or residual images) of shape
        (Npix,Npix,Nstars).
    order : int, optional
      The order of the variations. 0-constant, 1-linear terms.  If order=1,
        Then coords and shape must be input.
    coords : tuple, optional
      Two-element tuple of the X/Y coordinates of the stars.  This is needed
        to generate the linear empirical model (order=1).
    shape : tuple, optional
      Two-element tuple giving the shape (Ny,Nx) of the image.  This is
        needed to generate the linear empirical model (order=1).
    lookup : boolean, optional
      Parameter to indicate if this is a lookup table.  If lookup=False, then
      the constant term is constrained to be non-negative. Default is False.

    Returns
    -------
    epsf : numpy array
      The empirical PSF model  If order=0, then this is just a 2D image.  If
        order=1, then it will be a 3D cube (Npix,Npix,4) where the four terms
        are [constant, X-term, Y-term, X*Y-term].  If rect=True, then a list
        of RectBivariateSpline functions are returned.

    Example
    -------

    epsf = mkempirical(cube,order=0)

    or

    epsf = mkempirical(cube,order=1,coords=coords,shape=im.shape)

    """

    ny,nx,nstar = cube.shape
    npix = ny
    nhpix = ny//2
    
    # Do outlier rejection in each pixel
    med = np.nanmedian(cube,axis=2)
    bad = ~np.isfinite(med)
    if np.sum(bad)>0:
        med[bad] = np.nanmedian(med)
    sig = dln.mad(cube,axis=2)
    bad = ~np.isfinite(sig)
    if np.sum(bad)>0:
        sig[bad] = np.nanmedian(sig)        
    # Mask outlier points
    outliers = ((np.abs(cube-med.reshape((med.shape)+(-1,)))>3*sig.reshape((med.shape)+(-1,)))
                & np.isfinite(cube))
    nbadstar = np.sum(outliers,axis=(0,1))
    goodmask = ((np.abs(cube-med.reshape((med.shape)+(-1,)))<3*sig.reshape((med.shape)+(-1,)))
                & np.isfinite(cube))    
    # Now take the mean of the unmasked pixels
    macube = np.ma.array(cube,mask=~goodmask)
    medim = macube.mean(axis=2)
    medim = medim.data

    # Check how well each star fits the median
    goodpix = macube.count(axis=(0,1))
    rms = np.sqrt(np.nansum((cube-medim.reshape((medim.shape)+(-1,)))**2,axis=(0,1))/goodpix)

    xx,yy = np.meshgrid(np.arange(npix)-nhpix,np.arange(npix)-nhpix)
    rr = np.sqrt(xx**2+yy**2)        
    x = xx[0,:]
    y = yy[:,0]
    mask = (rr<=nhpix)
    
    # Constant
    if order==0:
        # Make sure it goes to zero at large radius
        medim *= mask  # mask corners
        # Make sure values are positive
        if lookup==False:
            medim = np.maximum(medim,0.0)
        else:
            fpars = medim
            
    # Linear
    elif order==1:
        if coords is None or shape is None:
            raise ValueError('Need coords and shape with order=1')
        fpars = np.zeros((ny,nx,4),float)
        # scale coordinates to -1 to +1
        xcen,ycen = coords
        relx,rely = mnb.relcoord(xcen,ycen,shape)
        # Loop over pixels and fit line to x/y
        for i in range(ny):
            for j in range(nx):
                data1 = cube[i,j,:]
                if np.sum(np.abs(data1)) != 0:
                    # maybe use a small maxiter
                    pars1,perror1 = utils.poly2dfit(relx,rely,data1)
                    fpars[i,j,:] = pars1
                
    return fpars,nbadstar,rms


#@njit
def fitpsf(psf,image,tab,fitradius=None,method='qr',maxiter=10,minpercdiff=1.0,
           verbose=False):
    """
    Fit PSF model to stars in an image.

    Parameters
    ----------
    psf : PSF object
       PSF object with initial parameters to use.
    image : CCDData object
       Image to use to fit PSF model to stars.
    tab : table
       Catalog with initial amp/x/y values for the stars to use to fit the PSF.
    fitradius : float, table
       The fitting radius.  If none is input then the initial PSF FWHM will be used.
    method : str, optional
       Method to use for solving the non-linear least squares problem: "qr",
       "svd", "cholesky", and "curve_fit".  Default is "qr".
    maxiter : int, optional
       Maximum number of iterations to allow.  Only for methods "qr", "svd", and "cholesky".
       Default is 10.
    minpercdiff : float, optional
       Minimum percent change in the parameters to allow until the solution is
       considered converged and the iteration loop is stopped.  Only for methods
       "qr" and "svd".  Default is 1.0.
    verbose : boolean, optional
       Verbose output.

    Returns
    -------
    newpsf : PSF object
       New PSF object with the best-fit model parameters.
    pars : numpy array
       Array of best-fit model parameters
    perror : numpy array
       Uncertainties in "pars".
    psftab : table
       Table of best-fitting amp/xcen/ycen values for the PSF stars.

    Example
    -------

    newpsf,pars,perror,psftab = fitpsf(psf,image,tab)

    """

    t0 = time.time()
    print = utils.getprintfunc() # Get print function to be used locally, allows for easy logging   

    # Initialize the output catalog best-fitting values for the PSF stars
    dt = np.dtype([('id',int),('amp',float),('x',float),('y',float),('npix',int),('rms',float),
                   ('chisq',float),('ixmin',int),('ixmax',int),('iymin',int),('iymax',int)])
    psftab = np.zeros(len(tab),dtype=dt)
    if 'id' in tab.colnames:
        psftab['id'] = tab['id']
    else:
        psftab['id'] = np.arange(len(tab))+1
    

    # Fitting the PSF to the stars
    #-----------------------------

    # Empirical PSF - done differently
    if type(psf)==mnb.PSFEmpirical:
        cube1 = starcube(tab,image,npix=psf.npix,fillvalue=np.nan)
        coords = (tab['x'].data,tab['y'].data)
        epsf1,nbadstar1,rms1 = mkempirical(cube1,order=psf.order,coords=coords,shape=psf._shape)
        initpsf = mnb.PSFEmpirical(epsf1,imshape=image.shape,order=psf.order)
        pf = PSFFitter(initpsf,image,tab,fitradius=fitradius,verbose=False)
        # Fit the amp, xcen, ycen properly
        xdata = np.arange(pf.ntotpix)
        out = pf.model(xdata,[])
        # Put information into the psftab table
        psftab['amp'] = pf.staramp
        psftab['x'] = pf.starxcen
        psftab['y'] = pf.starycen
        psftab['chisq'] = pf.starchisq
        psftab['rms'] = pf.starrms
        psftab['npix'] = pf.starnpix    
        for i in range(len(tab)):
            bbox = pf.bboxdata[i]
            psftab['ixmin'][i] = bbox.ixmin
            psftab['ixmax'][i] = bbox.ixmax
            psftab['iymin'][i] = bbox.iymin
            psftab['iymax'][i] = bbox.iymax        
        psftab = Table(psftab)
        # Remake the empirical EPSF    
        cube = starcube(psftab,image,npix=psf.npix,fillvalue=np.nan)
        epsf,nbadstar,rms = mkempirical(cube,order=psf.order,coords=coords,shape=psf._shape)
        newpsf = mnb.PSFEmpirical(epsf,imshape=image.shape,order=psf.order)
        if verbose:
            print('Median RMS: '+str(np.median(pf.starrms)))
            print('dt = %.2f sec' % (time.time()-t0))
        return newpsf, None, None, psftab, pf

    
    pf = PSFFitter(psf,image,tab,fitradius=fitradius,verbose=False) #verbose)
    xdata = np.arange(pf.ntotpix)
    initpar = psf.params.copy()
    method = str(method).lower()
    

    # Iterate
    count = 0
    percdiff = 1e10
    bestpar = initpar.copy()

    dchisq = -1
    oldchisq = 1e30
    bounds = psf.bounds
    maxsteps = psf._steps
    while (count<maxiter and percdiff>minpercdiff and dchisq<0):
        # Get the Jacobian and model
        m,jac = pf.jac(xdata,*bestpar,retmodel=True)
        chisq = np.sum((pf.imflatten-m)**2/pf.errflatten**2)
        dy = pf.imflatten-m
        # Weights
        wt = 1/pf.errflatten**2
        # Solve Jacobian
        dbeta = lsq.jac_solve(jac,dy,method=method,weight=wt)

        # Perform line search
        alpha,new_dbeta = pf.linesearch(xdata,bestpar,dbeta,m,jac)
            
        if verbose:
            print('  pars = '+str(bestpar))
            print('  dbeta = '+str(dbeta))

        # Update the parameters
        oldpar = bestpar.copy()
        #bestpar = psf.newpars(bestpar,dbeta,bounds,maxsteps)
        bestpar = psf.newpars(bestpar,new_dbeta,bounds,maxsteps)  
        diff = np.abs(bestpar-oldpar)
        denom = np.abs(oldpar.copy())
        denom[denom==0] = 1.0  # deal with zeros
        percdiff = np.max(diff/denom*100)
        dchisq = chisq-oldchisq
        percdiffchisq = dchisq/oldchisq*100
        oldchisq = chisq
        count += 1
            
        if verbose:
            print('  '+str(count+1)+' '+str(bestpar)+' '+str(percdiff)+' '+str(chisq))
                
    # Make the best model
    bestmodel = pf.model(xdata,*bestpar)

    # Estimate uncertainties
    if method != 'curve_fit':
        # Calculate covariance matrix
        cov = lsq.jac_covariance(jac,dy,wt=wt)
        perror = np.sqrt(np.diag(cov))
                
    pars = bestpar
    if verbose:
        print('Best-fitting parameters: '+str(pars))
        print('Errors: '+str(perror))
        print('Median RMS: '+str(np.median(pf.starrms)))

    # create the best-fitting PSF
    newpsf = psf.copy()
    newpsf._params = pars                

    # Output best-fitting values for the PSF stars as well
    dt = np.dtype([('id',int),('amp',float),('x',float),('y',float),('npix',int),('rms',float),
                   ('chisq',float),('ixmin',int),('ixmax',int),('iymin',int),('iymax',int)])
    psftab = np.zeros(len(tab),dtype=dt)
    if 'id' in tab.colnames:
        psftab['id'] = tab['id']
    else:
        psftab['id'] = np.arange(len(tab))+1
    psftab['amp'] = pf.staramp
    psftab['x'] = pf.starxcen
    psftab['y'] = pf.starycen
    psftab['chisq'] = pf.starchisq
    psftab['rms'] = pf.starrms
    psftab['npix'] = pf.starnpix    
    for i in range(len(tab)):
        bbox = pf.bboxdata[i]
        psftab['ixmin'][i] = bbox.ixmin
        psftab['ixmax'][i] = bbox.ixmax
        psftab['iymin'][i] = bbox.iymin
        psftab['iymax'][i] = bbox.iymax        
    psftab = Table(psftab)
    
    if verbose:
        print('dt = %.2f sec' % (time.time()-t0))
        
    # Make the star models
    #starmodels = pf.starmodel(pars=pars)
    
    return newpsf, pars, perror, psftab, pf


#@njit
def getpsf(psf,image,tab,fitradius=None,lookup=False,lorder=0,method='qr',subnei=False,
           alltab=None,maxiter=10,minpercdiff=1.0,reject=False,maxrejiter=3,verbose=False):
    """
    Fit PSF model to stars in an image with outlier rejection of badly-fit stars.

    Parameters
    ----------
    psf : PSF object
       PSF object with initial parameters to use.
    image : CCDData object
       Image to use to fit PSF model to stars.
    tab : table
       Catalog with initial amp/x/y values for the stars to use to fit the PSF.
    fitradius : float, table
       The fitting radius.  If none is input then the initial PSF FWHM will be used.
    lookup : boolean, optional
       Use an empirical lookup table.  Default is False.
    lorder : int, optional
       The order of the spatial variations (0=constant, 1=linear).  Default is 0.
    method : str, optional
       Method to use for solving the non-linear least squares problem: "qr",
       "svd", "cholesky", and "curve_fit".  Default is "qr".
    subnei : boolean, optional
       Subtract stars neighboring the PSF stars.  Default is False.
    alltab : table, optional
       Catalog of all objects in the image.  This is needed for bad PSF star
       rejection.
    maxiter : int, optional
       Maximum number of iterations to allow.  Only for methods "qr", "svd", and "cholesky".
       Default is 10.
    minpercdiff : float, optional
       Minimum percent change in the parameters to allow until the solution is
       considered converged and the iteration loop is stopped.  Only for methods
       "qr" and "svd".  Default is 1.0.
    reject : boolean, optional
       Reject PSF stars with high RMS values.  Default is False.
    maxrejiter : int, boolean
       Maximum number of PSF star rejection iterations.  Default is 3.
    verbose : boolean, optional
       Verbose output.

    Returns
    -------
    newpsf : PSF object
       New PSF object with the best-fit model parameters.
    pars : numpy array
       Array of best-fit model parameters
    perror : numpy array
       Uncertainties in "pars".
    psftab : table
       Table of best-fitting amp/xcen/ycen values for the PSF stars.

    Example
    -------

    newpsf,pars,perror,psftab = getpsf(psf,image,tab)

    """

    t0 = time.time()
    print = utils.getprintfunc() # Get print function to be used locally, allows for easy logging   

    psftype,psfpars,_,_ = mnb.unpackpsf(psf)
    
    # Fitting radius
    if fitradius is None:
        tpars = np.zeros(len(psfpars)+3,float)
        tpars[0] = 1.0
        tpars[3:] = psfpars
        if psftype == 3:  # Penny
            fitradius = mnb.penny2d_fwhm(tpars)*1.5
        else:
            fitradius = mnb.model2d_fwhm(psftype,tpars)
        
    # subnei but no alltab input
    if subnei and alltab is None:
        raise ValueError('alltab is needed for PSF neighbor star subtraction')
        
    if 'id' not in tab.dtype.names:
        tab['id'] = np.arange(len(tab))+1
    psftab = tab.copy()

    # Initializing output PSF star catalog
    dt = np.dtype([('id',int),('amp',float),('x',float),('y',float),('npix',int),
                   ('rms',float),('chisq',float),('ixmin',int),('ixmax',int),
                   ('iymin',int),('iymax',int),('reject',int)])
    outtab = np.zeros(len(tab),dtype=dt)
    outtab = Table(outtab)
    for n in ['id','x','y']:
        outtab[n] = tab[n]
    
    # Remove stars that are too close to the edge
    ny,nx = image.shape
    bd = ((psftab['x']<fitradius) | (psftab['x']>(nx-1-fitradius)) |
          (psftab['y']<fitradius) | (psftab['y']>(ny-1-fitradius)))
    nbd = np.sum(bd)
    if nbd > 0:
        if verbose:
            print('Removing '+str(nbd)+' stars near the edge')
        psftab = psftab[~bd]

    # Generate an empirical image of the stars
    # and fit a model to it to get initial estimates
    if psftype != 6:
        cube = starcube(psftab,image,npix=psf.npix,fillvalue=np.nan)
        epsf,nbadstar,rms = mkempirical(cube,order=0)
        #epsfim = CCDData(epsf,error=epsf.copy()*0+1,mask=~np.isfinite(epsf))
        epsfim = epsf.copy()
        epsferr = np.ones(epsf.shape,float)
        ny,nx = epsf.shape
        xx,yy = np.meshgrid(np.arange(nx),np.arange(ny))
        out = model2dfit(epsfim,epsferr,xx,yy,psftype,1.0,nx//2,ny//2,verbose=False)
        pars,perror,cov,flux,fluxerr,chisq = out
        mparams = pars[3:]  # model parameters
        #pars,perror,mparams = mnb.model2d_fit(epsfim,pars=[1.0,psf.npix/2,psf.npix//2])
        initpar = mparams.copy()
        curpsf = mnb.packpsf(psftype,mparams,0,0)
        #curpsf = psf.copy()
        #curpsf.params = initpar
        if verbose:
            print('Initial estimate from empirical PSF fit = '+str(mparams))
    else:
        curpsf = psf.copy()
        _,initpar,_,_ = mnb.unpackpsf(psf)
        #initpar = psf.params.copy()

    # Outlier rejection iterations
    nrejiter = 0
    flag = 0
    nrejstar = 100
    fitrad = fitradius
    useimage = image.copy()
    while (flag==0):
        if verbose:
            print('--- Iteration '+str(nrejiter+1)+' ---')                

        # Update the fitting radius
        if nrejiter>0:
            fitrad = curpsf.fwhm()
        if verbose:
            print('  Fitting radius = %5.3f' % (fitrad))
                    
        # Reject outliers
        if reject and nrejiter>0:
            medrms = np.median(ptab['rms'])
            sigrms = dln.mad(ptab['rms'].data)
            gd, = np.where(ptab['rms'] < medrms+3*sigrms)
            nrejstar = len(psftab)-len(gd)
            if verbose:
                print('  RMS = %6.4f +/- %6.4f' % (medrms,sigrms))
                print('  Threshold RMS = '+str(medrms+3*sigrms))
                print('  Rejecting '+str(nrejstar)+' stars')
            if nrejstar>0:
                psftab = psftab[gd]

        # Subtract neighbors
        if nrejiter>0 and subnei:
            if verbose:
                print('Subtracting neighbors')
                # Find the neighbors in alltab
                # Fit the neighbors and PSF stars
                # Subtract neighbors from the image
                useimage = image.copy()  # start with original image
                useimage = subtractnei(useimage,alltab,tab,curpsf)
                
        # Fitting the PSF to the stars
        #-----------------------------
        newpsf,pars,perror,ptab,pf = fitpsf(curpsf,useimage,psftab,fitradius=fitrad,method=method,
                                            maxiter=maxiter,minpercdiff=minpercdiff,verbose=verbose)
        
        # Add information into the output catalog
        ind1,ind2 = dln.match(outtab['id'],ptab['id'])
        outtab['reject'] = 1
        for n in ptab.columns:
            outtab[n][ind1] = ptab[n][ind2]
        outtab['reject'][ind1] = 0

        # Compare PSF parameters
        if type(newpsf)!=mnb.PSFEmpirical:
            pardiff = newpsf.params-curpsf.params
        else:
            pardiff = newpsf._data-curpsf._data
        sumpardiff = np.sum(np.abs(pardiff))
        curpsf = newpsf.copy()
        
        # Stopping criteria
        if reject is False or sumpardiff<0.05 or nrejiter>=maxrejiter or nrejstar==0: flag=1
        if subnei is True and nrejiter==0: flag=0   # iterate at least once with neighbor subtraction
        
        nrejiter += 1
        
    # Generate an empirical look-up table of corrections
    if lookup:
        if verbose:
            print('Making empirical lookup table with order='+str(lorder))

        pf.mklookup(lorder)
        # Fit the stars again and get new RMS values
        xdata = np.arange(pf.ntotpix)
        out = pf.model(xdata,*pf.psf.params)
        newpsf = pf.psf.copy()
        # Update information in the output catalog
        ind1,ind2 = dln.match(outtab['id'],ptab['id'])
        outtab['reject'] = 1
        outtab['reject'][ind1] = 0
        outtab['amp'][ind1] = pf.staramp[ind2]
        outtab['x'][ind1] = pf.starxcen[ind2]
        outtab['y'][ind1] = pf.starycen[ind2]
        outtab['rms'][ind1] = pf.starrms[ind2]
        outtab['chisq'][ind1] = pf.starchisq[ind2]                
        if verbose:
            print('Median RMS: '+str(np.median(pf.starrms)))            
            
    if verbose:
        print('dt = %.2f sec' % (time.time()-t0))
    
    return newpsf, pars, perror, outtab
        
        

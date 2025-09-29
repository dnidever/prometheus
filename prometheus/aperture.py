#!/usr/bin/env python

"""APERTURE.PY - Aperture photometry

"""

__authors__ = 'David Nidever <dnidever@montana.edu?'
__version__ = '20210912'  # yyyymmdd


import os
import sys
import numpy as np
import warnings
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
from scipy.optimize import curve_fit, least_squares
from scipy.interpolate import interp1d
from dlnpyutils import utils as dln, bindata
from photutils.aperture import aperture_photometry, CircularAperture, CircularAnnulus
import copy
import logging
import time
import matplotlib
from .ccddata import BoundingBox,CCDData
from matplotlib.patches import Ellipse
import sep

def circaperphot(im,positions,rap=[5.0],rbin=None,rbout=None):
    """
    Calculate circular aperture photometry for a list of sources.

    Parameters
    ----------
    im : 2D numpy array
       The image to estimate the background for.
    positions : list
       List of two-element positions or catalog.
    rap : float, optional
       Radius of the aperture.  Default is 5.0 pixels.
    rbin : float, optional
       Radius of the inner background aperture.  Default is no background subtraction.
    rbout : float, optional
       Radius of the outer background aperture.  Default is no background subtraction.

    Returns
    -------
    phot : astropy table
       Catalog of measured aperture photometry.

    Example
    -------
    
    phot = circaperphot(im)

    """

    # Positions is a catalog
    if type(positions) is not list and type(positions) is not tuple:    
        pcat = positions
        if 'xpos' in pcat.colnames:
            positions = list(zip(np.array(pcat['xpos']),np.array(pcat['ypos'])))
        elif 'x' in pcat.colnames:
            positions = list(zip(np.array(pcat['x']),np.array(pcat['y'])))
        elif 'xcenter' in pcat.colnames:
            positions = list(zip(np.array(pcat['xcenter']),np.array(pcat['ycenter'])))
        elif 'xcentroid' in pcat.colnames:
            positions = list(zip(np.array(pcat['xcentroid']),np.array(pcat['ycentroid'])))
        else:
            raise ValueError('No X/Y positions found')

    # Do background subtraction
    if rbin is not None:
        doback = True
    else:
        doback = False
        
    # Apertures loop
    for i,rr in enumerate(rap):
        # Define the aperture right around our star
        aperture = CircularAperture(positions, r=rr)
        # Background
        if doback:
            # Define the sky background circular annulus aperture
            annulus_aperture = CircularAnnulus(positions, r_in=rbin, r_out=rbout)
            # This turns our sky background aperture into a pixel mask that we can use to calculate the median value
            annulus_masks = annulus_aperture.to_mask(method='center')
            # Measure the median background value for each star
            bkg_median = []
        area = []
        ones = np.ones(im.shape,int)
        for j in range(len(positions)):
            # Get area in the circular aperture, account for edges        
            mask = aperture[j].to_mask()
            data = mask.multiply(ones)
            area.append(np.sum(data))
            if doback:
                # Get the data in the annulus                
                amask = annulus_masks[j]        
                annulus_data = amask.multiply(im,fill_value=np.nan)
                annulus_data_1d = annulus_data[(amask.data > 0) & np.isfinite(annulus_data)]  # Only want positive values
                _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)  # calculate median
                bkg_median.append(median_sigclip)                            # add to our median list
        if doback:
            bkg_median = np.array(bkg_median)                                # turn into numpy array
        # Calculate the aperture photometry
        phot1 = aperture_photometry(im, aperture)
        if i==0:
            phot = phot1.copy()
        # Stuff it in a table
        phot['aper'+str(i+1)+'_area'] = np.array(area)
        if doback:
            phot['annulus_median'+str(i+1)] = bkg_median
            phot['aper_bkg'+str(i+1)] = bkg_median * phot1['aper_area']
            phot['flux_aper'+str(i+1)] = phot1['aperture_sum'] - phot1['aper_bkg']  # subtract bkg contribution
        else:
            phot['flux_aper'+str(i+1)] = phot1['aperture_sum']            
        phot['mag_aper'+str(i+1)] = -2.5*np.log10(phot['flux_aper'+str(i+1)].data)+25
        
    return phot
    


def aperphot(image,objects,aper=[3],gain=None,mag_zeropoint=25.0):
    """
    Aperture photometry using sep.

    Parameters
    ----------
    im : CCDData object
       The image to estimate the background for.
    objects : table
       Table of objects with x/y coordinate.
    aper : float, optional
       Radius of the aperture.  Default is 3.0 pixels.
    gain : float, optional
       The gain.  Default is 1.
    mag_zeropoint : float
       The magnitude zero-point to use. Default is 25.

    Returns
    -------
    phot : astropy table
       Catalog of measured aperture photometry and other SE
        parameters.

    Example
    -------
    
    phot = aperphot(im,objects)

    """

    if isinstance(image,CCDData) is False:
        raise ValueError("Image must be a CCDData object")

    # Get C-continuous data
    data,error,mask,sky = image.ccont
    data_sub = data-sky
    
    # Get gain from image if possible
    gain = image.gain

    # Initialize the output catalog
    outcat = objects.copy()
    
    # Circular aperture photometry
    for i,ap in enumerate(aper):
        apflux, apfluxerr, apflag = sep.sum_circle(data_sub, outcat['x'], outcat['y'],
                                                   ap, err=error, mask=mask, gain=gain)
        # Add to the catalog
        outcat['flux_aper'+str(i+1)] = apflux
        outcat['fluxerr_aper'+str(i+1)] = apfluxerr
        outcat['mag_aper'+str(i+1)] = -2.5*np.log10(apflux)+mag_zeropoint
        outcat['magerr_aper'+str(i+1)] = (2.5/np.log(10))*(apfluxerr/apflux)  
        outcat['flag_aper'+str(i+1)] = apflag

    # Make sure theta's are between -pi/2 and +pi/2 radians
    if 'theta' in objects.columns:
        theta = objects['theta'].copy()
        hi = theta>0.5*np.pi
        if np.sum(hi)>0:
            theta[hi] -= np.pi
        lo = theta<-0.5*np.pi    
        if np.sum(lo)>0:
            theta[lo] += np.pi
    else:
        theta = np.zeros(len(outcat),float)
            
    # We have morphology parameters
    if 'a' in outcat.columns and 'b' in outcat.columns:
        kronrad, krflag = sep.kron_radius(data_sub, outcat['x'], outcat['y'], outcat['a'],
                                          outcat['b'], theta, 6.0, mask=mask)
    else:
        kronrad, krflag = None, None
        
    # Add more columns
    outcat['flux_auto'] = 0.0
    outcat['fluxerr_auto'] = 0.0
    outcat['mag_auto'] = 0.0
    outcat['magerr_auto'] = 0.0
    outcat['kronrad'] = kronrad
    outcat['flag_auto'] = np.int16(0)

    # BACKGROUND ANNULUS???

    # FLUX_AUTO
    
    # Only use elliptical aperture if Kron radius is large enough
    # Use circular aperture photometry if the Kron radius is too small
    r_min = 1.75  # minimum diameter = 3.5    
    if kronrad is not None:
        use_circle = kronrad * np.sqrt(outcat['a'] * outcat['b']) < r_min
    else:
        use_circle = np.ones(len(outcat),bool)
    nuse_ellipse = np.sum(~use_circle)
    nuse_circle = np.sum(use_circle)
        
    # Elliptical aperture
    if nuse_ellipse>0:
        #import pdb; pdb.set_trace()
        flux, fluxerr, flag = sep.sum_ellipse(data=data_sub, x=outcat['x'][~use_circle], y=outcat['y'][~use_circle],
                                              a=outcat['a'][~use_circle],b=outcat['b'][~use_circle],
                                              theta=outcat['theta'][~use_circle], r=2.5*kronrad[~use_circle],
                                              subpix=1, err=error, mask=mask)
        flag |= krflag[~use_circle]  # combine flags into 'flag'
        outcat['flux_auto'][~use_circle] = flux
        outcat['fluxerr_auto'][~use_circle] = fluxerr
        outcat['mag_auto'][~use_circle] = -2.5*np.log10(flux)+mag_zeropoint
        outcat['magerr_auto'][~use_circle] = (2.5/np.log(10))*(fluxerr/flux) 
        outcat['flag_auto'][~use_circle] = flag
        
    # Use circular aperture photometry if the Kron radius is too small
    if nuse_circle>0:
        cflux, cfluxerr, cflag = sep.sum_circle(data_sub, outcat['x'][use_circle],
                                                outcat['y'][use_circle], r_min, subpix=1,
                                                err=error, mask=mask)
        outcat['flux_auto'][use_circle] = cflux
        outcat['fluxerr_auto'][use_circle] = cfluxerr
        outcat['mag_auto'][use_circle] = -2.5*np.log10(cflux)+mag_zeropoint
        outcat['magerr_auto'][use_circle] = (2.5/np.log(10))*(cfluxerr/cflux) 
        outcat['flag_auto'][use_circle] = cflag
        outcat['kronrad'][use_circle] = r_min

    # Add S/N
    outcat['snr'] = 1.087/outcat['magerr_auto']
    
    return outcat


def sumprofile(rk,A,B,C,Rseeing):
    """
    Sum of radial stellar profile from 0 to a radius rk.
    See Stetson (1990) pg. 4.

    Parameters
    ----------
    rk : float or numpy array
      The radius of the aperture.
    A : float
      A-parameter that affects the asymptotic power-law slope of the outer part of the profile.
    B : float
      B-parameter that affects the relative amplitude of the Moffat function versus the Gaussian
         and exponential part of the profile.
    C : float
      C-parameter that defines the relative importance of the Gaussian and exponential
         contributions to the seeing-dependent part of the profile.
    Rseeing : float
       The seeing radius in pixels.

    Returns
    -------
    sum : float
       The total flux from 0 to radius rk.

    Example
    -------

    sum = sumprofile(xdata,A,B,C,Rseeing)

    """

    # From Stetson (1990), DAOGROW paper, pg. 4
    # I(r,Xi; Ri,A,B,C,D,E) = (B+E*Xi)*(Mr;A) + (1-B-E*Xi)*[ C*G(r;Ri) +
    #                                                         (1-C)*H(r:D*Ri)]
    # where Xi is the airmass of the ith frame
    # Ri is the seeing-related radial scale parameter for the ith frame
    # M, G, and H are Moffat, Gaussian and exponential functions
    #
    # M(r;A) = (A-1)/pi * (1+r**2)**(-A)
    # G(r;Ri) = (1/(2piRi**2)) * exp(-r**2/(2Ri**2))
    # H(r;D*Ri) = (1/(2pi(D*Ri)*22)) * exp(-r/(D*Ri))
    #
    # The fraction of the tota flux of the star which is contained in the aureole
    # is given by B+E*Xi (the parameter E allowing the relative strenght of the
    # wings to depend linearly on airmass).
    # C defines the relative importance of the Gaussian and exponential contributions
    # to the seeing-dependent part of the profile
    # D permits the Gaussian and exponential components to have different - though
    # linearly related - seeing-imposed scale lengths
    # A Moffat function is used rather than a simple power law to prevent an
    # unphysical divergence of the profile at r=0.
    # The Moffat function scale length has been arbitrarily set to 1 pixel.
    # The D and E parameters are comparatively unimportant and could be fixed
    # to 0.9 and 0.0 respectively.
    #
    # Constraints on parameters:
    # A>1
    # 0<=B<=1
    # 0<=C<=1
    # D>0
    #
    # Since I'm only running this on a single frame, we don't need the airmass (Xi)
    # or seeing (Ri) parameters.  Therefore, I'm dropping the D and E parameters.
    # The equation simplify to:
    # I(r;A,B,C,R) = B*(Mr;A)+(1-B)*[C*G(r,R)+(1-C)*H(r,R)]    
    # M(r;A) = (A-1)/pi * (1+r**2)**(-A)
    # G(r;R) = (1/(2piR**2)) * exp(-r**2/(2R**2))
    # H(r;R) = (1/(2piRi**2)) * exp(-r/Ri)


    # The Moffat, Gaussian and Exponential functions have analytic integrals
    # Integral 0->rk  M(r;A) 2pir dr = 1-(1+rk**2)**(-A)
    # Integral 0->rk  G(r;R) 2pir dr = 1-exp(-rk**2/(2R**2))
    # Integral 0->rk  H(r;R) 2pir dr = 1-[1+rk/R]*exp(-rk/R)
    # and have been normalized to have unit total volume when integrated to infinity

    mint = 1-(1+rk**2)**(-A)
    gint = 1-np.exp(-rk**2/(2*Rseeing**2))
    hint = 1-(1+rk/Rseeing)*np.exp(-rk/Rseeing)
    toti = B*mint + (1-B)*( C*gint + (1-C)*hint )
    
    return toti

def diffprofile(xdata,A,B,C,Rseeing):
    """
    Differential stellar flux profile.

    Parameters
    ----------
    xdata : float or numpy array
      Two-element list or tuple of the outer and inner radii.
    A : float
      A-parameter that affects the asymptotic power-law slope of the outer part of the profile.
    B : float
      B-parameter that affects the relative amplitude of the Moffat function versus the Gaussian
         and exponential part of the profile.
    C : float
      C-parameter that defines the relative importance of the Gaussian and exponential
         contributions to the seeing-dependent part of the profile.
    Rseeing : float
       The seeing radius in pixels.

    Returns
    -------
    diff : float
       The relative flux between two radii.

    Example
    -------

    diff = diffprofile(xdata,A,B,C,Rseeing)

    """

    rk = xdata[0]
    rkminus1 = xdata[1]
    diff = -2.5*np.log10( sumprofile(rk,A,B,C,Rseeing) / sumprofile(rkminus1,A,B,C,Rseeing) )
    return diff

def fudgefactor(err,resid,a=2,b=1):
    """
    Lower the weights of outlier points.
    
    Parameters
    ----------
    err : numpy array
       The uncertainty array.
    resid : numpy array
       The residual array of the data minus the best-fit model.
    a : int
       A-parameter, normally kept at 2.  Default is 2.
    b : int
       B-parameter that is normally between 1 and 3.  Default is 1.

    Returns
    -------
    fudge : numpy array
       The fudge factor to apply to the uncertainties to downweight
          outlier points.

    Example
    -------

    fudge = fudgefactor(erri,resid,a=2,b=1)

    """
    # See Stetson (1990) pg. 7
    wt = 1/err**2
    fudge = 1 / (1+np.abs(resid/(a*err))**b)
    # convert to fudge factor for errors
    fudge = 1/np.sqrt(fudge)
    return fudge

def fitgrowth(apercat,apers,rseeing):
    """
    Fit the curve of growth to aperture photometry.

    Parameters
    ----------
    apercat : table
       Table of aperture magnitudes.
    apers : list or numpy array
       List of aperture radii.
    rseeing : float
       The seeing radius in pixels.

    Returns
    -------
    pars : numpy array
       The bestfit parameters.
    agrow : numpy array
       The analytical differential growth curve.
    agrowerr : numpy array
       Uncertainty in agrow.

    Example
    -------
    
    pars,agrow,agrowerr = fitgrowth(apercat,apers,rseeing)

    """
    

    # Get magnitude differences
    nstars = len(apercat)
    napers = len(apers)
    dmag = np.zeros((nstars,napers-1),float)
    derr = np.zeros((nstars,napers-1),float)
    r1 = np.zeros((nstars,napers-1),float)
    r2 = np.zeros((nstars,napers-1),float)    
    for i in range(len(apers)-1):
        mag1 = apercat['mag_aper'+str(i+1)]
        err1 = apercat['magerr_aper'+str(i+1)]
        mag2 = apercat['mag_aper'+str(i+2)]
        err2 = apercat['magerr_aper'+str(i+2)]
        dmag[:,i] = mag2-mag1
        derr[:,i] = np.sqrt(err1**2+err2**2)
        r1[:,i] = apers[i]
        r2[:,i] = apers[i+1]

    # Toss bad values
    bad = (dmag>=0.0)
    dmag[bad] = np.nan
    derr[bad] = np.nan 
        
    # Fit A, B and C
    # allow only A to vary and converge, then allow A and B,
    # then finally A,B and C to vary.

    data = dmag.ravel()
    err = derr.ravel()
    r1ravel = r1.ravel()
    r2ravel = r2.ravel()    
                   
    bestpars = [1.5, 0.5, 0.5, rseeing]
    lbounds = [1.0, 0.0, 0.0, rseeing-1e-7]
    ubounds = [np.inf, 1.0, 1.0, rseeing+1e-7]
    fudgeb = [1, 2, 3]
    for i in range(3):
        bounds = (np.zeros(4,float)-1e-7,np.zeros(4,float)+1e-7)
        bounds[0][:] += bestpars
        bounds[1][:] += bestpars        
        bounds[0][0:i+1] = lbounds[0:i+1]
        bounds[1][0:i+1] = ubounds[0:i+1]

        # Keep only finite values
        gd, = np.where(np.isfinite(data) & np.isfinite(err))
        xdata = [r2ravel[gd], r1ravel[gd]]

        # Do the fit
        pars, cov = curve_fit(diffprofile,xdata,data[gd],bestpars,sigma=err[gd],bounds=bounds)
        # Apply fudge factors to the weights to reduce the weight
        # of the outlier points
        model = diffprofile([r2ravel,r2ravel],*pars)
        resid = data.ravel()-model        
        fudge = fudgefactor(err.ravel(),resid,b=fudgeb[i])
        #fudge = fudge.reshape(derr.shape)
        #err *= fudge
        
        # Apply minimum fudgefactor for that star for all the smaller apertures
        #for j in range(len(apers)-1):
        #    newerr[] = np.max()
        
        #print(i,pars)
        #bestpars = pars

    agrow = diffprofile([apers[1:],apers[:-1]],*bestpars)

    return bestpars,agrow,err



def empgrowth(apercat,apers):
    """
    Calculate empirical growth curve.

    Parameters
    ----------
    apercat : table
       Table of aperture magnitudes.
    apers : list or numpy array
       List of aperture radii.

    Returns
    -------
    egrow : numpy array
       The empirical differential growth curve.
    egrowerr : numpy array
       The uncertainty in egrow.

    Example
    -------

    egrow = empgrowth(apercat,apers)

    """

    # Get magnitude differences
    nstars = len(apercat)
    napers = len(apers)
    dmag = np.zeros((nstars,napers-1),float)
    derr = np.zeros((nstars,napers-1),float)
    for i in range(len(apers)-1):
        mag1 = apercat['mag_aper'+str(i+1)]
        err1 = apercat['magerr_aper'+str(i+1)]
        mag2 = apercat['mag_aper'+str(i+2)]
        err2 = apercat['magerr_aper'+str(i+2)]
        dmag[:,i] = mag2-mag1
        derr[:,i] = np.sqrt(err1**2+err2**2)

    # Toss bad values
    bad = (dmag>=0.0)
    dmag[bad] = np.nan
    derr[bad] = np.nan    
        
    egrow = np.nanmedian(dmag,axis=0)
    egrowerr = np.nanmedian(derr,axis=0)
        
    return egrow, egrowerr
    

def totphot(apercat,apers,cgrow,cgrowerr):
    """
    Calculate total aperture photometry for stars.

    Parameters
    ----------
    apercat : table
       Table of aperture magnitudes.
    apers : list or numpy array
       List of aperture radii.
    cgrow : numpy array
       Cumulative aperture correction array for "apers".
    cgrowerr : numpy array
       Uncertainy in cgrow.

    Returns
    -------
    totmag : numpy array
       Array of total magnitude for each star.
    toterr : numpy array
       Uncertainty in totmag.

    Example
    -------

    totmag,toterr = totphot(aperct,apers,cgrow,cgrowerr)

    """

    # Apply the cumulative aperture correction to each aperture
    # and compute the uncertainty in the corrected magnitude
    # (raw mag error plus correction error)
    # Use the aperture with the lowest error

    nstars = len(apercat)
    napers = len(apers)
    mag = np.zeros((nstars,napers),float)
    err = np.zeros((nstars,napers),float)    
    rr = np.zeros((nstars,napers),float)
    cmag = np.zeros((nstars,napers),float)
    cerr = np.zeros((nstars,napers),float)    
    for i in range(napers):
        mag[:,i] = apercat['mag_aper'+str(i+1)]
        err[:,i] = apercat['magerr_aper'+str(i+1)]        
        cmag[:,i] = mag[:,i] + cgrow[i]
        cerr[:,i] = np.sqrt(err[:,i]**2+cgrowerr[i]**2)
    ind = np.argmin(cerr,axis=1)
    totmag = cmag[np.arange(nstars),ind]
    toterr = cerr[np.arange(nstars),ind]
    return totmag, toterr

def apercorr(psf,image,objects,psfobj,verbose=False):
    """
    Calculate aperture correction.

    Parameters
    ----------
    psf : PSF object
       The best-fitting PSF model.
    image : string or CCDData object
      The input image to fit.  This can be the filename or CCDData object.
    objects : table
       The output table of best-fit PSF values for all of the sources.
    psfobj : table
       The table of PSF objects.
    verbose : boolean, optional
      Verbose output to the screen.  Default is False.
    Returns
    -------
    objects : table
       The output table with an "apcorr" column inserted and the aperture correction
         applied to "psfmag".
    apcor : float
       The aperture correction in mag.
    cgrow : numpy array
       The cumulative aperture correction array.

    Example
    -------

    apcor = apercorr(psf,image,objects,psfobj)

    """

    # Get model of all stars except the PSF stars
    ind1,ind2 = dln.match(objects['id'],psfobj['id'])
    left = np.delete(np.arange(len(objects)),ind1)
    neiobj = objects[left]
    neimodel = image.copy()
    neimodel.data *= 0
    neimodel.error[:] = 1
    neimodelim = psf.add(neimodel,neiobj)
    neimodel.data = neimodelim
    
    # Subtract everything except the PSF stars from the image
    resid = image.copy()
    if image.mask is not None:
        resid.data[~resid.mask] -= neimodel.data[~resid.mask]
    else:
        resid.data -= modelnei.data            
    residim = np.maximum(resid.data-resid.sky,0)
    resid.data = residim
    resid.sky[:] = 0.0
    
    # Do aperture photometry with lots of apertures on the PSF
    #  stars
    # rk = (20/3.)**(1/11.) * rk-1  for k=2,..,12
    rseeing = psf.fwhm()*0.5
    apers = np.cumprod(np.hstack((3.0,np.ones(11,float)*(20/3.)**(1/11.))))  
    #apers = np.array([3.0,3.7965,4.8046,6.0803,7.6947,9.7377,12.3232,
    #                  15.5952,19.7360,24.9762,31.6077,40.0000])


    apercat = aperphot(resid,psfobj,apers)   
    
    # Fit curve of growth
    # use magnitude differences between successive apertures.
    apars, agrow, derr = fitgrowth(apercat,apers,rseeing=psf.fwhm()*0.5)
    

    # Get magnitude difference errors
    nstars = len(apercat)
    napers = len(apers)
    derr = np.zeros((nstars,napers-1),float)
    for i in range(len(apers)-1):
        err1 = apercat['magerr_aper'+str(i+1)]
        err2 = apercat['magerr_aper'+str(i+2)]
        derr[:,i] = np.sqrt(err1**2+err2**2)
    wt = 1/derr**2

    
    # THE CURVE TURNS OVER AT LARGE RADIUS!!!!???
    # It shouldn't EVER do that.

    
    # Calculate empirical growth curve
    egrow,egrowerr = empgrowth(apercat,apers)
    
    # Get "adopted" growth curve by taking the weighted average
    # of the analytical and empirical growth curves
    # with the empirical weighted higher at small r and
    # the analytical weighted higher at large r
    gwt = np.mean(wt,axis=0)   # mean weights over the stars
    adopgrow = (egrow*gwt + agrow*(1/(0.1*agrow))**2) / (gwt+(1/(0.1*agrow))**2)
    adopgrowerr =  1 / (gwt+(1/(0.1*agrow))**2)

    # Adopted cumulative growth curve
    # sum from the outside in, with an outer tail given by
    # extrapolation of the analytic model to 2*outer aperture
    cadopgrow = np.cumsum(adopgrow[::-1])[::-1]
    # add extrapolation from rlast t=o2*rlast
    tail = diffprofile([2*apers[-1],apers[-1]],*apars)
    cadopgrow += tail
    cadopgrow = np.hstack((cadopgrow,tail))  # add value for outer aperture
    cadopgrowerr = np.hstack((adopgrowerr,0.0))
    
    # Calculate "total" magnitude for the PSF stars
    totmag,toterr = totphot(apercat,apers,cadopgrow,cadopgrowerr)
    
    # Calculate median offset between total and PSF magnitude
    # psf - total
    ind1,ind2 = dln.match(objects['id'],psfobj['id'])
    diffmag = objects['psfmag'][ind1] - totmag[ind2]
    apcor = np.median(diffmag)   # positive value
    
    # Apply aperture correction to the data
    #  add apcorr column and keep initial mags in instmag
    objects['apcorr'] = apcor
    objects['inst_psfmag'] = objects['psfmag']
    objects['psfmag'] -= apcor    # make brighter

    if verbose:
        print('Aperture correction = %.3f mag' % apcor)
    
    return objects, apcor, cadopgrow
    

#!/usr/bin/env python

"""PROMETHEUS.PY - PSF photometry

"""

__authors__ = 'David Nidever <dnidever@montana.edu?'
__version__ = '20210915'  # yyyymmdd


import os
import sys
import numpy as np
import warnings
from astropy.io import fits
from astropy.table import Table,vstack
import logging
import time
from dlnpyutils import utils as dln
from . import detection, aperture, models, getpsf, allfit, utils
from .ccddata import CCDData
try:
    import __builtin__ as builtins # Python 2
except ImportError:
    import builtins # Python 3

    
# run PSF fitting on an image
def run(image,psfname='gaussian',detmethod='sep',iterdet=0,ndetsigma=1.5,snrthresh=5,
        psfsubnei=False,psffitradius=None,fitradius=None,npsfpix=51,binned=False,
        lookup=False,lorder=0,psftrim=None,recenter=True,reject=False,apcorr=False,
        timestamp=False,verbose=False):
    """
    Run PSF photometry on an image.

    Parameters
    ----------
    image : string or CCDData object
      The input image to fit.  This can be the filename or CCDData object.
    psfname : string, optional
      The name of the PSF type to use.  The options are "gaussian", "moffat",
      "penny" and "gausspow".  Default is "gaussian".
    detmethod : string, optional
      Detection method.  The options are "sep", "dao" and "iraf".
        Default is "sep".
    iterdet : boolean, optional
      Number of iterations to use for detection.  Default is iterdet=0, meaning
       detection is only performed once.
    ndetsigma : float, optional
      Detection threshold in units of sigma.  Default is 1.5.
    snrthresh : float, optional
       Signal-to-Noise threshold for detections.  Default is 5.
    psfsubnei : boolean, optional
      Subtract neighboring stars to PSF stars when generating the PSF.  Default is False.
    psffitradius : float, optional
       The fitting readius when constructing the PSF (in pixels).  By default
          the FWHM is used.
    fitradius: float, optional
       The fitting radius when fitting the PSF to the stars in the image (in pixels).
         By default the PSF FWHM is used.
    npsfpix : int, optional
       The size of the PSF footprint.  Default is 51.
    binned : boolean, optional
       Use a binned model that integrates the analytical function across a pixel.
         Default is false.
    lookup : boolean, optional
        Use an empirical lookup table.  Default is False.
    lorder : int, optional
       The order of the spatial variations (0=constant, 1=linear).  Default is 0.
    psftrim: float, optional
       Trim the PSF size to a radius where "psftrim" fraction of flux is removed.  Default is None.
    recenter : boolean, optional
       Allow the centroids to be fit.  Default is True.
    reject : boolean, optional
       When constructin the PSF, reject PSF stars with high RMS values.  Default is False.
    apcorr : boolean, optional
       Apply aperture correction.  Default is False.
    timestamp : boolean, optional
         Add timestamp in verbose output (if verbose=True). Default is False.       
    verbose : boolean, optional
      Verbose output to the screen.  Default is False.

    Returns
    -------
    cat : table
       The output table of best-fit PSF values for all of the sources.
    model : CCDData object
       The best-fitting model for the stars (without sky).
    sky : CCDData object
       The background sky image used for the image.
    psf : PSF object
       The best-fitting PSF model.

    Example
    -------

    cat,model,sky,psf = prometheus.run(image,psfname='gaussian',verbose=True)

    """
    
    # Set up the logger
    if timestamp and verbose:
        logger = dln.basiclogger()
        logger.handlers[0].setFormatter(logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s"))
        logger.handlers[0].setStream(sys.stdout)
        builtins.logger = logger   # make it available globally across all modules

    start = time.time()        
    print = utils.getprintfunc() # Get print function to be used locally, allows for easy logging   
    
    # Load the file
    if isinstance(image,str):
        filename = image
        if verbose:
            print('Loading image from "'+filename+'"')
        image = CCDData.read(filename)
    if isinstance(image,CCDData) is False:
        raise ValueError('Input image must be a filename or CCDData object')

    if verbose:
        print('Image shape ',image.shape)
    
    residim = image.copy()
    
    # Processing steps
    #-----------------
    for niter in range(iterdet+1):

        if verbose and iterdet>0:
            print('--- Iteration = '+str(niter+1)+' ---')
    
        # 1) Detection
        #-------------
        if verbose:
            print('Step 1: Detection')
        objects = detection.detect(residim,method=detmethod,
                                   nsigma=ndetsigma,verbose=verbose)
        objects['ndetiter'] = niter+1
        if verbose:
            print(str(len(objects))+' objects detected')
    
        # 2) Aperture photometry
        #-----------------------
        if verbose:
            print('Step 2: Aperture photometry')    
        objects = aperture.aperphot(residim,objects)
        nobjects = len(objects)
        # Bright and faint limit, use 5th and 95th percentile
        if niter==0:
            minmag, maxmag = np.nanpercentile(objects['mag_auto'],(5,95))
            if verbose:
                print('Min/Max mag: %5.2f, %5.2f' % (minmag,maxmag))

        # Imposing S/N cut
        gd, = np.where((objects['snr'] >= snrthresh) & np.isfinite(objects['mag_auto']))
        if len(gd)==0:
            print('No objects passed S/N cut')
            return None,None,None,None
        objects = objects[gd]
        objects['id'] = np.arange(len(objects))+1  # renumber
        if verbose:
            print('%d objects left after S/N=%5.1f cut' % (len(objects),snrthresh))

                
        # 3) Construct the PSF
        #---------------------        
        #  only on first iteration
        if niter==0:
            if verbose:
                print('Step 3: Construct the PSF')
            # 3a) Estimate FWHM
            #------------------
            fwhm = utils.estimatefwhm(objects,verbose=verbose)
    
            # 3b) Pick PSF stars
            #------------------
            psfobj = utils.pickpsfstars(objects,fwhm,verbose=verbose)

            
            # 3c) Construct the PSF iteratively
            #---------------------------------
            # Make the initial PSF slightly elliptical so it's easier to fit the orientation
            if psfname.lower() != 'empirical':
                initpsf = models.psfmodel(psfname,[fwhm/2.35,0.9*fwhm/2.35,0.0],binned=binned,npix=npsfpix)
            else:
                initpsf = models.psfmodel(psfname,npix=npsfpix,imshape=image.shape,order=lorder)
            # run getpsf
            psf,psfpars,psfperror,psfcat = getpsf.getpsf(initpsf,image,psfobj,fitradius=psffitradius,
                                                         lookup=lookup,lorder=lorder,subnei=psfsubnei,
                                                         allcat=objects,reject=reject,verbose=(verbose>=2))

            # Trim the PSF
            if psftrim is not None:
                oldnpix = psf.npix
                psf.trim(psftrim)
                if verbose:
                    print('Trimming PSF size from '+str(oldnpix)+' to '+str(self.npix))
            if verbose:
                print('Final PSF: '+str(psf))
                gd, = np.where(psfcat['reject']==0)
                print('Median RMS:  %.4f' % np.median(psfcat['rms'][gd]))
                
        # 4) Run on all sources
        #----------------------
        # If niter>0, then use combined object catalog
        if iterdet>0:
            # Add detection
            # Combine objects catalogs
            if niter==0:
                allobjects = objects.copy()
            else:
                objects['id'] = np.arange(len(objects))+1+np.max(allobjects['id'])
                allobjects = vstack((allobjects,objects))
            if 'group_id' in allobjects.keys():
                allobjects.remove_column('group_id')
        else:
            allobjects = objects
                
        if verbose:
            print('Step 4: Get PSF photometry for all '+str(len(allobjects))+' objects')
        psfout,model,sky = allfit.fit(psf,image,allobjects,fitradius=fitradius,
                                      recenter=recenter,verbose=(verbose>=2))
        
        # Construct residual image
        if iterdet>0:
            residim = image.copy()
            residim.data -= model.data
            
        # Combine aperture and PSF columns
        outobj = allobjects.copy()
        # rename some columns for clarity
        outobj['x'].name = 'xc'
        outobj['y'].name = 'yc'
        outobj['a'].name = 'asemi'
        outobj['b'].name = 'bsemi'        
        outobj['flux'].name = 'sumflux'
        outobj.remove_columns(['cxx','cyy','cxy'])
        # copy over PSF output columns
        for n in psfout.columns:
            outobj[n] = psfout[n]
        outobj['psfamp'] = outobj['amp'].copy()
        outobj['amp_error'].name = 'psfamp_error'        
        outobj['flux'].name = 'psfflux'
        outobj['flux_error'].name = 'psfflux_error'        
        # change mag, magerr to psfmag, psfmag_error
        outobj['mag'].name = 'psfmag'
        outobj['mag_error'].name = 'psfmag_error'
        # put ID at the beginning
        cols = np.char.array(list(outobj.columns))
        newcols = ['id']+list(cols[cols!='id'])
        outobj = outobj[newcols]
        
    # 5) Apply aperture correction
    #-----------------------------
    if apcorr:
        if verbose:
            print('Step 5: Applying aperture correction')
        outobj,grow,cgrow = aperture.apercorr(psf,image,outobj,psfcat,verbose=verbose)

    # Add exposure time correction
    exptime = image.header.get('exptime')
    if exptime is not None:
        if verbose:
            print('Applying correction for exposure time %.2f s' % exptime)
        outobj['psfmag'] += 2.5*np.log10(exptime)
        
    # Add coordinates if there's a WCS
    if image.wcs is not None:
        if image.wcs.has_celestial:
            if verbose:
                print('Adding RA/DEC coordinates to catalog')
            skyc = image.wcs.pixel_to_world(outobj['x'],outobj['y'])
            outobj['ra'] = skyc.ra
            outobj['dec'] = skyc.dec     
        
    if verbose:
        print('dt = %.2f sec' % (time.time()-start))

    # Breakdown logger
    if timestamp and verbose:
        del builtins.logger
        
    return outobj,model,sky,psf

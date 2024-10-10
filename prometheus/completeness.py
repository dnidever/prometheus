#!/usr/bin/env python

"""COMPLETENESS.PY - Artificial star tests and completeness

"""

__authors__ = 'David Nidever <dnidever@montana.edu?'
__version__ = '20241011'  # yyyymmdd


import os
import sys
import numpy as np
import scipy
import warnings
from astropy.io import fits
from astropy.table import Table
from . import allfit

def ast(image,psf,atab,detmethod='sep',iterdet=0,ndetsigma=1.5,snrthresh=5,
        fitradius=None,recenter=True,apcorr=False,timestamp=False,verbose=False):
        
    """
    Artificial stars.

    Parameters
    ----------
    image : CCDData object
       The image to add artificial stars to.
    psf : PSF object
       The best-fit PSF.
    atab : table
       Table of artificial stars to add.  Need columns: x, y, amp/ht.


    Returns
    -------
    ctab : table
       Completeness table.

    
    Examples
    --------

    ctab = ast(image,psf,atab)


    """

    # Add the stars to a new image
    newim = image.copy()
    for i in range(len(atab)):
        newim += psf.add()
        # add noise too

    # Rerun prometheus run() steps, but use existing PSF
    ######################################################
    # I should really modify prometheus.run() to use an existing input PSF
    # rather than always constructing a new one.
    residim = newimage.copy()


    # Processing steps
    #-----------------
    for niter in range(iterdet+1):
    
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
        psfout,model,sky = allfit.fit(psf,newimage,allobjects,fitradius=fitradius,
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


    ############################################
    # Now figure out which ASTs were recovered and calculate completeness

    
    if verbose:
        print('dt = %.2f sec' % (time.time()-start))

    # Breakdown logger
    if timestamp and verbose:
        del builtins.logger
        
    return outobj,model,sky,psf        

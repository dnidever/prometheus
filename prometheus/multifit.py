# Fit stars in multiple exposures simultaneously ALLFRAME-style

import os
import numpy as np
from scipy import linalg
from scipy import sparse
from . import leastsquares as lsq, utils

# How to do ALLFRAME-like processing
# -Let's say we have 1 star in 20 exposures and there are 10 pixels in each exposure that contribute
#   to the footprint and that we are fitting.
# -The position of the star in each exposure will be set by the global position and proper motion of the
#   star and is NOT something that is fit at the exposure level
# -At the exposure level we are just fitting the flux.  this is more important when there are multiple,
#    overlapping stars and we need to solve for the fluxes simultaneously.
# -For each exposure, calculate the residuals (data - best model) in the 10 pixels, also calculate the
#   jacobian, i.e the derivatives with respect to the three parameters (x position, y position and flux)
#   for the 10 pixels, [Npix,Npar] or [10,3]
# -With our 20 exposures we now have 20 sets of residuals (10 pixels each), and 20 sets of jacobians
#   (10x3 each).
# -Our goal is to solve for the star position (at some reference time) and proper motion in both
#   X and Y (or RA and DEC).  We will need to solve for the fluxes in each exposure as well.
#   We have 20x3 = 60 exposure-lvel parameters, and we want to solve 2 positions + 2 proper motions +
#   20 exposures = 24 parameters
#   How do we accomplish that?
# -We need a new equation with residuals and jabian for the "global" problem, where our parameters are
#   the 4 positions+proper motion and 20 fluxes.
# -For the residuals it's pretty straighforward.  We want to solve all of the pixels in all the exposures
#   simultaneously.  The pixels are independent, so we just concatenate all of the residuals, giving
#   us 20 x 10 = 200 pixels values.
# -For the jacobian we want the derivative of each pixel with respect to X0, Y0, mu_x, mu_y and the
#   fluxes.  We should be able to get this information from the exposure-level jacobians.
#   The flux part is pretty easy.  We just take the flux column of the relevant exposures's jacobian
#   matrix for the 10 pixels of that exposure and rest of the pixels get zero.
# -The more complicated bit is the derivatives with respect to X0/Y0/mu_x/mu_y.
#   But after some thought this is actually also pretty straightforwad.
#   If I shift X0 slightly by +dx, then the position in each exposure will also be shifted by +dx.
#   So the derivative with respect to X0 is the same as the derivative with respect to the central
#   X position in each exposure.  Therefore, the X0 column of the global jacobian is just the
#   concatenation of the x position columns of the exposure-level jacobians, each one contributing 10
#   values. We end up with 10 pixels x 20 exposures = 200 non-zero values in the X0 column.
# -The mu_x is a little bit more complicated, but not much more.  Here the time lapse with respect to
#   the reference epoch (when X0 and Y0 are defined), is important.  We want to know, if we shift
#   mu_x by a small dmu_x value how does that affect the model for each exposure.  The time lapse is
#   just a multiplicative factor here.  The positional shift for exposure i for a small mu_x shift
#   of dmu_x is just dx_i = t_i * dmu_x.  Since we know what the derivative is for each model with
#   respect to the x position in each exposure, all we need to convert that into a derivative with
#   respect to mu_x is the time factor (t_i).
#   The derivative d/dx_i goes to d/d(t_i * mu_x) = 1/t_i * d/dmu_x or
#   d/dmu_x = t_i * d/dx_i
#   Effectively the exposure-level x position derivative gets simply multiplied by the time lapse
#   (t_i) for that exposure.
#   We populate the mu_x column by again concatenating the exposure-level x position jacobian values,
#   appropriately multipled by their exposure's time laps value.
# -We are left with a new "global" problem/equation with 24 parameters and 200 pixels.  The global
#   jacobian matrix (24 parameters x 200 pixels) is mostly empty.  The four positional columns
#   (X0, Y0, mu_x, mu_y) are fully filled with non-zero values, but the other 20 columns for the
#   exposure-level fluxes are mostly empty/zero except for the 10 pixels for that exposure.
# -We could directly solve this with a simple solver of Ax = b using jacobians.  But since most of
#   the jacobian is empty and "banded", there probably are faster techniques.  I'll have to look
#   at Stetson's ALLFRAME paper to see what he does.  I'm guessing this is the matrix manipulation
#   that you mentioned on the call.
# -I wonder if this could be solved iteratively by first solving for the 4 positional parameters,
#   and then solving the fluxes separately (one flux at a time with its 10 pixels)?  I'm not entirely sure.


def applytrans(trans,data):
    """
    Apply linear transformation to data (rotation and scale).

     xi = CD1_1 * x + CD1_2 * y
    eta = CD2_1 * x + CD2_2 * y

    Parameters
    ----------
    trans : numpy array
       Transformation matrix [2 x 2].
    data : numpy array
       Data to be transformed. The second dimension should have size 2.

    Returns
    -------
    transdata : numpy array
       Transformed data.

    Example
    -------

    transdata = applytrans(trans,data)

    """

    outdata = data.copy()
    outdata[:,0] = trans[0,0] * data[:,0] + trans[1,0] * data[:,1]
    outdata[:,1] = trans[1,0] * data[:,0] + trans[1,1] * data[:,1]
    return outdata


def starfit(data,reftime=None):
    """
    Perform multfit on multiple exposures of a single star.
    
    Parameters
    ----------
    data : list
       List of information.  One element per exposure.  Each element should be 
        a dictionary that contains the following information:
          JAC:  Jacobian matrix [Npix,Npars], where Npix is the number of pixels being
                  fitted for that star and exposure and Npars is 3 for x position, y position
                  and flux.
          RESID: Residuals of the best model-fit and the data (data-model) [Npix].
          WEIGHT: The weights (i.e., 1/sigma^2) of the pixels.
          META:  Dictionary of meta-data.  This should the exposure timestamp information
                  (either DATE-TIME, JD or MJD). And TRANS, the transformation of the X/Y positions
                  in this exposure and at the position of this star, into the reference frame
                  (normally RA/DEC in arcsec).  This must be a 2 x 2 rotation and scaling matrix
                  (like the standard WCS CD matrix).
    reftime : float, optional
       Timestamp to use for the reference epoch.  By default, the first exposure is used.

    Returns
    -------
    pars : numpy array
       Final, best-fit parameters [4 + Nexposures]:  Xref, Yref, mu_x, mu_y (proper motion), and
        amplitudes for each exposure.  Note, these are the CORRECTION terms for the 4 parameters,
        no the actual values themselves.
    perror: numpy array
       Uncertainties in pars.

    Example
    -------

    pars,perror = starfit(data)

    """

    nexp = len(data)

    # Construct the "global" arrays
    #   JAC : Jacobian matrix [Npix total, 4 + Nexposures] for all of the pixels across all
    #            of the exposures and the parameters Xref, Yref, mu_x, mu_y, and amplitudes
    #            for all Nexp exposures.
    #   RESID : residuals of all pixels of all exposures concatenated.
    #   WEIGHT : weights of all pixels of all exposures concatenated.

    # Reference time
    if reftime is None:
        reftime = data[0]['META']['MJD']
    
    # Initialize the global arrays
    npix = np.sum([len(d['RESID']) for d in data])  # add up total number of pixels
    npar = 4 + nexp
    jac = np.zeros([npix,npar],float)
    resid = np.zeros(npix,float)
    weight = np.zeros(npix,float)

    # Loop over the exposures
    pixcount = 0
    for i in range(nexp):
        data1 = data[i]
        jac1 = data1['JAC']
        resid1 = data1['RESID']
        weight1 = data1['WEIGHT']
        meta1 = data1['META']
        time1 = meta1['MJD']
        trans1 = meta['TRANS']
        npix1 = len(resid1)

        # Delta time, time since reference epoch
        delta_time = time1 - reftime   # MJD, days
        delta_time /= 365.2425         # convert to years
        
        # Concatenate RESID and WEIGHT
        resid[count:count+npix1] = resid1
        weight[count:count+npix1] = weight1
            
        # Translate X/Y into the reference frame
        tjac1 = jac1.copy()        
        transjac2 = applytrans(trans1,jac[:,0:2])  # transform X/Y into reference frame
        tjac1[:,0:2] = transjac2

        # Fill in the Global Jacobian
        # Convert X and Y derivatives from the exposure-level jacobian
        #   to the global Xref, Yref, mu_x, and mu_y parameters
        # Xref and Yref derivatives are the same as the X/Y derivatives
        jac[count:count+npix1,0] = tjac1[:,0]              # Xref derivative
        jac[count:count+npix1,1] = tjac1[:,1]              # Yref derivative
        # mu_x and mu_y derivatives are just X derivatives * delta_time        
        jac[count:count+npix1,2] = tjac1[:,0]*delta_time   # mu_x derivative (arcsec/year)
        jac[count:count+npix1,3] = tjac1[:,1]*delta_time   # mu_y derivative (arcsec/year)

        # Amplitude/flux derivatives
        jac[count:count+npix1,i] = jac1[:,2]      # amplitude
        
        # Increment pixcount
        pixcount += npix1

    # Now solve the matrix

    #par = sparse.linalg.lsqr(A,dy2,atol=1e-4,btol=1e-4)
    # pars = lsq.cholesky_jac_sparse_solve(jac,resid,weight)

    # Solve Jacobian
    #  these are the correction terms, no the final parameters themselves
    pars = lsq.jac_solve(jac,resid,method='qr',weight=weight)
    cov = lsq.jac_covariance(jac,resid,weight)
    perror = np.sqrt(np.diag(cov))

    return pars,perror
    

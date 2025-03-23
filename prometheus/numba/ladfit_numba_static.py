#!/usr/bin/env python
#
# LADFIT.PY - robust linear fitting.
#

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@montana.edu>'
__version__ = '20200321'  # yyyymmdd

import numpy as np
import warnings
from numba import njit

# Ignore these warnings, it's a bug
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from numba.pycc import CC
cc = CC('_ladfit_numba_static')

@njit
@cc.export('ladmdfuncf', '(f8[:],f8[:],f8[:],f8)')
@cc.export('ladmdfunci', '(i8[:],i8[:],i8[:],f8)')
def ladmdfunc(b, x, y, eps=1e-7):
    a = np.median(y - b*x)
    d = y - (b * x + a)
    absdev = np.sum(np.abs(d))
    nz, = np.where(y != 0.0)
    nzcount = len(nz)
    if nzcount != 0:
        d[nz] = d[nz] / np.abs(y[nz]) #Normalize
    nz, = np.where(np.abs(d) > eps)
    nzcount = len(nz)
    if nzcount != 0:        #Sign fcn, +1 for d > 0, -1 for d < 0, else 0.
        return np.sum(x[nz] * ((d[nz] > 0)*1 - (d[nz] < 0)*1)), a, absdev
    else:
        return 0.0, a, absdev

@njit
@cc.export('ladfitf', '(f8[:],f8[:])')
@cc.export('ladfiti', '(i8[:],i8[:])')
def ladfit(x, y):
    """
    Copyright (c) 1994-2015, Exelis Visual Information Solutions, Inc. All
    rights reserved. Unauthorized reproduction is prohibited.

    LADFIT

    This function fits the paired data {X(i), Y(i)} to the linear model,
    y = A + Bx, using a "robust" least absolute deviation method. The
    result is a two-element vector containing the model parameters, A
    and B.


    Result = LADFIT(X, Y)

    Parameters
    ----------
       X:    An n-element vector of type integer, float or double.

       Y:    An n-element vector of type integer, float or double.

    EXAMPLE:
       Define two n-element vectors of paired data.
         x = [-3.20, 4.49, -1.66, 0.64, -2.43, -0.89, -0.12, 1.41, $
               2.95, 2.18,  3.72, 5.26]
         y = [-7.14, -1.30, -4.26, -1.90, -6.19, -3.98, -2.87, -1.66, $
              -0.78, -2.61,  0.31,  1.74]
       Compute the model parameters, A and B.
         result = ladfit(x, y, absdev = absdev)
       The result should be the two-element vector:
         [-3.15301, 0.930440]
       The keyword parameter should be returned as:
         absdev = 0.636851

    REFERENCE:
       Numerical Recipes, The Art of Scientific Computing (Second Edition)
       Cambridge University Press, 2nd Edition.
       ISBN 0-521-43108-5
    This is adapted from the routine MEDFIT described in:
    Fitting a Line by Minimizing Absolute Deviation, Page 703.

    MODIFICATION HISTORY:
      Written by:  GGS, RSI, September 1994
      Modified:    GGS, RSI, July 1995
                    Corrected an infinite loop condition that occured when
                    the X input parameter contained mostly negative data.
      Modified:    GGS, RSI, October 1996
                    If least-absolute-deviation convergence condition is not
                    satisfied, the algorithm switches to a chi-squared model.
                    Modified keyword checking and use of double precision.
      Modified:    GGS, RSI, November 1996
                    Fixed an error in the computation of the median with
                    even-length input data. See EVEN keyword to MEDIAN.
      Modified:    DMS, RSI, June 1997
         Simplified logic, remove SIGN and MDfunc2 functions.
      Modified:    RJF, RSI, Jan 1999
         Fixed the variance computation by adding some double
         conversions.  This prevents the function from generating
         NaNs on some specific datasets (bug 11680).
       Modified: CT, RSI, July 2002: Convert inputs to float or double.
            Change constants to double precision if necessary.
       CT, March 2004: Check for quick return if we found solution.
    """
    
    nx = len(x)
    
    if nx != len(y):
        raise ValueError("X and Y must be vectors of equal length.")

    sx = np.sum(x)
    sy = np.sum(y)

    #  the variance computation is sensitive to roundoff, so we do this
    #  math in DP
    sxy = np.sum(x*y)
    sxx = np.sum(x*x)
    delx = nx * sxx - sx**2

    if (delx == 0.0):                  # All X's are the same
        result = [np.median(y), 0.0]   # Bisect the range w/ a flat line
        absdev = np.sum(np.abs(y-np.median(y)))/nx
        return np.array(result), absdev

    aa = (sxx * sy - sx * sxy) / delx  # Least squares solution y = x * aa + bb
    bb = (nx * sxy - sx * sy) / delx
    chisqr = np.sum((y - (aa + bb*x))**2)
    sigb = np.sqrt(chisqr / delx)      # Standard deviation
    
    b1 = bb
    eps = 1e-7
    f1,aa,absdev = ladmdfunc(b1, x, y, eps=eps)

    # Quick return. The initial least squares gradient is the LAD solution.
    if (f1 == 0.):
        bb = b1
        absdev = absdev / nx
        return np.array([aa, bb]), absdev

    #delb = ((f1 >= 0) ? 3.0 : -3.0) * sigb
    delb = 3.0*sigb if (f1 >= 0) else -3.0*sigb
    
    b2 = b1 + delb
    f2,aa,absdev = ladmdfunc(b2, x, y, eps=eps)

    while (f1*f2 > 0):      # Bracket the zero of the function
        b1 = b2
        f1 = f2
        b2 = b1 + delb
        f2,aa,absdev = ladmdfunc(b2, x, y, eps=eps)


    # In case we finish early.
    bb = b2
    f = f2

    #Narrow tolerance to refine 0 of fcn.
    sigb = 0.01 * sigb

    while ((np.abs(b2-b1) > sigb) and (f != 0)):  # bisection of interval b1,b2.
        bb = 0.5 * (b1 + b2) 
        if (bb == b1 or bb == b2):
            break
        f,aa,absdev = ladmdfunc(bb, x, y, eps=eps)
        if (f*f1 >= 0):
            f1 = f
            b1 = bb
        else:
            f2 = f
            b2 = bb

    absdev = absdev / nx

    return np.array([aa, bb]), absdev

if __name__ == "__main__":
    cc.compile()

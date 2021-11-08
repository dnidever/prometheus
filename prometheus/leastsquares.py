#!/usr/bin/env python

"""LEASTSQUARES.PY - Least squares solvers

"""

__authors__ = 'David Nidever <dnidever@montana.edu?'
__version__ = '20210908'  # yyyymmdd


import sys
import numpy as np
import scipy

def ishermitian(A):
    """ check if a matrix is Hermitian (equal to it's conjugate transpose)."""
    return np.allclose(A, np.asmatrix(A).H)

def isposdef(A):
    """ Check if a matrix positive definite."""
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False

def inverse(a):
    """ Safely take the inverse of a square 2D matrix."""
    # This checks for zeros on the diagonal and "fixes" them.
    
    # If one of the dimensions is zero in the R matrix [Npars,Npars]
    # then replace it with a "dummy" value.  A large value in R
    # will give a small value in inverse of R.
    badpar, = np.where(np.abs(np.diag(a))<sys.float_info.min)
    if len(badpar)>0:
        a[badpar,badpar] = 1e10
    try:
        ainv = np.linalg.inv(a)
    except:
        print('inverse problem')
        import pdb; pdb.set_trace()
    # Fix values
    a[badpar,badpar] = 0  # put values back
    ainv[badpar,badpar] = 0
    
    return ainv

def checkbounds(pars,bounds):
    """ Check the parameters against the bounds."""
    # 0 means it's fine
    # 1 means it's beyond the lower bound
    # 2 means it's beyond the upper bound
    npars = len(pars)
    lbounds,ubounds = bounds
    check = np.zeros(npars,int)
    check[pars<=lbounds] = 1
    check[pars>=ubounds] = 2
    return check
        
def limbounds(pars,bounds):
    """ Limit the parameters to the boundaries."""
    lbounds,ubounds = bounds
    outpars = np.minimum(np.maximum(pars,lbounds),ubounds)
    return outpars

def limsteps(steps,maxsteps):
    """ Limit the parameter steps to maximum step sizes."""
    signs = np.sign(steps)
    outsteps = np.minimum(np.abs(steps),maxsteps)
    outsteps *= signs
    return outsteps

def newpars(pars,steps,bounds=None,maxsteps=None):
    """ Return new parameters that fit the constraints."""
    # Limit the steps to maxsteps
    if maxsteps is not None:
        limited_steps = limsteps(steps,maxsteps)
    else:
        limited_steps = steps

    # No bounds input
    if bounds is None:
        return pars+limited_steps
        
    # Make sure that these don't cross the boundaries
    lbounds,ubounds = bounds
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
    newpars = pars + newsteps
            
    # Make sure to limit them to the boundaries
    check = checkbounds(newpars,bounds)
    badpars = (check!=0)
    if np.sum(badpars)>0:
        # add a tiny offset so it doesn't fit right on the boundary
        newpars = np.minimum(np.maximum(newpars,lbounds+1e-30),ubounds-1e-30)
    return newpars

def lsq_solve(xdata,data,jac,initpar,error=None,method='qr',model=None,
              bounds=None,fixed=None,steps=None,maxiter=20,minpercdiff=0.5,
              verbose=False):
    """
    Solve a non-linear problem with least squares.
    
    xdata : list or numpy array
        x and y values of the data.
    data : numpy array
       Data values.
    jac : function
       Jacobian function.  If model is not input then this is assumed to return
         *both* the model and jacobian.
    initpar : numpy array
       Initial guess parameters.
    error : numpy array, optional
       Uncertainties in data.
    method : str, optional
       Method to use for solving the non-linear least squares problem: "cholesky",
         "qr", "svd", and "curve_fit".  Default is "qr".
    model : function, optional
       Model function.
    bounds : list, optional
       Input lower and upper bounds/constraints on the fitting parameters (tuple of two
         lists.
    fixed : boolean list, optinal
       List of boolean values if what to hold fixed and what should be free to vary.
    steps : function, optional
       Function to limit the steps to some maximum values.  Should take parameters
         and bounds.
    maxiter : int, optional
       Maximum number of iterations.  Default is 20.
    minpercdiff : float, optional
       Minimum percent change in the parameters to allow until the solution is
         considered converged and the iteration loop is stopped.  Default is 0.5.
    verbose : boolean, optional
       Verbose output to the screen.  Default is False.
    
    Returns
    -------
    pars : numpy array
       Best-fit parameters.
    perror : numpy array
       Uncertainties in best-fit parameters.
    cov : numpy array
       Covariance matrix.

    Example
    -------

    pars,perror,cov = lsq_solve(xdata,data,jac,initpar)


    """


    # ADD A FIXED PARAMETER TO HOLD CERTAIN PARAMETERS FIXED!!
    
    # Iterate
    count = 0
    bestpar = initpar.copy()
    maxpercdiff = 1e10
    if steps is not None:
        maxsteps = steps(initpar,bounds)  # maximum steps
    else:
        maxsteps = None
    if error is not None:
        wt = 1.0/error.ravel()**2
    while (count<maxiter and maxpercdiff>minpercdiff):
        # Use Cholesky, QR or SVD to solve linear system of equations
        if model is None:
            m,j = jac(xdata,*bestpar)
        else:
            m = model(xdata,*bestpar)
            j = jac(xdata,*bestpar)            
        dy = data.ravel()-m.ravel()
        # Solve Jacobian
        if error is not None:
            dbeta = jac_solve(j,dy,method=method,weight=wt)
        else:
            dbeta = jac_solve(j,dy,method=method)
        dbeta[~np.isfinite(dbeta)] = 0.0  # deal with NaNs


        # -add "shift cutting" and "line search" in the least squares method
        # basically scale the beta vector to find the best match.
        # check this out
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.line_search.html
        
        # Update parameters
        oldpar = bestpar.copy()
        # limit the steps to the maximum step sizes and boundaries
        if bounds is not None or maxsteps is not None:
            bestpar = newpars(bestpar,dbeta,bounds,maxsteps)
        else:
            bestpar += dbeta
        # Check differences and changes
        diff = np.abs(bestpar-oldpar)
        denom = np.maximum(np.abs(oldpar.copy()),0.0001)
        percdiff = diff.copy()/denom*100  # percent differences
        maxpercdiff = np.max(percdiff)
                
        if verbose:
            print('N = '+str(ount))
            print('bestpars = '+str(bestpar))
            print('dbeta = '+str(dbeta))
                
        count += 1

    # Get covariance and errors
    if model is None:
        m,j = jac(xdata,*bestpar)
    else:
        m = model(xdata,*bestpar)
        j = jac(xdata,*bestpar)            
    dy = data.ravel()-m.ravel()
    if error is not None:
        cov = jac_covariance(j,dy,wt)
    else:
        cov = jac_covariance(j,dy) 
    perror = np.sqrt(np.diag(cov))

    return bestpar,perror,cov
    

def jac_solve(jac,resid,method=None,weight=None):
    """ Thin wrapper for the various jacobian solver method."""

    npix,npars = jac.shape

    #if npars==3:
    #    import pdb; pdb.set_trace()
    
    ## Check the sample covariance to make sure that each
    ## column is independent (and not zero)
    ## check if the sample covariance matrix is singular (has determinant of zero)
    # Just check if one entire column is zeros
    badpars, = np.where(np.sum(jac==0,axis=0) == npix)
    badpars = []
    usejac = jac
    if len(badpars)>0:
        if len(badpars)==npars:
            raise ValueError('All columns in the Jacobian matrix are zero')
        # Remove the offending column(s) in the Jacobian, solve and put zeros
        #  in the output dbeta array
        if len(badpars)>0:
            usejac = jac.copy()
            usejac = np.delete(usejac,badpars,axis=1)
            goodpars, = np.where(np.sum(jac==0,axis=0) != npix)
            #print('removing '+str(len(badpars))+' parameters ('+'.'.join(badpars.astype(str))+') with all zeros in jacobian')
    else:
        badpars = []
        usejac = jac

    # Solve the problem
    if method=='qr':
        dbeta = qr_jac_solve(usejac,resid,weight=weight)
    elif method=='svd':
        dbeta = svd_jac_solve(usejac,resid,weight=weight)
    elif method=='cholesky':
        dbeta = cholesky_jac_solve(usejac,resid,weight=weight)
    elif method=='lu':
        dbeta = lu_jac_solve(usejac,resid,weight=weight)        
    elif method=='sparse':
        dbeta = cholesky_jac_sparse_solve(usejac,resid,weight=weight)
    elif method=='kkt':
        dbeta = kkt_jac_solve(usejac,resid,weight=weight)        
    else:
        raise ValueError(method+' not supported')
    
    # Add back columns that were all zero
    if len(badpars)>0:
        origdbeta = dbeta.copy()
        dbeta = np.zeros(npars,float)
        dbeta[goodpars] = origdbeta
    
    return dbeta

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

def svd_jac_solve(jac,resid,weight=None):
    """ Solve part of a non-linear least squares equation using Single Value
        Decomposition (SVD) using the Jacobian."""
    # jac: Jacobian matrix, first derivatives, [Npix, Npars]
    # resid: residuals [Npix]

    # Precondition??
    
    # Singular Value decomposition (SVD)
    if weight is None:
        u,s,vt = np.linalg.svd(jac)
        #u,s,vt = sparse.linalg.svds(jac)
        # u: [Npix,Npix]
        # s: [Npars]
        # vt: [Npars,Npars]
        # dy: [Npix]
        sinv = s.copy()*0  # pseudo-inverse
        sinv[s!=0] = 1/s[s!=0]
        npars = len(s)
        dbeta = vt.T @ ((u.T @ resid)[0:npars]*sinv)

    # multply resid and jac by weights
    else:
        u,s,vt = np.linalg.svd( jac * weight.reshape(-1,1) )
        sinv = s.copy()*0  # pseudo-inverse
        sinv[s!=0] = 1/s[s!=0]
        npars = len(s)
        dbeta = vt.T @ ((u.T @ (resid*weight))[0:npars]*sinv)
        
    return dbeta

def cholesky_jac_sparse_solve(jac,resid,weight=None):
    """ Solve part a non-linear least squares equation using Cholesky decomposition
        using the Jacobian, with sparse matrices."""
    # jac: Jacobian matrix, first derivatives, [Npix, Npars]
    # resid: residuals [Npix]

    # Precondition??

    from scipy import sparse
    
    # Multipy dy and jac by weights
    if weight is not None:
        resid = resid * weight        
        jac = jac * weight.reshape(-1,1)

    if weight is None:
        # J * x = resid
        # J.T J x = J.T resid
        # A = (J.T @ J)
        # b = np.dot(J.T*dy)
        # J is [3*Nstar,Npix]
        # A is [3*Nstar,3*Nstar]
        jac = sparse.csc_matrix(jac)  # make it sparse
        A = jac.T @ jac
        b = jac.T.dot(resid)
        # Now solve linear least squares with sparse
        # Ax = b
        from sksparse.cholmod import cholesky
        factor = cholesky(A)
        dbeta = factor(b)

    # multply resid and jac by weights
    else:
        wjac = jac * weight.reshape(-1,1)        
        wjac = sparse.csc_matrix(wjac)  # make it sparse
        A = wjac.T @ wjac
        b = wjac.T.dot(resid*weight)
        # Now solve linear least squares with sparse
        # Ax = b
        from sksparse.cholmod import cholesky
        factor = cholesky(A)
        dbeta = factor(b)

    return dbeta

def kkt_jac_solve(jac,resid,weight=None,maxiter=None):
    """ Solve part a non-linear least squares equation using KKT (Karush-Kuhn-Tucker)
        method with the Jacobian."""
    # jac: Jacobian matrix, first derivatives, [Npix, Npars]
    # resid: residuals [Npix]

    if weight is None:
        # J * x = resid
        # J.T J x = J.T resid
        # A = (J.T @ J)
        # b = np.dot(J.T*dy)
        A = jac.T @ jac
        b = np.dot(jac.T,resid)

    # multply resid and jac by weights        
    else:
        wjac = jac * weight.reshape(-1,1)
        A = wjac.T @ wjac
        b = np.dot(wjac.T,resid*weight)        

    # Use KKT method
    x = scipy.optimize.nnls(A,b,maxiter=maxiter)

    return x

def cholesky_jac_solve(jac,resid,weight=None):
    """ Solve part a non-linear least squares equation using Cholesky decomposition
        using the Jacobian."""
    # jac: Jacobian matrix, first derivatives, [Npix, Npars]
    # resid: residuals [Npix]

    if weight is None:
        # J * x = resid
        # J.T J x = J.T resid
        # A = (J.T @ J)
        # b = np.dot(J.T*dy)
        A = jac.T @ jac
        b = np.dot(jac.T,resid)

    # multply resid and jac by weights        
    else:
        wjac = jac * weight.reshape(-1,1)
        A = wjac.T @ wjac
        b = np.dot(wjac.T,resid*weight)        

    # Now solve linear least squares with cholesky decomposition
    # Ax = b
    #  try Cholesky decomposition, but it can fail if the matrix
    #  is not positive definite
    try:
        x = cholesky_solve(A,b)
    # try LU decomposition if cholesky fails, LU works for more cases
    except:
        x = lu_solve(A,b)
    return x

def cholesky_solve(A,b):
    """ Solve linear least squares problem with Cholesky decomposition."""

    # Now solve linear least squares with cholesky decomposition
    # Ax = b
    # decompose A into L L* using cholesky decomposition
    #  L and L* are triangular matrices
    #  L* is conjugate transpose
    # solve Ly=b (where L*x=y) for y by forward substitution
    # finally solve L*x = y for x by back substitution

    #L = np.linalg.cholesky(A)
    #Lstar = L.T.conj()   # Lstar is conjugate transpose
    #y = scipy.linalg.solve_triangular(L,b)
    #x = scipy.linalg.solve_triangular(Lstar,y)

    # this gives better results, not sure why
    c, low = scipy.linalg.cho_factor(A)
    x = scipy.linalg.cho_solve((c, low), b)
    
    return x

def lu_jac_solve(jac,resid,weight=None):
    """ Solve part a non-linear least squares equation using LU decomposition
        using the Jacobian."""
    # jac: Jacobian matrix, first derivatives, [Npix, Npars]
    # resid: residuals [Npix]

    # Multipy dy and jac by weights
    if weight is not None:
        resid = resid * weight        
        jac = jac * weight.reshape(-1,1)
    
    # J * x = resid
    # J.T J x = J.T resid
    # A = (J.T @ J)
    # b = np.dot(J.T*dy)
    A = jac.T @ jac
    b = np.dot(jac.T,resid)

    # Now solve linear least squares with cholesky decomposition
    # Ax = b    
    return lu_solve(A,b)

def lu_solve(A,b):
    """ Solve linear least squares problem with LU decomposition."""

    lu, piv = scipy.linalg.lu_factor(A)
    x = scipy.linalg.lu_solve((lu, piv), b)
    
    # Solve by two back substitution problems
    # Ax = b
    # Use LU decomposition to get A=LU
    # LUx = b
    # now break into two equations
    # Ly = b, solve by forward substitution to get y
    # Ux = y, solve by backward substitution to get x
    #P,L,U = scipy.linalg.lu(A)
    #y = scipy.linalg.solve_triangular(P@L,b)
    #x = scipy.linalg.solve_triangular(U,y)
    return x
    

def jac_covariance(jac,resid,wt=None):
    """
    Determine the covariance matrix.
    
    Parameters
    ----------
    jac : numpy array
       The 2-D jacobian (first derivative relative to the parameters) array
         with dimensions [Npix,Npar].
    resid : numpy array
       Residual array (data-best model) with dimensions [Npix].
    wt : numpy array, optional
       Weight array (typically 1/sigma**2) with dimensions [Npix].

    Returns
    -------
    cov : numpy array
       Covariance array with dimensions [Npar,Npar].

    Example
    -------

    cov = jac_covariance(jac,resid,wt)

    """
    
    # https://stats.stackexchange.com/questions/93316/parameter-uncertainty-after-non-linear-least-squares-estimation
    # more background here, too: http://ceres-solver.org/nnls_covariance.html        

    # Hessian = J.T * T, Hessian Matrix
    #  higher order terms are assumed to be small
    # https://www8.cs.umu.se/kurser/5DA001/HT07/lectures/lsq-handouts.pdf

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

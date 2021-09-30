#!/usr/bin/env python

"""LEASTSQUARES.PY - Least squares solvers

"""

__authors__ = 'David Nidever <dnidever@montana.edu?'
__version__ = '20210908'  # yyyymmdd


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


def jac_solve(jac,resid,method=None,weight=None):
    """ Thin wrapper for the various jacobian solver method."""

    npix,npars = jac.shape
    
    ## Check the sample covariance to make sure that each
    ## columns is independent (and not zero)
    ## check if the sample covariance matrix is singular (has determinant of zero)
    #sample_cov = np.cov(jac.T)
    #sample_cov_det = np.linalg.det(sample_cov)
    # Just check if one entire column is zeros
    badpars, = np.where(np.sum(jac==0,axis=0) == npix)
    usejac = jac
    badpars = []
    #if sample_cov_det==0:
    #    # Check if a whole row is zero
    #    badpars, = np.where(np.diag(sample_cov)==0.0)
    if len(badpars)>0:
        if len(badpars)==npars:
            raise ValueError('All columns in the Jacobian matrix are zero')
        # Remove the offending column(s) in the Jacobian, solve and put zeros
        #  in the output dbeta array
        if len(badpars)>0:
            usejac = jac.copy()
            usejac = np.delete(usejac,badpars,axis=1)
            goodpars, = np.where(np.diag(sample_cov)!=0.0)
            print('removing '+str(len(badpars))+' parameters ('+'.'.join(badpars)+') with all zeros in jacobian')
    else:
        badpars = []

    if npars==6:
        import pdb; pdb.set_trace()
    #print('badpars = ',badpars)
    
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
        rinv = np.linalg.inv(r)
        dbeta = rinv @ (q.T @ resid)
    # Weights input, multiply resid and jac by weights        
    else:
        q,r = np.linalg.qr( jac * weight.reshape(-1,1) )
        rinv = np.linalg.inv(r)
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
    cov_orig = np.linalg.inv(hess)

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

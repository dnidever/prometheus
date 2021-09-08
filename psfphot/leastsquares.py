#!/usr/bin/env python

"""LEASTSQUARES.PY - Least squares solvers

"""

__authors__ = 'David Nidever <dnidever@montana.edu?'
__version__ = '20210908'  # yyyymmdd


import numpy as np


def jac_solve(jac,resid,method=None,weight=None):
    """ Thin wrapper for the various jacobian solver method."""

    if method=='qr':
        dbeta = qr_jac_solve(jac,resid,weight=weight)
    elif method=='svd':
        dbeta = svd_jac_solve(jac,resid,weight=weight)
    elif method=='cholesky':
        dbeta = cholesky_jac_solve(jac,resid,weight=weight)
    elif method=='sparse':
        dbeta = cholesky_jac_sparse_solve(jac,resid,weight=weight)        
    else:
        raise ValueError(method+' not supported')
    
    return dbeta

def qr_jac_solve(jac,resid,weight=None):
    """ Solve part of a non-linear least squares equation using QR decomposition
        using the Jacobian."""
    # jac: Jacobian matrix, first derivatives, [Npix, Npars]
    # resid: residuals [Npix]
    # weight: weights, ~1/error**2 [Npix]
    
    # Multipy dy and jac by weights
    if weight is not None:
        resid = resid * weight
        jac = jac * weight.reshape(-1,1)
    
    # QR decomposition
    q,r = np.linalg.qr(jac)
    rinv = np.linalg.inv(r)
    dbeta = rinv @ (q.T @ resid)
    return dbeta

def svd_jac_solve(jac,resid,weight=None):
    """ Solve part of a non-linear least squares equation using Single Value
        Decomposition (SVD) using the Jacobian."""
    # jac: Jacobian matrix, first derivatives, [Npix, Npars]
    # resid: residuals [Npix]

    # Precondition??
    
    # Multipy dy and jac by weights
    if weight is not None:
        resid = resid * weight        
        jac = jac * weight.reshape(-1,1)
    
    # Singular Value decomposition (SVD)
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
    return dbeta

def cholesky_jac_sparse_solve(jac,resid,weight=None):
    """ Solve part a non-linear least squares equation using Cholesky decomposition
        using the Jacobian, with sparse matrices."""
    # jac: Jacobian matrix, first derivatives, [Npix, Npars]
    # resid: residuals [Npix]

    # Precondition??

    # Multipy dy and jac by weights
    if weight is not None:
        resid = resid * weight        
        jac = jac * weight.reshape(-1,1)
    
    # J * x = resid
    # J.T J x = J.T resid
    # A = (J.T @ J)
    # b = np.dot(J.T*dy)
    # J is [3*Nstar,Npix]
    # A is [3*Nstar,3*Nstar]
    from scipy import sparse
    jac = sparse.csc_matrix(jac)  # make it sparse
    A = jac.T @ jac
    b = jac.T.dot(resid)
    # Now solve linear least squares with sparse
    # Ax = b
    from sksparse.cholmod import cholesky
    factor = cholesky(A)
    dbeta = factor(b)
    
    # Precondition?

    return dbeta
    
def cholesky_jac_solve(jac,resid,weight=None):
    """ Solve part a non-linear least squares equation using Cholesky decomposition
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
    return cholesky_solve(A,b)


def cholesky_solve(A,b):
    """ Solve linear least squares problem with Cholesky decomposition."""

    # Now solve linear least squares with cholesky decomposition
    # Ax = b
    # decompose A into L L* using cholesky decomposition
    #  L and L* are triangular matrices
    #  L* is conjugate transpose
    # solve Ly=b (where L*x=y) for y by forward substitution
    # finally solve L*x = y for x by back substitution

    L = np.linalg.cholesky(A)
    Lstar = L.T.conj()   # Lstar is conjugate transpose
    y = scipy.linalg.solve_triangular(L,b)
    x = scipy.linalg.solve_triangular(Lstar,y)
    return x

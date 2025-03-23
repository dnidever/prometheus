# import os
import numpy as np
# import numba
from numba import njit,types,from_dtype,typed
# from numba.typed import Dict,List
from numba.experimental import jitclass
from numba_kdtree import KDTree

# Fit a PSF model to multiple stars in an image

PI = 3.141592653589793

from numba.pycc import CC
cc = CC('coords_numba_static')

            
# # from astroML
@njit
@cc.export('crossmatch', '(f8[:,:],f8[:,:],f8,i8)')
def crossmatch(X1, X2, max_distance=np.inf,k=1):
    """Cross-match the values between X1 and X2

    By default, this uses a KD Tree for speed.

    Parameters
    ----------
    X1 : array_like
        first dataset, shape(N1, D)
    X2 : array_like
        second dataset, shape(N2, D)
    max_distance : float (optional)
        maximum radius of search.  If no point is within the given radius,
        then inf will be returned.

    Returns
    -------
    dist, ind: ndarrays
        The distance and index of the closest point in X2 to each point in X1
        Both arrays are length N1.
        Locations with no match are indicated by
        dist[i] = inf, ind[i] = N2
    """
    #X1 = np.asarray(X1, dtype=float)
    #X2 = np.asarray(X2, dtype=float)

    N1, D1 = X1.shape
    N2, D2 = X2.shape

    if D1 != D2:
        raise ValueError('Arrays must have the same second dimension')

    kdt = KDTree(X2)

    dist, ind, neigh = kdt.query(X1, k=k, distance_upper_bound=max_distance)

    return dist, ind
    

# from astroML, modified by D. Nidever
@njit
@cc.export('xmatch', '(f8[:],f8[:],f8[:],f8[:],f8,b1,b1)')
def xmatch(ra1, dec1, ra2, dec2, dcr=2.0, unique=False, sphere=True):
    """Cross-match angular values between RA1/DEC1 and RA2/DEC2

    Find the closest match in the second list for each element
    in the first list and within the maximum distance.

    By default, this uses a KD Tree for speed.  Because the
    KD Tree only handles cartesian distances, the angles
    are projected onto a 3D sphere.

    This can return duplicate matches if there is an element
    in the second list that is the closest match to two elements
    of the first list.

    Parameters
    ----------
    ra1/dec1 : array_like
        first dataset, arrays of RA and DEC
        both measured in degrees
    ra2/dec2 : array_like
        second dataset, arrays of RA and DEC
        both measured in degrees
    dcr : float (optional)
        maximum radius of search, measured in arcsec.
        This can be an array of the same size as ra1/dec1.
    unique : boolean, optional
        Return unique one-to-one matches.  Default is False and
           allows duplicates.
    sphere : boolean, optional
        The coordinates are spherical in degrees.  Otherwise, the dcr
          is assumed to be in the same units as the input values.
          Default is True.


    Returns
    -------
    ind1, ind2, dist: ndarrays
        The indices for RA1/DEC1 (ind1) and for RA2/DEC2 (ind2) of the
        matches, and the distances (in arcsec).
    """
    n1 = len(ra1)
    n2 = len(ra2)
    X1 = np.zeros((n1,2),float)
    X1[:,0] = ra1
    X1[:,1] = dec1
    X2 = np.zeros((n2,2),float)
    X2[:,0] = ra2
    X2[:,1] = dec2
    
    # Spherical coordinates in degrees
    if sphere:
        X1 = X1 * (np.pi / 180.)
        X2 = X2 * (np.pi / 180.)
        #if utils.size(dcr)>1:
        #    max_distance = (np.max(dcr) / 3600) * (np.pi / 180.)
        #else:
        #    max_distance = (dcr / 3600) * (np.pi / 180.)
        max_distance = (dcr / 3600) * (np.pi / 180.)
        
        # Convert 2D RA/DEC to 3D cartesian coordinates
        Y1 = np.zeros((n1,3),float)
        Y1[:,0] = np.cos(X1[:, 0]) * np.cos(X1[:, 1])
        Y1[:,1] = np.sin(X1[:, 0]) * np.cos(X1[:, 1])
        Y1[:,2] = np.sin(X1[:, 1])
        #Y1 = np.transpose(np.vstack([np.cos(X1[:, 0]) * np.cos(X1[:, 1]),
        #                             np.sin(X1[:, 0]) * np.cos(X1[:, 1]),
        #                             np.sin(X1[:, 1])]))
        Y2 = np.zeros((n2,3),float)
        Y2[:,0] = np.cos(X2[:, 0]) * np.cos(X2[:, 1])
        Y2[:,1] = np.sin(X2[:, 0]) * np.cos(X2[:, 1])
        Y2[:,2] = np.sin(X2[:, 1])
        #Y2 = np.transpose(np.vstack([np.cos(X2[:, 0]) * np.cos(X2[:, 1]),
        #                             np.sin(X2[:, 0]) * np.cos(X2[:, 1]),
        #                             np.sin(X2[:, 1])]))

        # law of cosines to compute 3D distance
        max_y = np.sqrt(2 - 2 * np.cos(max_distance))
        k = 1 if unique is False else 10
        dist, ind = crossmatch(Y1, Y2, max_y, k=k)
        # dist has shape [N1,10] or [N1,1] (if unique)
    
        # convert distances back to angles using the law of tangents
        not_infy,not_infx = np.where(~np.isinf(dist))
        #x = 0.5 * dist[not_infy,not_infx]
        #dist[not_infy,not_infx] = (180. / np.pi * 2 * np.arctan2(x,
        #                           np.sqrt(np.maximum(0, 1 - x ** 2))))
        #dist[not_infy,not_infx] *= 3600.0      # in arcsec
    # Regular coordinates
    else:
        k = 1 if unique is False else 10
        dist, ind = crossmatch(X1, X2, dcr, k=k)
        #dist, ind = crossmatch(X1, X2, np.max(dcr), k=k)
        not_infy,not_infx = np.where(~np.isinf(dist))
        
    # Allow duplicates
    if unique==False:

        # no matches
        if len(not_infx)==0:
            return np.array([-1]), np.array([-1]), np.array([np.inf])
        
        # If DCR is an array then impose the max limits for each element
        #if utils.size(dcr)>1:
        #    bd,nbd = utils.where(dist > dcr)
        #    if nbd>0:
        #        dist[bd] = np.inf
        #        not_inf = ~np.isinf(dist)
        
        # Change to the output that I want
        # dist is [N1,1] if unique==False
        ind1 = np.arange(len(ra1))[not_infy]
        ind2 = ind[not_infy,0]
        mindist = dist[not_infy,0]
        
    # Return unique one-to-one matches
    else:

        # no matches
        if np.sum(~np.isinf(dist[:,0]))==0:
            return np.array([-1]), np.array([-1]), np.array([np.inf])
        
        done = 0
        niter = 1
        # Loop until we converge
        while (done==0):

            # If DCR is an array then impose the max limits for each element
            #if utils.size(dcr)>1:
            #    bd,nbd = utils.where(dist[:,0] > dcr)
            #    if nbd>0:
            #        for i in range(nbd):
            #            dist[bd[i],:] = np.inf

            # no matches
            if np.sum(~np.isinf(dist[:,0]))==0:
                return np.array([-1]), np.array([-1]), np.array([np.inf])

            # closest matches
            not_inf1 = ~np.isinf(dist[:,0])
            not_inf1_ind, = np.where(~np.isinf(dist[:,0]))
            ind1 = np.arange(len(ra1))[not_inf1]  # index into original ra1/dec1 arrays
            ind2 = ind[:,0][not_inf1]             # index into original ra2/dec2 arrays
            mindist = dist[:,0][not_inf1]
            if len(ind2)==0:
                return np.array([-1]), np.array([-1]), np.array([np.inf])
            find2 = np.zeros(len(ind2),float)
            find2[:] = ind2
            index = Index(find2)
            # some duplicates to deal with
            bd, = np.where(index.num>1)
            nbd = len(bd)
            if nbd>0:
                ntorem = 0
                for i in range(nbd):
                    ntorem += index.num[bd[i]]-1
                torem = np.zeros(ntorem,np.int32)  # index into shortened ind1/ind2/mindist
                trcount = 0
                for i in range(nbd):
                    # index into shortened ind1/ind2/mindist
                    indx = index.getindex(bd[i])
                    #indx = index['index'][index['lo'][bd[i]]:index['hi'][bd[i]]+1]
                    # keep the one with the smallest minimum distance
                    si = np.argsort(mindist[indx])
                    if index.num[bd[i]]>2:
                        bad = indx[si[1:]]
                        torem[trcount:trcount+len(bad)] = bad    # add
                        trcount += len(bad)
                    else:
                        torem[trcount:trcount+1] = indx[si[1:]][0]  # add single element
                #ntorem = utils.size(torem)
                torem_orig_index = not_inf1_ind[torem]  # index into original ind/dist arrays
                # For each object that was "removed" and is now unmatched, check the next possible
                # match and move it up in the dist/ind list if it isn't INF
                for i in range(ntorem):
                    # There is a next possible match 
                    if ~np.isinf(dist[torem_orig_index[i],niter-1]):
                        temp = np.zeros(10,np.int64)
                        temp[niter:] = ind[torem_orig_index[i],niter:]  #.squeeze()
                        temp[-niter:] = np.zeros(niter,np.int64)-1
                        ind[torem_orig_index[i],:] = temp
                        temp2 = np.zeros(10,float)
                        temp2[niter:] = dist[torem_orig_index[i],niter:]   #.squeeze()
                        temp2[-niter:] = np.zeros(niter,float)+np.inf
                        dist[torem_orig_index[i],:] = temp2
                        #ind[torem_orig_index[i],:] = np.hstack( (ind[torem_orig_index[i],niter:].squeeze(),
                        #                                         np.repeat(-1,niter)) )
                        #dist[torem_orig_index[i],:] = np.hstack( (dist[torem_orig_index[i],niter:].squeeze(),
                        #                                          np.repeat(np.inf,niter)) )
                    # All INFs
                    else:
                        ind[torem_orig_index[i],:] = -1
                        dist[torem_orig_index[i],:] = np.inf
                        # in the next iteration these will be ignored
            else:
                ntorem = 0

            niter += 1
            # Are we done, no duplicates or hit the maximum 10
            if (ntorem==0) or (niter>=10): done=1
                                
    return ind1, ind2, mindist

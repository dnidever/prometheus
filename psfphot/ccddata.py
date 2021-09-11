#!/usr/bin/env python

"""CCDDATA.PY - Thin wrapper around CCDData class

"""

__authors__ = 'David Nidever <dnidever@montana.edu?'
__version__ = '20210908'  # yyyymmdd


import numpy as np
from astropy.nddata import CCDData as CCD,StdDevUncertainty
from photutils.aperture import BoundingBox as BBox


class CCDData(CCD):


    def __init__(self, data, *args, bbox=None, **kwargs):

        # Initialize with the parent...
        super().__init__(data, *args, **kwargs)

        ndim = data.ndim
        if ndim==0:
            if bbox is None: bbox=BoundingBox(0,0,0,0)
            self._bbox = bbox
            self._x = None
            self._y = None         
        elif ndim==1:
            nx, = data.shape
            if bbox is None: bbox=BoundingBox(0,nx,0,0)
            self._bbox = bbox
            self._x = np.arange(bbox.xrange[0],bbox.xrange[-1])
            self._y = None
        elif ndim==2:
            nx,ny = data.shape
            if bbox is None: bbox=BoundingBox(0,nx,0,ny)
            self._bbox = bbox
            self._x = np.arange(bbox.xrange[0],bbox.xrange[-1])
            self._y = np.arange(bbox.yrange[0],bbox.yrange[-1])
        else:
            raise ValueError('3D CCDData not supported')


    # for the string representation also print out the bbox values
        
    def __getitem__(self, item):
        
        # Abort slicing if the data is a single scalar.
        if self.data.shape == ():
            raise TypeError('scalars cannot be sliced.')

        # Single slice or integer
        #   make sure we have values for each dimension        
        if self.ndim==2 and type(item) is not tuple:
            item = (item,slice(None,None,None))

        # Let the other methods handle slicing.
        kwargs = self._slice(item)        
        new = self.__class__(**kwargs)
            
        # Get number of starting values and number of output elements
        # 1-D
        if self.ndim==1:
            nx, = self.data.shape
            # slice object
            if isinstance(item,slice):
                start1,stop1,step1 = item1.indices(nx)                
                nel = stop1-start1
                start = start1
            # Integer
            else:
                nel = 0
                start = item
            newx = self._x[item].copy()
            if nel==0:  # 0-D
                new._x = np.array(newx)
                return new
            new._bbox = BoundingBox(newx[0],newx[-1]+1,0,0)
            new._x = newx
            new._y = None
            
        # 2-D
        elif self.ndim==2:
            shape = self.data.shape
            nel = np.zeros(2,int)
            start = np.zeros(2,int)
            for i in range(2):
                item1 = item[i]
                # Slice object
                if isinstance(item1,slice):
                    start1,stop1,step1 = item1.indices(shape[i])
                    nel[i] = stop1-start1
                    start[i] = start1
                # Integer
                else:
                    nel[i] = 0
                    start[i] = item1
            # python images are (Y,X)
            newx = self._x[item[1]].copy()
            newy = self._y[item[0]].copy()

            # Deal with various output types
            
            # 0-D output
            if np.sum(nel)==0:
                new._x = newx
                new._y = newy                
                return new
                
            # 1-D output
            elif np.min(nel)==0:
                rem, = np.where(nel==0)  # which dimension got removed
                if rem[0]==0:
                    new._bbox = BoundingBox(newy[0],newy[-1]+1,0,0)
                    new._x = newy
                    new._y = None
                else:
                    new._bbox = BoundingBox(newx[0],newx[-1]+1,0,0)
                    new._x = newx
                    new._y = None
            # 2-D output
            else:
                new._bbox = BoundingBox(newx[0],newx[-1]+1,newy[0],newy[-1]+1)
                new._x = newx
                new._y = newy

        else:
            raise ValueError('3D CCDData not supported')            
                
        return new
        
            
    @property
    def bbox(self):
        """ Boundary box."""
        # Upper values are EXCLUSIVE as is normal in python!
        return self._bbox

    @property
    def x(self):
        """ X-array."""
        return self._x

    @property
    def y(self):
        """ Y-array."""
        return self._y 

    
class BoundingBox(BBox):

    def __init__(self, *args, **kwargs):

        # Initialize with the parent...
        super().__init__(*args, **kwargs)

    @property
    def xrange(self):
        return (self.ixmin,self.ixmax)

    @property
    def yrange(self):
        return (self.iymin,self.iymax)    
        
    @property
    def data(self):
        return [(self.ixmin,self.ixmax),(self.iymin,self.iymax)]
        
    def __getitem__(self,item):
        return self.data[item]

    def __array__(self):
        return np.array(self.data)

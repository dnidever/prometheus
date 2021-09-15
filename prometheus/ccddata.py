#!/usr/bin/env python

"""CCDDATA.PY - Thin wrapper around CCDData class

"""

__authors__ = 'David Nidever <dnidever@montana.edu?'
__version__ = '20210908'  # yyyymmdd

import sys
import time
import numpy as np
from astropy.nddata import CCDData as CCD,StdDevUncertainty
from astropy.wcs import WCS
from astropy.io import fits
from photutils.aperture import BoundingBox as BBox
from copy import deepcopy
from . import sky as psky


def poissonnoise(data,gain=1.0,rdnoise=0.0):
    """ Generate Poisson noise model (ala DAOPHOT)."""
    # gain
    # rdnoise
    
    noise = np.sqrt(data/gain + rdnoise**2)
    return noise

def getgain(image):
    """ Get the gain from the header."""

    gain = 1.0  # default
    
    # check if there's a header
    if hasattr(image,'meta'):
        if image.meta is not None:
            # Try all versions of gain
            for f in ['gain','egain','gaina']:
                hgain = image.meta.get(f)
                if hgain is not None:
                    gain = hgain
                    break
    return gain
    
def getrdnoise(image):
    " Get the read noise from the header."""

    rdnoise = 0.0  # default
    
    # check if there's a header
    if hasattr(image,'meta'):
        if image.meta is not None:
            # Try all versions of rdnoise
            for f in ['rdnoise','readnois','enoise','rdnoisea']:
                hrdnoise = image.meta.get(f)
                if hrdnoise is not None:
                    rdnoise = hrdnoise
                    break
    return rdnoise


class CCDData(CCD):


    def __init__(self, data, *args, error=None, bbox=None, gain=None, rdnoise=None, sky=None,
                 copy=True, skyfunc=None, **kwargs):
        # Make sure the original version copies all of the input data
        # otherwise bad things will happen when we convert to native byte-order

        # Pull out error from arguments
        if len(args)>0:
            error = args[0]
            if len(args)==1:
                args = ()
            else:
                args = (None,*args[1:])
        
        # Initialize with the parent...
        super().__init__(data, *args, copy=copy, **kwargs)

        # Error
        self._error = error
        # Sky
        self._sky = sky
        # Sky estimation function
        if skyfunc is not None:
            self._skyfunc = skyfunc
        else:
            self._skyfunc = psky.sepsky

        # Gain
        self._gain = gain
        # Read noise
        self._rdnoise = rdnoise
            
        # Copy
        if copy:
            if self._gain is not None:
                self._gain = deepcopy(self._gain)
            if self._rdnoise is not None:
                self._rdnoise = deepcopy(self._rdnoise)
            if self._error is not None:
                self._error = deepcopy(self._error)
            if self._sky is not None:
                self._sky = deepcopy(self._sky)
            self._skyfunc = deepcopy(self._skyfunc)
            
        ndim = self.data.ndim
        if ndim==0:
            if bbox is None: bbox=BoundingBox(0,0,0,0)
            self._bbox = bbox
            self._x = None
            self._y = None         
        elif ndim==1:
            nx, = self.data.shape
            if bbox is None: bbox=BoundingBox(0,nx,0,0)
            self._bbox = bbox
            self._x = np.arange(bbox.xrange[0],bbox.xrange[-1])
            self._y = None
        elif ndim==2:
            nx,ny = self.data.shape
            if bbox is None: bbox=BoundingBox(0,nx,0,ny)
            self._bbox = bbox
            self._x = np.arange(bbox.xrange[0],bbox.xrange[-1])
            self._y = np.arange(bbox.yrange[0],bbox.yrange[-1])
        else:
            raise ValueError('3D CCDData not supported')
        
        self.native()

        # for sep we need to ensure that the data is "c-contiguous"
        # if we used a slice with no copy, then it won't be
        # check image.data.flags['C_CONTIGUOUS']
        # can make it c-contiguous by doing
        # foo = foo.copy(order='C')

        
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

        # Deal with error
        if self._error is not None:
            new._error = self._error[item]
        else:
            new._error = None
        # Deal with Sky
        if self._sky is not None:
            new._sky = self._sky[item]
        else:
            new._sky = None
        # Gain and rdnoise
        if self._gain is not None:
            new._gain = deepcopy(self._gain)
        if self._rdnoise is not None:
            new._rdnoise = deepcopy(self._rdnoise)
            
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
    def error(self):
        """ Return the uncertainty."""
        # if error not input
        # estimate error from image plus gain
        if self._error is None:
            self._error = poissonnoise(self.data,self.gain,self.rdnoise)
        return self._error
    
    @property
    def sky(self):
        """ Return the sky."""
        # estimate the sky
        if self._sky is None:
            self._sky = self._skyfunc(self)
        return self._sky

    @property
    def gain(self):
        """ Return the gain."""
        if self._gain is None:
            self._gain = getgain(self)
        return self._gain

    @property
    def rdnoise(self):
        """ Return the read noise."""
        if self._rdnoise is None:
            self._rdnoise = getrdnoise(self)
        return self._rdnoise    
    
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

    def native(self):
        """ Make sure that the arrays use native endian for sep."""

        # Deal with byte order for sep
        sys_is_le = sys.byteorder == 'little'
        native_code = sys_is_le and '<' or '>'
        # data
        if self.data.dtype.byteorder != native_code:
            self.data = self.data.byteswap(inplace=True).newbyteorder()
        # uncertainty
        if self.uncertainty is not None:
            if self.uncertainty.array.dtype.byteorder != native_code:
                self.uncertainty.array = self.uncertainty.array.byteswap(inplace=True).newbyteorder()
        # mask
        if self.mask is not None:
            if self.mask.dtype.byteorder != native_code:
                self.mask = self.mask.byteswap(inplace=True).newbyteorder()
        # sky
        if self._sky is not None:
            if self._sky.dtype.byteorder != native_code:
                self._sky = self._sky.byteswap(inplace=True).newbyteorder()            

    def copy(self):
        """
        Return a copy of the CCDData object.
        """
        return self.__class__(self, copy=True, error=self._error, gain=self._gain, rdnoise=self._rdnoise)

    
    # read/write methods? already exists
    # put data, error, mask, sky in separate extensions

    def write(self,outfile,overwrite=True):
        """ Write the image data to a file."""

        hdulist = fits.HDUList()
        # HDU0: Data and header
        hdulist.append(fits.PrimaryHDU(self.data,self.header))
        hdulist[0].header['IMAGTYPE'] = 'Prometheus'
        # HDU1: error
        hdulist.append(fits.ImageHDU(self.error))
        hdulist[1].header['BUNIT'] = 'Uncertainty'
        # HDU2: mask
        if self.mask is None:
            hdulist.append(fits.ImageHDU(self.mask))
        else:
            hdulist.append(fits.ImageHDU(self.mask.astype(int)))            
        hdulist[2].header['BUNIT'] = 'Mask'
        # HDU3: flags
        hdulist.append(fits.ImageHDU(self.flags))
        hdulist[3].header['BUNIT'] = 'Flags'
        # HDU4: sky
        hdulist.append(fits.ImageHDU(self.sky))
        hdulist[4].header['BUNIT'] = 'Sky'
        hdulist.writeto(outfile,overwrite=overwrite)
        hdulist.close()
        
    @classmethod
    def read(cls,filename):
        """ Read in an image from file."""

        hdulist = fits.open(filename)
        nhdu = len(hdulist)
        # HDU0: Data and header
        data = hdulist[0].data
        head = hdulist[0].header
        # HDU1: error
        if nhdu>1:
            error = hdulist[1].data
            ehead = hdulist[1].header
        else:
            error = None
        # HDU2: mask
        if nhdu>2:
            mask = hdulist[2].data
            mhead = hdulist[2].header
        else:
            mask = None
        # HDU3: flags
        if nhdu>3:
            flags = hdulist[3].data
            fhead = hdulist[3].header
        else:
            flags = None
        # HDU4: sky
        if nhdu>4:
            sky = hdulist[4].data
            shead = hdulist[4].header
        else:
            sky = None
        hdulist.close()
        # Make WCS, this doesn't capture the PV#_# values
        w = WCS(head)
        # Units
        unit = head.get('bunit')
        if unit is None:
            unit = 'adu'
        # make the ccddata object
        image = CCDData(data,error,mask=mask,meta=head,flags=flags,sky=sky,wcs=w,unit=unit)
        
        return image

    
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
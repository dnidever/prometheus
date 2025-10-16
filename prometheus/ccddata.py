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
from dlnpyutils import utils as dln
from . import sky as psky

if np.__version__ >= '2.0':
    NEWBYTESWAP = True
else:
    NEWBYTESWAP = False

def is_int_sequence(x):
    """Return True if x is a list, tuple, or ndarray of integers."""
    if isinstance(x, (list, tuple, np.ndarray)):
        # For numpy array, check dtype kind
        if isinstance(x, np.ndarray):
            return np.issubdtype(x.dtype, np.integer)
        # For list/tuple, check each element
        else:
            return all(isinstance(i, (int, np.integer)) for i in x)
    return False
    
def poissonnoise(data,gain=1.0,rdnoise=0.0):
    """ Generate Poisson noise model (ala DAOPHOT)."""
    # gain
    # rdnoise
    noise = np.sqrt(np.maximum(data,0)/gain + rdnoise**2)
    noise = np.maximum(noise,1)
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
                    gain = float(hgain)
                    break
    return gain
    
def getrdnoise(image):
    """ Get the read noise from the header."""

    rdnoise = 0.0  # default
    
    # check if there's a header
    if hasattr(image,'meta'):
        if image.meta is not None:
            # Try all versions of rdnoise
            for f in ['rdnoise','readnois','enoise','rdnoisea']:
                hrdnoise = image.meta.get(f)
                if hrdnoise is not None:
                    rdnoise = float(hrdnoise)
                    break
    return rdnoise

def mkbbox(data):
    """ Make BoundingBox and x,y arrays for data."""

    ndim = data.ndim
    if ndim==0:
        bbox = BoundingBox(0,0,0,0)
        x = None
        y = None         
    elif ndim==1:
        nx, = data.shape
        bbox = BoundingBox(0,nx,0,0)
        x = np.arange(bbox.xrange[0],bbox.xrange[-1])
        y = None
    elif ndim==2:
        ny,nx = data.shape
        bbox = BoundingBox(0,nx,0,ny)
        x = np.arange(bbox.xrange[0],bbox.xrange[-1])
        y = np.arange(bbox.yrange[0],bbox.yrange[-1])
    else:
        raise ValueError('3D CCDData not supported')
    return bbox,x,y


class CCDData(CCD):
    """
    
    A container for image data.  This is based on the astropy CCDdata class, but some
    functionality has been added.

    Parameters
    ----------
    data : `numpy.ndarray`-like or `NDData`-like
        The dataset.

    error : any type, optional
        Uncertainty in the dataset.  This must be a "reguar" standard deviation
        uncertainty. Defaults to ``None``.

    mask : any type, optional
        Mask for the dataset. Masks should follow the ``numpy`` convention that
        **valid** data points are marked by ``False`` and **invalid** ones with
        ``True``.
        Defaults to ``None``.

    wcs : any type, optional
        World coordinate system

    meta : `dict`-like object, optional
        Additional meta information about the dataset. If no meta is provided
        an empty `collections.OrderedDict` is created.
        Default is ``None``.

    bbox : BBox, optional
        Bounding Box of the image.

    gain : float, optional
        Gain of the image (e/ADU).  Default is 1.0. 

    rdnoise : float, optional
        Readnoise of the image.  Default is 0.0.

    sky : numpy array, optional
        The sky background array.

    copy : `bool`, optional
        Indicates whether to save the arguments as copy. ``True`` copies
        every attribute before saving it while ``False`` tries to save every
        parameter as reference.
        Note however that it is not always possible to save the input as
        reference.
        Default is ``False``.

    skyfunc : function, optional
        Function that computes the sky background of the image.

    unit : unit-like, optional
        Unit for the dataset. Strings that can be converted to a
        `~astropy.units.Unit` are allowed.
        Default is ``adu``.

    Methods
    -------
    read(fileame)
        ``Classmethod`` to create an CCDData instance based on a ``FITS`` file.
    write(filename)
        Writes the contents of the CCDData instance into a new ``FITS`` file.


    """

    def __init__(self, data, *args, error=None, mask=None, bbox=None, gain=None, rdnoise=None,
                 sky=None,copy=False, skyfunc=None, unit=None, **kwargs):
        # Make sure the original version copies all of the input data
        # otherwise bad things will happen when we convert to native byte-order

        # NDData class has these input parameters
        # data, uncertainty=None, mask=None, wcs=None, meta=None, unit=None, copy=False

        if data.ndim>2:
            raise ValueError('3D CCDData not supported')
        
        # Pull out error from arguments
        if len(args)>0:
            error = args[0]
            if len(args)==1:
                args = ()
            else:
                args = (None,*args[1:])

        # Make sure we have units
        if unit is None:
            unit = 'adu'

        # Check for non-finite values
        bad = (~np.isfinite(data))
        if error is not None:
            bad = bad | (~np.isfinite(error)) | (error <=0)
            if np.sum(bad)>0:
                error[bad] = 1e30
        if np.sum(bad)>0:
            data[bad] = 0.0            
            if mask is None:
                mask = np.array(data.shape,bool)
            mask[bad] = True    # masked means bad
            
        # Initialize with the parent...
        super().__init__(data, *args, mask=mask, copy=copy, unit=unit, **kwargs)

        # Create initial header if none input
        if self.header is None or len(self.header)==0:
            self.header = fits.PrimaryHDU(self.data).header
            # add unit, gain, rdnoise
        
        # Do some checks on error, mask and sky
        if error is not None:
            if error.shape != data.shape:
                raise ValueError('data and error arrays have different shapes')
        if mask is not None:
            if mask.shape != data.shape:
                raise ValueError('data and mask arrays have different shapes')
        if sky is not None:
            if sky.shape != data.shape:
                raise ValueError('data and sky arrays have different shapes')            
        
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

        # Make Bounding Box
        if bbox is None:
            bbox,x,y = mkbbox(self.data)
            self._bbox = bbox
            self._x = x
            self._y = y
        else:  # bbox input
            self._bbox = bbox
            self._x = None
            self._y = None
            if self.data.ndim==1:
                self._x = np.arange(bbox.xrange[0],bbox.xrange[-1])
            else:
                self._x = np.arange(bbox.xrange[0],bbox.xrange[-1])
                self._y = np.arange(bbox.yrange[0],bbox.yrange[-1])

        
        #self.native()

        # for sep we need to ensure that the data is "c-contiguous"
        # if we used a slice with no copy, then it won't be
        # check image.data.flags['C_CONTIGUOUS']
        # can make it c-contiguous by doing
        # foo = foo.copy(order='C')


        #maybe have CCDData have both a relative and absolute BoundingBox.  The relative being of the
        #last image that it was sliced from (what I'm using now), and the absolute one keeping track of
        #the position in the *original* image.

        
    def __repr__(self):
        prefix = self.__class__.__name__ + '('
        body = np.array2string(self.data, separator=', ', prefix=prefix)
        out = ''.join([prefix, body, ')']) +'\n'
        out += self.bbox.__repr__()
        return out

    def __array__(self):
        """ Return the main data array."""
        return self.data

    # Image arithmetic

    def __add__(self, value):
        newim = self.copy()
        if isinstance(value,CCDData):
            if self.shape != value.shape:
                raise ValueError('Shapes do not match')
            newim.data += value.data
            newim._error = np.sqrt(newim.error**2 + value.error**2)
            # combine masks, get the original by default
            if newim.mask is not None and value.mask is not None:
                newim.mask = np.bitwise_or.reduce((newim.mask,value.mask))
            elif newim.mask is None and value.mask is not None:
                newim.mask = value.mask.copy()
        else:
            newim.data += value
        return newim
        
    def __iadd__(self, value):
        if isinstance(value,CCDData):
            if self.shape != value.shape:
                raise ValueError('Shapes do not match')            
            self.data += value.data
        else:
            self.data += value
        return self
        
    def __radd__(self, value):
        return self + value
    
    def __sub__(self, value):
        newim = self.copy()
        if isinstance(value,CCDData):
            if self.shape != value.shape:
                raise ValueError('Shapes do not match')
            newim.data -= value.data
            newim._error = np.sqrt(newim.error**2 + value.error**2)
            # combine masks, get the original by default
            if newim.mask is not None and value.mask is not None:
                newim.mask = np.bitwise_or.reduce((newim.mask,value.mask))
            elif newim.mask is None and value.mask is not None:
                newim.mask = value.mask.copy()
        else:
            newim.data -= value
        return newim

    def __isub__(self, value):
        if isinstance(value,CCDData):
            if self.shape != value.shape:
                raise ValueError('Shapes do not match')            
            self.data -= value.data
        else:
            self.data -= value
        return self
         
    def __rsub__(self, value):
        return self - value 

    def __mul__(self, value):
        newim = self.copy()
        if isinstance(value,CCDData):
            if self.shape != value.shape:
                raise ValueError('Shapes do not match')
            newim.data *= value.data
            newim._error *= value.data
            # combine masks, get the original by default
            if newim.mask is not None and value.mask is not None:
                newim.mask = np.bitwise_or.reduce((newim.mask,value.mask))
            elif newim.mask is None and value.mask is not None:
                newim.mask = value.mask.copy()                        
        else:
            newim.data *= value
            newim._error *= value            
        return newim

    def __imul__(self, value):
        if isinstance(value,CCDData):
            if self.shape != value.shape:
                raise ValueError('Shapes do not match')            
            self.data *= value.data
            self._error *= value.data
        else:
            self.data *= value
            self._error *= value            
        return self
    
    def __rmul__(self, value):
        return self * value
               
    def __truediv__(self, value):
        newim = self.copy()
        if isinstance(value,CCDData):
            if self.shape != value.shape:
                raise ValueError('Shapes do not match')
            newim.data /= value.data
            newim._error /= value.error
            # combine masks, get the original by default
            if newim.mask is not None and value.mask is not None:
                newim.mask = np.bitwise_or.reduce((newim.mask,value.mask))
            elif newim.mask is None and value.mask is not None:
                newim.mask = value.mask.copy()
        else:
            newim.data /= value
            newim._error /= value            
        return newim

    def __itruediv__(self, value):
        if isinstance(value,CCDData):
            if self.shape != value.shape:
                raise ValueError('Shapes do not match')            
            self.data /= value.data
            self._error /= value.data
        else:
            self.data /= value
            self._error /= value            
        return self
      
    def __rtruediv__(self, value):
        return self / value

    # Comparison operations
    #  These will all return numpy boolean arrays
    
    def __lt__(self,value):
        if isinstance(value,CCDData):
            if self.shape != value.shape:
                raise ValueError('Shapes do not match')            
            return self.data < value.data
        elif isinstance(value,np.ndarray):
            if self.shape != value.shape:
                raise ValueError('Shapes do not match')            
            return self.data < value
        else:
            return self.data < value

    def __le__(self,value):
        if isinstance(value,CCDData):
            if self.shape != value.shape:
                raise ValueError('Shapes do not match')            
            return self.data <= value.data
        elif isinstance(value,np.ndarray):
            if self.shape != value.shape:
                raise ValueError('Shapes do not match')            
            return self.data <= value
        else:
            return self.data <= value    

    def __gt__(self,value):
        if isinstance(value,CCDData):
            if self.shape != value.shape:
                raise ValueError('Shapes do not match')            
            return self.data > value.data
        elif isinstance(value,np.ndarray):
            if self.shape != value.shape:
                raise ValueError('Shapes do not match')            
            return self.data > value
        else:
            return self.data > value    

    def __ge__(self,value):
        if isinstance(value,CCDData):
            if self.shape != value.shape:
                raise ValueError('Shapes do not match')            
            return self.data <= value.data
        elif isinstance(value,np.ndarray):
            if self.shape != value.shape:
                raise ValueError('Shapes do not match')            
            return self.data <= value
        else:
            return self.data <= value
    
    def __eq__(self,value):
        if isinstance(value,CCDData):
            if self.shape != value.shape:
                raise ValueError('Shapes do not match')            
            return self.data == value.data
        elif isinstance(value,np.ndarray):
            if self.shape != value.shape:
                raise ValueError('Shapes do not match')            
            return self.data == value
        else:
            return self.data == value
    
    
    # for the string representation also print out the bbox values
        
    def __getitem__(self, item):
        
        # Abort slicing if the data is a single scalar.
        if self.data.shape == ():
            raise TypeError('scalars cannot be sliced.')

        # BoundingBox input, use its slices
        if isinstance(item,BoundingBox):
            item = item.slices
        # Image input, use its bbox/slices
        if isinstance(item,CCDData):
            item = item.bbox.slices
            
        # Single slice or integer
        #   make sure we have values for each dimension        
        if self.ndim==2 and isinstance(item,tuple)==False:
            item = (item,slice(None,None,None))

        # Slicing with an array for x/y causes problems
        # with wcs and bounding box
        # NOT allowed
        if (isinstance(item,(list,tuple,np.ndarray)) and is_int_sequence(item[0])):
            raise Exception('Cannot use array subscripting on CCDData object')

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
                try:
                    new._bbox = BoundingBox(newx[0],newx[-1]+1,newy[0],newy[-1]+1)
                except:
                    print('slicing problem')
                    import pdb; pdb.set_trace()
                
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
        if hasattr(self,'_error') is False or self._error is None:
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

    @property
    def min(self):
        """ Calculate the min of the image data.  Uses only unmasked data."""
        if self.mask is not None:
            return np.min(self.data[~self.mask])
        else:
            return np.min(self.data)

    @property
    def max(self):
        """ Calculate the max of the image data.  Uses only unmasked data."""
        if self.mask is not None:
            return np.max(self.data[~self.mask])
        else:        
            return np.max(self.data)

    @property
    def mean(self):
        """ Calculate the mean of the image data.  Uses only unmasked data."""
        if self.mask is not None:
            return np.mean(self.data[~self.mask])
        else:        
            return np.mean(self.data)

    @property
    def median(self):
        """ Calculate the median of the image data.  Uses only unmasked data."""
        if self.mask is not None:
            return np.median(self.data[~self.mask])
        else:        
            return np.median(self.data)

    @property
    def std(self):
        """ Calculate the standard deviation of the image data.  Uses only unmasked data."""
        if self.mask is not None:
            return np.std(self.data[~self.mask])
        else:        
            return np.std(self.data)

    @property
    def mad(self):
        """ Calculate the MAD of the image data.  Uses only unmasked data."""
        if self.mask is not None:
            return dln.mad(self.data[~self.mask])
        else:        
            return dln.mad(self.data)

    def bin(self,binsize,tot=False):
        """ Bin the data in place."""
        if type(binsize) is int:
            binsize = [binsize,binsize]
        nbinpix = binsize[0]*binsize[1]
        # If error does NOT exist yet and not summing, then make it
        # otherwise we cannot use poisson statistics after averaging
        if (hasattr(self,'_error') is False or self._error is None) and tot==False:
            e = self.error   # calculated error behind the scenes
        self.data = dln.rebin(self.data,binsize=binsize,tot=tot)            
        if self._error is not None:
            binerror = dln.rebin(self._error**2,binsize=binsize,tot=True)
            if tot==False:
                binerror /= nbinpix**2
            binerror = np.sqrt(binerror)
            self._error = binerror
        if self.mask is not None:
            newmask = dln.rebin(self.mask.astype(int),binsize=binsize,tot=True)
            newmask = (newmask > 0.5*nbinpix)
            self.mask = newmask
        if self._sky is not None:
            self._sky = dln.rebin(self._sky,binsize=binsize,tot=tot)
        # Update header
        self.header['NAXIS1'] = self.data.shape[1]
        self.header['NAXIS2'] = self.data.shape[0]
        self.header['XBIN'] = binsize[0]
        self.header['YBIN'] = binsize[1]
        if 'CRPIX1' in self.header:
            self.header['CRPIX1'] /= binsize[0]
        if 'CRPIX2' in self.header:
            self.header['CRPIX2'] /= binsize[1]
        if 'CDELT1' in self.header:
            self.header['CDELT1'] *= binsize[0]
        if 'CDELT2' in self.header:
            self.header['CDELT2'] *= binsize[1]
        if 'CD1_1' in self.header:
            self.header['CD1_1'] *= binsize[0]
        if 'CD1_2' in self.header:
            self.header['CD1_2'] *= binsize[1]
        if 'CD2_1' in self.header:
            self.header['CD2_1'] *= binsize[0]
        if 'CD2_2' in self.header:
            self.header['CD2_2'] *= binsize[1]
            
        # Update BoundingBox
        bbox,x,y = mkbbox(self.data)
        self._bbox = bbox
        self._x = x
        self._y = y
        # Fix wcs
        #  crval  keep
        #  crpix  scale by binsize
        #  cd/cdelt  scale by binsize
        #  distortion terms,  scale by binsize with the appropriate power
        if self.wcs is not None:
            if self.wcs.wcs.ctype[0] != '':
                self.wcs.wcs.crpix /= np.array(binsize)
                if hasattr(self.wcs.wcs,'cd'):
                    self.wcs.wcs.cd[:,0] *= binsize[0]
                    self.wcs.wcs.cd[:,1] *= binsize[1]     
                else:
                    #self.wcs.wcs.pc[:,0] *= binsize[0]
                    #self.wcs.wcs.pc[:,1] *= binsize[1]                     
                    self.wcs.wcs.cdelt *= np.array(binsize)
            # something still isn't quite right
            self.wcs.array_shape = self.shape
        
    def resetbbox(self):
        """ Forgot the original coordinates in BoundingBox."""
        self.bbox.reset()
        
    def isnative(self,data):
        """ Check if data has native byte order."""
        sys_is_le = sys.byteorder == 'little'
        native_code = sys_is_le and '<' or '>'
        # = is native
        # | is for not applicable
        return (data.dtype.byteorder == native_code) or (data.dtype.byteorder=='=') or (data.dtype.byteorder=='|')

    def isccont(self,data):
        """ Check if data is c-continuous."""
        return data.flags['C_CONTIGUOUS']

    def issepready(self,data):
        """ Check if data is ready for sep (native byte order and c-continuous)."""
        return self.isnative(data) & self.isccont(data)
    
    def native(self):
        """ Make sure that the arrays use native endian for sep."""

        # Deal with byte order for sep
        sys_is_le = sys.byteorder == 'little'
        native_code = sys_is_le and '<' or '>'
        # data
        if self.data.dtype.byteorder != native_code:
            #new_arr = arr.view(arr.dtype.newbyteorder('>'))
            self.data = self.data.byteswap(inplace=True).newbyteorder()
        # error
        if self._error is not None:
            if self._error.dtype.byteorder != native_code:
                self._error = self._error.byteswap(inplace=True).newbyteorder()
        # mask
        if self.mask is not None:
            if self.mask.dtype.byteorder != native_code:
                self.mask = self.mask.byteswap(inplace=True).newbyteorder()
        # sky
        if self._sky is not None:
            if self._sky.dtype.byteorder != native_code:
                self._sky = self._sky.byteswap(inplace=True).newbyteorder()            

    @property
    def ccont(self):
        """ Return C-Continuous data for data, error, mask, sky."""

        if self.mask is not None:
            return (self.sepready(self.data), self.sepready(self.error),
                    self.sepready(self.mask), self.sepready(self.sky))
        else:
            return (self.sepready(self.data), self.sepready(self.error),
                    None, self.sepready(self.sky))
            
        #if self.data.flags['C_CONTIGUOUS']==False:
        #    data = self.data.copy(order='C')
        #else:
        #    data = self.data
        #if self.error.flags['C_CONTIGUOUS']==False:
        #    error = self.error.copy(order='C')
        #else:
        #    error = self.error
        #if self.mask is not None:
        #    if self.mask.flags['C_CONTIGUOUS']==False:
        #        mask = self.mask.copy(order='C')
        #    else:
        #        mask = self.mask
        #else:
        #    mask = self.mask
        #if self.sky.flags['C_CONTIGUOUS']==False:
        #    sky = self.sky.copy(order='C')
        #else:
        #    sky = self.sky

        return (data,error,mask,sky)

    def sepready(self,data=None):
        """ Return sep-ready data (native byte order and c-continuous)."""

        # Deal with byte order for sep
        sys_is_le = sys.byteorder == 'little'
        native_code = sys_is_le and '<' or '>'
        
        # No data nput, return a sep-ready version of the
        #   the image object
        if data is None:
            # Check all data arrays
            ready = True
            for n in ['data','_error','mask','_sky']:
                dat = getattr(self,n)
                if dat is not None:
                    ready &= self.sepready(dat)
            # Already sep-ready
            if ready:
                return self
            # Not ready, get sep-ready versions of the data
            new = self.copy()
            for n in ['data','_error','mask','_sky']:
                dat = getattr(new,n)
                if dat is not None:
                    setattr(new,n,new.sepready(dat))
            return new
        
        # Data array input
        else:
            if self.issepready(data)==False:
                new = np.ascontiguousarray(data)
                # Check if correction is needed
                if ((data.flags['C_CONTIGUOUS']==False) or
                    (data.dtype.byteorder not in (native_code, '=', '|'))):
                    # Make a C-contiguous copy and ensure native endianness
                    # Flip bytes in-place if necessary
                    if data.dtype.byteorder not in ('=', '|'):
                        new = new.byteswap().view(new.dtype.newbyteorder('='))
                #new = data.copy(order='C')
                #if self.isnative(new)==False:
                #    new = new.byteswap(inplace=True).newbyteorder()
                return new
                #return data.copy(order='C').byteswap(inplace=True).newbyteorder()
            else:
                return data
                
    @property
    def sepdata(self):
        """ Return C-Continuous and native byte order data for sep."""

        # Deal with byte order for sep
        sys_is_le = sys.byteorder == 'little'
        native_code = sys_is_le and '<' or '>'
        
        # Loop over data types
        out = []
        for name in ['data','error','mask','_sky']:
            data = getattr(self,name)
            if data is not None:
                # Check if correction is needed
                if ((data.flags['C_CONTIGUOUS']==False) or
                    (data.dtype.byteorder not in (native_code, '=', '|'))):
                    # Make a C-contiguous copy and ensure native endianness
                    out1 = np.ascontiguousarray(data)
                    # Flip bytes in-place if necessary
                    if data.dtype.byteorder not in ('=', '|'):
                        out1 = out1.byteswap().view(out1.dtype.newbyteorder('='))
                    out.append(out1)
                else:
                    out.append(data)
                
                ## if not c-contiguous or not native byte order, copy + correct
                #if data.flags['C_CONTIGUOUS']==False or data.dtype.byteorder!=native_code:
                #    out.append(data.copy(order='C').byteswap(inplace=True).newbyteorder())
                #else:
                #    out.append(data)
            else:
                out.append(data)
        return tuple(out)
            
    def copy(self):
        """
        Return a copy of the CCDData object.
        """
        return self.__class__(self, copy=True, error=self._error, gain=self._gain, rdnoise=self._rdnoise)

    def write(self,outfile,overwrite=True):
        """
        Write the image data to a file.

        Parameters
        ----------
        outfile : str
          The output filename.
        overwrite : boolean, optional
          Overwrite the file if it already exists.  Default is True.

        Example
        -------

        image.write('image.fits',overwrite=True)

        """

        hdulist = fits.HDUList()
        # HDU0: Data and header
        hdulist.append(fits.PrimaryHDU(self.data,self.header))
        hdulist[0].header['IMAGTYPE'] = 'Prometheus'
        hdulist[0].header['BUNIT'] = 'Flux'
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

    def tohdu(self):
        """
        Convert the image to an HDU so it can be written to a file.
        Note that only the image data is converted to the HDU
        (no error, mask, flags or sky).

        Returns
        -------
        hdu : fits HDU object
          The FITS HDU object.

        Example
        -------

        hdu = image.tohdu()

        """
        
        # Data and header
        if len(self.header)>0:
            hdu = fits.PrimaryHDU(self.data,self.header)
        else:
            hdu = fits.PrimaryHDU(self.data)
        hdu.header['IMAGTYPE'] = 'Prometheus'        
        return hdu
        
    @classmethod
    def read(cls,filename):
        """ Read in an image from file."""

        hdulist = fits.open(filename)
        nhdu = len(hdulist)
        # Checking if there's an image in HDU0 or HDU1
        if nhdu==1 and hdulist[0].data is None:
            print('No image in '+filename)
        if nhdu>1 and hdulist[0].data is None and hdulist[1].data is None:
            print('No image in first two extensions of '+filename)            
        # Prometheus file type
        if hdulist[0].header.get('IMAGTYPE') == 'Prometheus':
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
        # Other file type
        else:
            datanum = 0
            if hdulist[0].data is not None: datanum=1
                
            # Image in HDU0
            if hdulist[0].data is not None:
                # data
                data = hdulist[0].data
                head = hdulist[0].header
                # error
                if nhdu>1 and hdulist[1].data is not None:
                    error = hdulist[1].data
                else:
                    error = None
            else:
                # data
                data = hdulist[1].data
                head = hdulist[1].header
                # error
                if nhdu>2 and hdulist[2].data is not None:
                    error = hdulist[2].data
                else:
                    error = None
            # mask
            mask = np.zeros(data.shape,bool)
            bad = (~np.isfinite(data))
            if error is not None:
                bad = bad | (~np.isfinite(error))
            mask[bad] = True    # masked means bad
            # flags, sky
            flags = None
            sky = None
            
        hdulist.close()
        # Make WCS, this doesn't capture the PV#_# values
        w = WCS(head)
        # Units
        unit = head.get('bunit')
        if unit=='ADU':
            unit = 'adu'
        if unit is None:
            unit = 'adu'

        # Make sure the data is float
        data = data.astype(float)
            
        # make the ccddata object
        image = CCDData(data,error=error,mask=mask,meta=head,
                        flags=flags,sky=sky,wcs=w,unit=unit)
        
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

    @property
    def slices(self):
        """ Return the slices."""
        return (slice(self.iymin,self.iymax,None),
                slice(self.ixmin,self.ixmax,None))

    def reset(self):
        """ Forget the original coordinates."""
        self.ixmax -= self.ixmin
        self.iymax -= self.iymin
        self.ixmin = 0
        self.iymin = 0

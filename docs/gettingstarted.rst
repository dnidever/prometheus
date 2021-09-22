***************
Getting Started
***************



How it works
============

|Prometheus| uses stars to derive an analytical point spread function (PSF) for an image.  It then fits the height and central coordinates of all stars *simultaneously* using the PSF.


The CCDData Object
==================

This is the object type used for images and is based on the astropy CCDData type but has some additional capabilities that are needed for |Prometheus|.

.. code-block:: python

        from prometheus.ccdata import CCDData
	image = CCDData(flux,error,mask=mask)

You can also read an image directly from a file.
	
.. code-block:: python

        from prometheus.ccdata import CCDData
	image = CCDData.read('ccd1100.fits')

You can also write a CCDData object to file.  This saves the flux, error, mask and sky (computed) to file.
	
.. code-block:: python

	image.write('outfile.fits')
        
CCDData objects have a ``sky`` image property.  This can be set manually, but is calculated directly from the image using ``sep`` (Python and C library for Source Extraction and Photometry) package.

It also has a bounding box property (``bbox``) using the photutil ``BoundingBox`` class.  This can be useful when you take a slice of an image because it will remember what the original coordinates were.

	
The PSF Object
==============

There are currently three different analytical functions available in |Prometheus|: Gaussian, Moffat and Penny (Gaussian core and Lorentizn-like wings).  Eventually an empirical look-up table option will also be available.  There are separate classes for each type.  You can instantiate an object directly using these classes or use the ``psfmodel()`` utility function.

.. code-block:: python

	# Create PSF model specific class
	from prometheus.models import PSFGaussian,PSFMoffat,PSFPenny
	psf = PSFGaussian([3.5,4.5,0.5])

	# Create PSF model using psfmodel() utility function

The first PSF parameter is the ``height`` of the star.  This means that the PSF is defined with a height of 1 (not a flux of 1).  To get the flux you can use ``psf.flux()`` for height=1 or input a set of parameters ``psf.flux(pars)`` (only the height is used).

The Full Width at Half-Max (FWHM) can be easily obtained for any PSF type using ``psf.fwhm()``.
	
You can easily read and write a PSF to file.
	
.. code-block:: python

	# Write a PSF to a file
	psf.write('mypsf.psf')	

	# Now read it in
	from prometheus import models
	newpsf = models.read('mypsf.psf')

You can fit a single star directly using the PSF.

.. code-block:: python

	# Fit a star in an image
	initpars = [1000.0,5.6,15.6]  # height, xcen, ycen
	outpars,model = psf.fit(image,initpars)


You can also subtract many stars from an image using the ``sub()`` method.

.. code-block:: python

	# Subtract all stars in a catalog from an image
	subim = psf.sub(image,starcat)



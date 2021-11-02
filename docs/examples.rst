********
Examples
********


Running prometheus
==================
The simplest way to run |Prometheus| is with the command-line script ``prometheus``.  The only required argument is the name of an image FITS file.

.. code-block:: bash

    prometheus image.fits -v 1

By default, |Prometheus| doesn't print anything to the screen.  That's why we set the ``--verbose`` or ``-v`` parameter.


Running Prometheus from python
==============================
You can also run |Prometheus| directly from python.  The :mod:`~prometheus.prometheus` module as a
:func:`~prometheus.prometheus.run` function that runs through all of the steps.

.. code-block:: python

    from prometheus import prometheus
    out,model,sky,psf = prometheus.run('image.fits','gaussian',verbose=True)
    Step 1: Detection
    1102 objects detected
    Step 2: Aperture photometry
    Min/Max mag: 10.93, 17.51
    Step 3: Estimate FWHM
    FWHM =  4.15 pixels (99 sources)
    Step 3: Pick PSF stars
    62 PSF stars found
    Step 4: Construct PSF
    Final PSF: PSFGaussian([1.4803302566036458, 1.2352774728098717, 0.1822164238686703],binned=False)
    Median RMS:  0.0657
    Step 5: Get PSF photometry for all objects
    dt =  25.73023295402527

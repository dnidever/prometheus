********
Examples
********


Running prometheus
==================
The simplest way to run |Prometheus| is with the command-line script ``prometheus``.  The only required argument is the name of an image FITS file.

.. code-block:: bash

    prometheus image.fits

By default, |Prometheus| doesn't print anything to the screen.  So let's set the ``--verbose`` or ``-v`` parameter.


Running Prometheus from python
==============================
|Prometheus| has multiple modules.

    >>> import prometheus
    >>> image = prometheus.read('image.fits')

.. prometheus documentation master file, created by
   sphinx-quickstart on Tue Feb 16 13:03:42 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

**********
Prometheus
**********

Introduction
============
|Prometheus| [#f1]_ is a point spread function (PSF) photometry package that implements many of the standard practices
in Peter Stetson's DAOPHOT package.

.. toctree::
   :maxdepth: 1

   install
   

Description
===========
|Prometheus| has a number of modules to perform photometry.

|Prometheus| can be called from python directly or the command-line script `hofer` can be used.


Examples
========

.. toctree::
    :maxdepth: 1

    examples


prometheus
==========
Here are the various input arguments for command-line script `prometheus`::

  usage: prometheus [-h] [--outfile OUTFILE] [--figfile FIGFILE] [-d OUTDIR]
                    [-l] [-p] [-v] [-t]
                    files [files ...]

  Run Prometheus on an image

  positional arguments:
    files                 Images FITS files or list

  optional arguments:
    -h, --help            show this help message and exit
    --outfile OUTFILE     Output filename
    --figfile FIGFILE     Figure filename
    -d OUTDIR, --outdir OUTDIR
                          Output directory
    -l, --list            Input is a list of FITS files
    -p, --plot            Save the plots
    -v, --verbose         Verbose output
    -t, --timestamp       Add timestamp to Verbose output

.. rubric:: Footnotes

.. [#f1] In Greek mythology, `Prometheus <https://en.wikipedia.org/wiki/Prometheus>`_ is a Titan that brings fire from the heavens down to humans on earth.

#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup, find_packages

# Change name to "theprometheus" when you want to
#  load to PYPI
pypiname = 'theprometheus'

setup(name="prometheus",
      version='1.0.17',
      description='PSF Photometry Software',
      author='David Nidever',
      author_email='dnidever@montana.edu',
      url='https://github.com/dnidever/prometheus',
      packages=find_packages(exclude=["tests"]),
      scripts=['bin/prometheus'],
      requires=['numpy','astropy(>=4.0)','scipy','dlnpyutils','sep','photutils','skimage'],
#      include_package_data=True,
)

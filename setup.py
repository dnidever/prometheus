#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup, find_packages

setup(name='prometheus',
      version='1.0',
      description='PSF Photometry Software',
      author='David Nidever',
      author_email='dnidever@montana.edu',
      url='https://github.com/dnidever/prometheus',
      packages=find_packages(exclude=["tests"]),
#      scripts=['bin/dopfit','bin/dopjointfit','bin/doppler'],
      requires=['numpy','astropy(>=4.0)','scipy','dlnpyutils'],
#      include_package_data=True,
)

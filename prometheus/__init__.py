__all__ = ["models","getpsf","psfphot","synth","groupfit","starfit","leastsquares",
           "ccddata","detection","aperture","sky","prometheus","utils"]
__version__ = '1.0.0'

from .ccddata import CCDData

read = CCDData.read

# function to run all steps
#fit = 

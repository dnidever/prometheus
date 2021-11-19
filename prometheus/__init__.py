__all__ = ["models","getpsf","synth","groupfit","leastsquares","allfit",
           "ccddata","detection","aperture","sky","prometheus","utils"]
__version__ = '1.0.14'

from .ccddata import CCDData
from prometheus import prometheus as pm

read = CCDData.read
run = pm.run

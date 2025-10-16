__all__ = ["models","getpsf","synth","groupfit","leastsquares","allfit","multifit",
           "ccddata","detection","aperture","sky","prometheus","utils","galfit","forced"]
__version__ = '1.0.26'

from .ccddata import CCDData
from prometheus import prometheus as pm

read = CCDData.read
run = pm.run

""" CPU-time returning clock() function which works from within njit-ted code """
# from https://github.com/numba/numba/issues/4003
import ctypes
#import platform
#
#if platform.system() == "Windows":
#    from ctypes.util import find_msvcrt
#
#    __LIB = find_msvcrt()
#    if __LIB is None:
#        __LIB = "msvcrt.dll"
#else:
#    from ctypes.util import find_library
#
#    __LIB = find_library("c")
#
#clock = ctypes.CDLL(__LIB).clock
#clock.argtypes = []
#clock.restype = ctypes.c_int64

# https://github.com/open-atmos/PyMPDATA/blob/main/PyMPDATA/impl/clock.py
clock = ctypes.pythonapi._PyTime_GetSystemClock  # pylint:disable=protected-access
clock.argtypes = []
clock.restype = ctypes.c_int64

# returns time in nanoseconds
# divide time difference by 1e9 to get seconds

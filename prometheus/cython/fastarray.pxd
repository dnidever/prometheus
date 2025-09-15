# fastarray.pxd
# C-level declarations for FastArrayND

cimport numpy as cnp

cdef class FastArrayND:
    cdef double[::1] data         # 1D contiguous memoryview
    cdef int ndim
    cdef Py_ssize_t size
    cdef Py_ssize_t[:] shape
    cdef Py_ssize_t[:] strides

    cdef inline Py_ssize_t _get_offset(self, tuple idx)

    # C-level methods
    cdef void _binary_op(self, FastArrayND other, FastArrayND out, char op)

    cdef FastArrayND _ensure_array(self, other)

    cdef tuple shape_tuple(self)

    # -------------------
    # Elementwise ufuncs
    # -------------------
    cpdef FastArrayND exp(self)
    cpdef FastArrayND log(self)
    cpdef FastArrayND sin(self)
    cpdef FastArrayND cos(self)

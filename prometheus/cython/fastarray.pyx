# fastarray.pyx
# cython: boundscheck=False, wraparound=False
import numpy as np       # Python-level for @, np.array
cimport numpy as cnp    # C-level for cdef ndarray

import cython

from libc.math cimport exp,sqrt,atan2,pi,NAN,log,log10,abs,pow,sin,cos

cdef class FastArrayND:
    # These are defined in the pxd file
    # cdef double[::1] data         # 1D contiguous memoryview
    # #cdef int[::1] shape           # shape as 1D contiguous memoryview
    # cdef int ndim
    # cdef Py_ssize_t size
    # cdef Py_ssize_t[:] shape
    # cdef Py_ssize_t[:] strides
    
    # -------------------
    # Constructor
    # -------------------
    def __init__(self, shape):        
        cdef int i, total=1

        #if not isinstance(shape, (tuple, list)):
        #    raise TypeError("shape must be a tuple or list")

        self.ndim = len(shape)
        if self.ndim < 1 or self.ndim > 3:
            raise ValueError("Only 1D, 2D, 3D supported")

        # store shape and strides as numpy intp arrays
        self.shape = np.array(shape, dtype=np.int64)
        self.strides = np.empty_like(self.shape)

        cdef Py_ssize_t stride = 1
        for i in range(self.ndim - 1, -1, -1):
            self.strides[i] = stride
            stride *= self.shape[i]

        # copy shape into contiguous memoryview
        self.shape = np.empty(self.ndim, dtype=np.int64)
        for i in range(self.ndim):
            self.shape[i] = shape[i]
            total *= shape[i]

        self.size = total
        self.data = np.zeros(total, dtype=np.float64)

    cdef inline Py_ssize_t _get_offset(self, tuple idx):
        """Convert N-D index into flat offset."""
        if len(idx) != self.ndim:
            raise IndexError(f"Expected {self.ndim} indices, got {len(idx)}")
        cdef Py_ssize_t offset = 0
        cdef int i
        for i in range(self.ndim):
            ii = idx[i]
            if ii < 0 or ii >= self.shape[i]:
                raise IndexError("index out of range")
            offset += ii * self.strides[i]
        return offset
        
    # -------------------
    # Internal binary operation (with broadcasting)
    # -------------------
    cdef void _binary_op(self, FastArrayND other, FastArrayND out, char op):
        cdef int i
        cdef int d0=1,d1=1,d2=1
        cdef int od0=1,od1=1,od2=1
        cdef int i0,i1,i2, idx_out, idx_self, idx_other
        if self.size == other.size:
            for i in range(self.size):
                if op=='+': out.data[i] = self.data[i] + other.data[i]
                elif op=='-': out.data[i] = self.data[i] - other.data[i]
                elif op=='*': out.data[i] = self.data[i] * other.data[i]
                elif op=='/': out.data[i] = self.data[i] / other.data[i]
                elif op=='^': out.data[i] = pow(self.data[i], other.data[i])
        elif other.size == 1:
            for i in range(self.size):
                if op=='+': out.data[i] = self.data[i] + other.data[0]
                elif op=='-': out.data[i] = self.data[i] - other.data[0]
                elif op=='*': out.data[i] = self.data[i] * other.data[0]
                elif op=='/': out.data[i] = self.data[i] / other.data[0]
                elif op=='^': out.data[i] = pow(self.data[i], other.data[0])
        else:
            # Simple broadcasting for 1D-3D arrays
            if self.ndim <= 3 and other.ndim <= 3:
                d0, d1, d2 = 1, 1, 1
                od0, od1, od2 = 1, 1, 1
                if self.ndim >=1: d0=self.shape[0]
                if self.ndim >=2: d1=self.shape[1]
                if self.ndim ==3: d2=self.shape[2]
                if other.ndim >=1: od0=other.shape[0]
                if other.ndim >=2: od1=other.shape[1]
                if other.ndim ==3: od2=other.shape[2]
                for i0 in range(d0):
                    for i1 in range(d1):
                        for i2 in range(d2):
                            idx_out = i0*d1*d2 + i1*d2 + i2
                            idx_self = idx_out
                            idx_other = (i0 if od0>1 else 0)*(od1*od2) + (i1 if od1>1 else 0)*od2 + (i2 if od2>1 else 0)
                            if op=='+': out.data[idx_out] = self.data[idx_self] + other.data[idx_other]
                            elif op=='-': out.data[idx_out] = self.data[idx_self] - other.data[idx_other]
                            elif op=='*': out.data[idx_out] = self.data[idx_self] * other.data[idx_other]
                            elif op=='/': out.data[idx_out] = self.data[idx_self] / other.data[idx_other]
                            elif op=='^': out.data[idx_out] = pow(self.data[idx_self], other.data[idx_other])
            else:
                raise ValueError("Unsupported shapes for broadcasting")

                

    # # -------------------
    # # Helper to handle scalars
    # # -------------------
    # cdef FastArrayND _ensure_array(self, other):
    #     if isinstance(other, FastArrayND):
    #         return other
    #     else:
    #         cdef FastArrayND tmp = FastArrayND(np.array([1], dtype=np.int32))
    #         tmp.data[0] = cython.cast(double, other)
    #         return tmp


    # cdef FastArrayND _binary_out(self, other, char op):
    #     cdef FastArrayND o = self._ensure_array(other)
    #     cdef FastArrayND out = FastArrayND(self.shape)
    #     self._binary_op(o, out, op)
    #     return out

    # -------------------
    # Helper to handle scalar or FastArrayND
    # -------------------
    cdef FastArrayND _ensure_array(self, other):
        """
        If other is FastArrayND, return as-is.
        If other is a Python scalar, wrap as FastArrayND of size 1.
        """
        cdef FastArrayND tmp
        if isinstance(other, FastArrayND):
            return other
        else:
            tmp = FastArrayND(np.array([1], dtype=np.int32))
            tmp.data[0] = cython.cast(double, other)
            return tmp

    # -------------------
    # Operator overloading with automatic scalar broadcasting
    # -------------------
    def __add__(self, other):
        cdef FastArrayND o = self._ensure_array(other)
        cdef FastArrayND out = FastArrayND(self.shape)
        self._binary_op(o, out, '+')
        return out

    def __sub__(self, other):
        cdef FastArrayND o = self._ensure_array(other)
        cdef FastArrayND out = FastArrayND(self.shape)
        self._binary_op(o, out, '-')
        return out

    def __mul__(self, other):
        cdef FastArrayND o = self._ensure_array(other)
        cdef FastArrayND out = FastArrayND(self.shape)
        self._binary_op(o, out, '*')
        return out

    def __truediv__(self, other):
        cdef FastArrayND o = self._ensure_array(other)
        cdef FastArrayND out = FastArrayND(self.shape)
        self._binary_op(o, out, '/')
        return out

    def __pow__(self, other):
        cdef FastArrayND o = self._ensure_array(other)
        cdef FastArrayND out = FastArrayND(self.shape)
        self._binary_op(o, out, '^')
        return out

    
    # -------------------
    # In-place operators
    # -------------------
    def __iadd__(self, other):
        self._binary_op(self._ensure_array(other), self, '+')
        return self
    def __isub__(self, other):
        self._binary_op(self._ensure_array(other), self, '-')
        return self
    def __imul__(self, other):
        self._binary_op(self._ensure_array(other), self, '*')
        return self
    def __itruediv__(self, other):
        self._binary_op(self._ensure_array(other), self, '/')
        return self
    def __ipow__(self, other):
        self._binary_op(self._ensure_array(other), self, '^')
        return self


    # -------------------
    # Elementwise ufuncs
    # -------------------
    cpdef FastArrayND exp(self):
        cdef FastArrayND out = FastArrayND(self.shape)
        cdef int i
        for i in range(self.size):
            out.data[i] = exp(self.data[i])
        return out
    cpdef FastArrayND log(self):
        cdef FastArrayND out = FastArrayND(self.shape)
        cdef int i
        for i in range(self.size):
            out.data[i] = log(self.data[i])
        return out
    cpdef FastArrayND sin(self):
        cdef FastArrayND out = FastArrayND(self.shape)
        cdef int i
        for i in range(self.size):
            out.data[i] = sin(self.data[i])
        return out
    cpdef FastArrayND cos(self):
        cdef FastArrayND out = FastArrayND(self.shape)
        cdef int i
        for i in range(self.size):
            out.data[i] = cos(self.data[i])
        return out

    # -------------------
    # Indexing / slicing
    # -------------------
    def __getitem__(self, idx):
        """
        Returns a FastArrayND slice or a scalar depending on idx.
        Supports:
            a[i]         -> 1D slice or scalar
            a[i,j]       -> 2D slice or scalar
            a[i:j, k:l]  -> subarray
        """
        cdef int ndim = self.ndim
        cdef FastArrayND out
        cdef int i, j, k, l
        cdef int shape0, shape1, shape2
        cdef int start, stop, step
        cdef int i_idx, j_idx, k_idx
        cdef int i_start, i_stop, i_step
        cdef int j_start, j_stop, j_step
        cdef int k_start, k_stop, k_step
        # Handle 1D array
        if ndim == 1:
            if isinstance(idx, int):
                return self.data[idx]
            elif isinstance(idx, slice):
                start, stop, step = idx.indices(self.shape[0])
                out = FastArrayND(np.array([stop-start], dtype=np.int32))
                for i in range(stop-start):
                    out.data[i] = self.data[start + i*step]
                return out
        # Handle 2D array
        elif ndim == 2:
            if isinstance(idx, tuple):
                i_idx, j_idx = idx
                # both integers -> scalar
                if isinstance(i_idx, int) and isinstance(j_idx, int):
                    return self.data[i_idx*self.shape[1]+j_idx]
                # slice -> subarray
                i_start, i_stop, i_step = i_idx.indices(self.shape[0]) if isinstance(i_idx, slice) else (i_idx,i_idx+1,1)
                j_start, j_stop, j_step = j_idx.indices(self.shape[1]) if isinstance(j_idx, slice) else (j_idx,j_idx+1,1)
                shape0 = (i_stop - i_start + i_step -1)//i_step
                shape1 = (j_stop - j_start + j_step -1)//j_step
                out = FastArrayND(np.array([shape0, shape1], dtype=np.int32))
                for i in range(shape0):
                    for j in range(shape1):
                        out.data[i*shape1 + j] = self.data[(i_start+i*i_step)*self.shape[1] + (j_start+j*j_step)]
                return out
        # Handle 3D array
        elif ndim == 3:
            if isinstance(idx, tuple):
                i_idx, j_idx, k_idx = idx
                # all integers -> scalar
                if isinstance(i_idx,int) and isinstance(j_idx,int) and isinstance(k_idx,int):
                    return self.data[i_idx*self.shape[1]*self.shape[2] + j_idx*self.shape[2] + k_idx]
                # slices -> subarray
                i_start,i_stop,i_step = i_idx.indices(self.shape[0]) if isinstance(i_idx,slice) else (i_idx,i_idx+1,1)
                j_start,j_stop,j_step = j_idx.indices(self.shape[1]) if isinstance(j_idx,slice) else (j_idx,j_idx+1,1)
                k_start,k_stop,k_step = k_idx.indices(self.shape[2]) if isinstance(k_idx,slice) else (k_idx,k_idx+1,1)
                shape0 = (i_stop-i_start+i_step-1)//i_step
                shape1 = (j_stop-j_start+j_step-1)//j_step
                shape2 = (k_stop-k_start+k_step-1)//k_step
                out = FastArrayND(np.array([shape0,shape1,shape2], dtype=np.int32))
                for i in range(shape0):
                    for j in range(shape1):
                        for k in range(shape2):
                            out.data[i*shape1*shape2 + j*shape2 + k] = \
                                self.data[(i_start+i*i_step)*self.shape[1]*self.shape[2] + 
                                          (j_start+j*j_step)*self.shape[2] + 
                                          (k_start+k*k_step)]
                return out
        raise IndexError("Unsupported indexing")

    # def __setitem__(self, Py_ssize_t i, double value):
    #     if i < 0 or i >= self.size:
    #         raise IndexError("index out of range")
    #     self.data[i] = value


    def __setitem__(self, idx, value):
        cdef double val = float(value)
        if self.ndim == 1:
            if not isinstance(idx, (int, np.integer)):
                raise IndexError("Expected integer index for 1D array")
            if idx < 0 or idx >= self.shape[0]:
                raise IndexError("index out of range")
            self.data[idx] = val
        elif isinstance(idx, tuple):
            self.data[self._get_offset(idx)] = val
        else:
            raise IndexError("Invalid index type")

    
    # def __setitem__(self, idx, double value):
    #     if self.ndim == 1:
    #         if not isinstance(idx, (int, np.integer)):
    #             raise IndexError("Expected integer index for 1D array")
    #         if idx < 0 or idx >= self.shape[0]:
    #             raise IndexError("index out of range")
    #         self.data[idx] = value
    #     elif isinstance(idx, tuple):
    #         self.data[self._get_offset(idx)] = value
    #     else:
    #         raise IndexError("Invalid index type")
        
    # # -------------------
    # # Matmul for 2D arrays
    # # -------------------
    # def matmul(self, FastArrayND other):
    #     if self.ndim != 2 or other.ndim != 2:
    #         raise ValueError("matmul only supports 2D arrays")
    #     cdef int m = self.shape[0]
    #     cdef int k1 = self.shape[1]
    #     cdef int k2 = other.shape[0]
    #     cdef int n = other.shape[1]
    #     if k1 != k2:
    #         raise ValueError("Incompatible shapes for matrix multiply")

    #     # Wrap buffers as Cython memoryviews for speed
    #     cdef cnp.ndarray[cnp.double_t, ndim=2, mode="c"] A = \
    #         np.ndarray((m, k1), dtype=np.float64, buffer=self.data)
    #     cdef cnp.ndarray[cnp.double_t, ndim=2, mode="c"] B = \
    #         np.ndarray((k2, n), dtype=np.float64, buffer=other.data)

    #     # Python-level matmul
    #     C = A @ B  # <-- must be Python-level

    #     # Copy to FastArrayND
    #     out = FastArrayND(np.array([m, n], dtype=np.int32))
    #     out.data[:] = C.ravel()
    #     return out

    # # -------------------
    # # Optional: operator overloading for @
    # # -------------------
    # def __matmul__(self, FastArrayND other):
    #     return self.matmul(other)

    # -------------------
    # Utility: get shape as tuple
    # -------------------
    cdef tuple shape_tuple(self):
        return tuple([self.shape[i] for i in range(self.ndim)])

    def to_numpy(self):
        """
        Return a NumPy array view of the FastArrayND data.
        """
        # cast data to a Cython memoryview for Python buffer interface
        cdef double[:] mv = self.data[:self.size]  # memoryview over contiguous buffer
        # convert to NumPy array (copies only if necessary)
        arr = np.array(mv, copy=False, dtype=np.float64)
        return arr.reshape(tuple(int(s) for s in self.shape))
    
    
    def flatten(self):
        """
        Return a new 1D FastArrayND copy of the data.
        Equivalent to numpy.flatten().
        """
        cdef FastArrayND out = FastArrayND((self.size,))
        cdef Py_ssize_t i
        for i in range(self.size):
            out.data[i] = self.data[i]
        return out

    def __repr__(self):
        """Return a string representation of the array."""
        # Convert to a NumPy array and use NumPy's printing
        return repr(self.to_numpy())

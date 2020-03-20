# cython: language_level=2
# distutils: language = c++

# Test case for defining a XLA custom call target in Cython, and registering
# it via the xla_client SWIG API.
from cpython.pycapsule cimport PyCapsule_New


cdef void multiply_add_f32(void* out_ptr, void** data_ptr) nogil:
    cdef float x = (<float*>(data_ptr[0]))[0]
    cdef float y = (<float*>(data_ptr[1]))[0]
    cdef float z = (<float*>(data_ptr[2]))[0]
    cdef float* out = <float*>(out_ptr)
    out[0] = x*y + z


cpu_custom_call_targets = {}

cdef register_custom_call_target(fn_name, void* fn):
    cdef const char* name = "xla._CUSTOM_CALL_TARGET"
    cpu_custom_call_targets[fn_name] = PyCapsule_New(fn, name, NULL)


register_custom_call_target(b"multiply_add_f32", <void*>(multiply_add_f32))

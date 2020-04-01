#include <pybind11/pybind11.h>

namespace py = pybind11;

const void multiply_add_f32(void* out_ptr, void** data_ptr) {
    float x = ((float*) data_ptr[0])[0];
    float y = ((float*) data_ptr[1])[0];
    float z = ((float*) data_ptr[2])[0];
    float* out = (float*) out_ptr;
    out[0] = x*y + z;
}

PYBIND11_MODULE(custom_call_for_test, m) {
    m.doc() = "pybind11 capsuling for registering XLA custom calls";
    m.def("return_multiply_add_f32_capsule",
            []() {
        const char* name = "xla._CUSTOM_CALL_TARGET";
        return py::capsule((void *) &multiply_add_f32, name);}, "Returns a capsule.");
}

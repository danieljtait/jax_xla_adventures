# Introduction

Similar to [the example using SWIG](https://github.com/danieljtait/jax_xla_adventures/tree/master/initial_custom_call) this demonstrates
a simple example of registering a `CustomCall` using
`pybind11`. 

The basic idea is the same, we 

1. write our C++ function definition in `custom_call_for_test.cpp`
2. bundle it up in a `PyCapsule`
3. finally create a `pybind11` module with a function that can return the capsule.

We then compile the C++ code, for example using

```shell script
% c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes` custom_call_for_test.cpp -o custom_call_for_test`python3-config --extension-suffix`
```
though see [the pybind11 docs](https://pybind11.readthedocs.io/en/stable/compiling.html#building-manually)
for any additional guidance needed here.

We can now get access to this function by importing the module
```python
>>> import custom_call_for_test
>>> custom_call_for_test.return_multiply_add_f32_capsule()
<capsule object "xla._CUSTOM_CALL_TARGET" at 0x7fcd10156930>
```
and register it in the usual way
```python
xla_client.register_cpu_custom_call_target(
    b'multiply_add_f32',
    custom_call_for_test.return_multiply_add_f32_capsule())
```
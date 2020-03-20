# Introduction

This provides a small working example of registering a 
`CustomCall` for JIT compilation of a `JAX` primitive. 

First run 

```shell script
$ python setup.py build_ext --inplace
```

to generate the `Cython` code. Then

``` shell script
$ python test.py
```

should run without error. However,

```shell script
$ python multiply_add.py 
```

with fail. Uncomment the lines
```python
# for name, fn in custom_call_for_test.cpu_custom_call_targets.items():
#    xla_client.register_cpu_custom_call_target(name, fn)
```
to register the `multipl_add_f32` C++ function, and the code should
now successfully run! 

Read [this blog post]() for more explanation of what is going on.
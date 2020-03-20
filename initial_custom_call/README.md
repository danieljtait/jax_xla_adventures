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

will fail. Uncomment the lines
```python
# for name, fn in custom_call_for_test.cpu_custom_call_targets.items():
#    xla_client.register_cpu_custom_call_target(name, fn)
```
to register the `multipl_add_f32` C++ function, and the code should
now successfully run!

## References 

This code is drawn together from several sources
* For explanation of JAX primitives see [the JAX docs](https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html)
* The `.pyx` and implementation of a CustomCall was found in the
test suite of [Tensorflow xla compiler](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/xla),
in particular `tensorflow/compiler/xla/python/custom_call_for_test.pyx`
* Much less helpful was the [XLA CustomCall documentation](https://www.tensorflow.org/xla/custom_call) :( 
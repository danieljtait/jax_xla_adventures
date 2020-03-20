from jax import lax
from jax import abstract_arrays, core, xla, api
import numpy as onp
import jax.numpy as jnp
import custom_call_for_test
from jaxlib import xla_client
"""
See https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html
for an explanation on most of the primatives
"""
multiply_add_p = core.Primitive("multiply_add")  # Create the primitive


# register the function -- uncomment these lines for this to run successfully
# for name, fn in custom_call_for_test.cpu_custom_call_targets.items():
#    xla_client.register_cpu_custom_call_target(name, fn)


def multiply_add_prim(x, y, z):
    """The JAX-traceable way to use the JAX primitive.

    Note that the traced arguments must be passed as positional arguments
    to `bind`.
    """
    return multiply_add_p.bind(x, y, z)


def multiply_add_impl(x, y, z):
    """Concrete implementation of the primitive.

    This function does not need to be JAX traceable.
    Args:
    x, y, z: the concrete arguments of the primitive. Will only be caled with
      concrete values.
    Returns:
    the concrete result of the primitive.
    """
    # Note that we can use the original numpy, which is not JAX traceable
    return onp.add(onp.multiply(x, y), z)


def multiply_add_abstract_eval(xs, ys, zs):
    """Abstract evaluation of the primitive.

    This function does not need to be JAX traceable. It will be invoked with
    abstractions of the actual arguments.
    Args:
    xs, ys, zs: abstractions of the arguments.
    Result:
    a ShapedArray for the result of the primitive.
    """
    assert xs.shape == ys.shape
    assert xs.shape == zs.shape
    return abstract_arrays.ShapedArray(xs.shape, xs.dtype)


def multiply_add_xla_translation(c, xc, yc, zc):
    """The compilation to XLA of the primitive.

    Given an XlaBuilder and XlaOps for each argument, return the XlaOp for the
    result of the function.

    Does not need to be a JAX-traceable function.
    """
    return c.CustomCall(b'multiply_add_f32',
                        operands=(xc, yc, zc),
                        shape_with_layout=xla_client.Shape.array_shape(jnp.dtype(jnp.float32), (), ()),
                        operand_shapes_with_layout=(
                            xla_client.Shape.array_shape(jnp.dtype(jnp.float32), (), ()),
                            xla_client.Shape.array_shape(jnp.dtype(jnp.float32), (), ()),
                            xla_client.Shape.array_shape(jnp.dtype(jnp.float32), (), ())
                        ))


# Define the concrete implementation
multiply_add_p.def_impl(multiply_add_impl)
# Define the abstract evaluation
multiply_add_p.def_abstract_eval(multiply_add_abstract_eval)
# Register XLA compilation rule
xla.backend_specific_translations['cpu'][multiply_add_p] = multiply_add_xla_translation

x, y, z = (1., 2., 3.)

jit_res = api.jit(multiply_add_prim)(x, y, z)
res = multiply_add_prim(x, y, z)

print("Result {} Expected {}".format(jit_res, res))

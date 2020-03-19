import jax.numpy as jnp
import numpy as onp
import jax
import custom_call_for_test
from jax import core, xla, abstract_arrays
from jaxlib import xla_client

# register the function
for name, fn in custom_call_for_test.cpu_custom_call_targets.items():
    xla_client.register_cpu_custom_call_target(name, fn)


# create a primitive for our custom function
subtract_f32_p = core.Primitive("subtract_f32_p")  # Create the primitive


def subtract_f32_prim(x, y):
    """The JAX-traceable way to use the JAX primitive.

    Note that the traced arguments must be passed as positional arguments
    to `bind`.
    """
    return subtract_f32_p.bind(x, y)


def subtract_f32_impl(x, y):
    """Concrete implementation of the primitive.

    This function does not need to be JAX traceable.
    Args:
    x, y: the concrete arguments of the primitive. Will only be caled with
      concrete values.
    Returns:
    the concrete result of the primitive.
    """
    # Note that we can use the original numpy, which is not JAX traceable
    return onp.subtract(x, y)


# Now we register the primal implementation with JAX
subtract_f32_p.def_impl(subtract_f32_impl)


def subtract_f32_abstract_eval(xs, ys):
    """Abstract evaluation of the primitive.

    This function does not need to be JAX traceable. It will be invoked with
    abstractions of the actual arguments.
    Args:
    xs, ys: abstractions of the arguments.
    Result:
    a ShapedArray for the result of the primitive.
    """
    assert xs.shape == ys.shape
    return abstract_arrays.ShapedArray(xs.shape, xs.dtype)


# Now we register the abstract evaluation with JAX
subtract_f32_p.def_abstract_eval(subtract_f32_abstract_eval)


def subtract_float_32_xla_translation(c, xc, yc):
    """The compilation to XLA of the primitive.

    Given an XlaBuilder and XlaOps for each argument, return the XlaOp for the
    result of the function.

    Does not need to be a JAX-traceable function.
    """
    return c.CustomCall(b'test_subtract_f32',
                        operands=(xc, yc),
                        shape_with_layout=xla_client.Shape.array_shape(jnp.dtype(jnp.float32), (), ()),
                        operand_shapes_with_layout=(
                            xla_client.Shape.array_shape(jnp.dtype(jnp.float32), (), ()),
                            xla_client.Shape.array_shape(jnp.dtype(jnp.float32), (), ())
                        ))


# Now we register the XLA compilation rule with JAX
xla.backend_specific_translations['cpu'][subtract_f32_p] = subtract_float_32_xla_translation

# now run it!
x = 1.25
y = 0.5
result = jax.api.jit(subtract_f32_prim)(x, y)

print("Result {} Expected {}".format(result, x-y))
import jax.numpy as jnp
from jaxlib import xla_client
import custom_call_for_test

# register the function
for name, fn in custom_call_for_test.cpu_custom_call_targets.items():
    xla_client.register_cpu_custom_call_target(name, fn)

c = xla_client.ComputationBuilder('comp_builder')

c.CustomCall(b'multiply_add_f32',
             operands=(c.ConstantF32Scalar(2.), c.ConstantF32Scalar(0.5), c.ConstantF32Scalar(2.5)),
             shape_with_layout=xla_client.Shape.array_shape(jnp.dtype(jnp.float32), (), ()),
             operand_shapes_with_layout=(
                 xla_client.Shape.array_shape(jnp.dtype(jnp.float32), (), ()),
                 xla_client.Shape.array_shape(jnp.dtype(jnp.float32), (), ()),
                 xla_client.Shape.array_shape(jnp.dtype(jnp.float32), (), ())
             ))

compiled_c = c.Build().Compile()
result = xla_client.execute_with_python_values(compiled_c, ())
print("Result: {} Expected: {}".format(result, 3.5))
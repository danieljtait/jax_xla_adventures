import custom_call_for_test
import jax.numpy as jnp
from jaxlib import xla_client

xla_client.register_cpu_custom_call_target(
    b'multiply_add_f32',
    custom_call_for_test.return_multiply_add_f32_capsule())

c = xla_client.ComputationBuilder('comp_builder')

x, y, z = (0.6, 5., 0.14)

c.CustomCallWithLayout(b'multiply_add_f32',
             operands=(c.ConstantF32Scalar(x), c.ConstantF32Scalar(y), c.ConstantF32Scalar(z)),
             shape_with_layout=xla_client.Shape.array_shape(jnp.dtype(jnp.float32), (), ()),
             operand_shapes_with_layout=(
                 xla_client.Shape.array_shape(jnp.dtype(jnp.float32), (), ()),
                 xla_client.Shape.array_shape(jnp.dtype(jnp.float32), (), ()),
                 xla_client.Shape.array_shape(jnp.dtype(jnp.float32), (), ())
             ))

compiled_c = c.Build().Compile()
result = xla_client.execute_with_python_values(compiled_c, ())
print("Result: {} Expected: {}".format(result, x*y + z))

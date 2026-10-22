import jax.numpy as jnp
from ajx.sensors.base import Sensor
from ajx.definitions import ParameterNode
from ajx.constraints.base import is_wrapped_degree
from flax import struct
import jax
from typing import Tuple


@struct.dataclass
class OffsetParameters(ParameterNode):
    names: Tuple[str] = struct.field(pytree_node=False)
    offset: jax.Array
    scale: jax.Array


class DegreeEncoder(Sensor):
    def __init__(self, name: str, constraint, degree: int):
        """
        Measures a degree of a constraint, such as the hinge degree 5 or the linear
        degree 0, as scale * degree + offset. The residual of an angle that wraps
        around, such as the hinge degree, is wrapped to [-pi, pi).
        """
        self.name = name
        self.constraint = constraint
        self.degree = degree
        self.is_wrapped = is_wrapped_degree(constraint.constraint_residual, degree)

        observable_name = "theta" if self.is_wrapped else "x"
        self.observable_names = [observable_name]
        self.residual_names = [observable_name]

    def observe(self, state, qdot_next, param):
        value = self.constraint.get_degree(state, param, self.degree)
        idx = param.sparse_param.offset_param.names.index(self.name)
        offset = param.sparse_param.offset_param.offset[idx]
        scale = param.sparse_param.offset_param.scale[idx]
        return jnp.stack([scale * value + offset])

    def residual(self, target, prediction):
        delta = target - prediction
        if self.is_wrapped:
            delta = (delta + jnp.pi) % (2 * jnp.pi) - jnp.pi
        return delta

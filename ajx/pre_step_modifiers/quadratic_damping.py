import jax.numpy as jnp
from flax import struct
from ajx.pre_step_modifiers.base import PreStepModifier
from ajx.tree_util import ParameterNode


@struct.dataclass
class QuadraticDampingParameters(ParameterNode):
    b: float
    c: float


class QuadraticDampingComponent(PreStepModifier):
    def __init__(self, name: str, target_constraint: str):
        self.name = name
        self.target_constraint = target_constraint
        self.overrides = f"{self.target_constraint.name}.b"

    def update_params(self, state, u, param):
        D = self.target_constraint.tangential_projection(param, state)
        tangential_velocity = sum(D[b] @ s.velocity for b, s in state.items())
        damping_param = param[self.name]

        quadratic_b = damping_param.b + damping_param.c * jnp.abs(
            tangential_velocity[0]
        )
        return {self.target_constraint.name: {"b": quadratic_b}}

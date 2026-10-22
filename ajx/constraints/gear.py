import jax
from flax import struct
from jax import jit
import jax.numpy as jnp

from ajx.definitions import State
from ajx.param import SimulationParameters
from typing import Union, Tuple
from ajx.constraints.base import Constraint, ConstraintResidual


@struct.dataclass
class GearConstraint(Constraint):
    """
    A constraint that couples the positions of two scalar bodies,
        q_a - gear_ratio * q_b = 0,
    where the gear ratio is read from the scalar constraint parameters.
    """

    name: str = struct.field(pytree_node=False)
    scalar_body_a: str = struct.field(pytree_node=False)
    scalar_body_b: str = struct.field(pytree_node=False)
    constraint_residual: ConstraintResidual = struct.field(pytree_node=False, default=0)

    @classmethod
    def get_num_bodies(cls):
        return 2

    def get_operand_sizes():
        return (1, 1)

    @classmethod
    def get_constrained_degrees(cls):
        return 1

    def get_parameter_group_names():
        return ("scalar_constraint_param",)

    def get_body_group_names():
        return ("scalar_body_param", "scalar_body_param")

    def get_gvel_names():
        return ("scalar", "scalar")

    @property
    def bodies(self):
        return (self.scalar_body_a, self.scalar_body_b)

    @property
    def names(self):
        return (self.name,)

    def get_multiplier_names(self) -> Tuple[str]:
        return ("gear",)

    def compute_offset(
        default_offset: jax.Array,
        target: jax.Array,
        constraint_residual: ConstraintResidual,
    ):
        return default_offset - target

    @jit
    def func(
        param: SimulationParameters,
        state: State,
        body_ids: Tuple[Union[int, jax.Array]],
        constraint_ids: Tuple[Union[int, jax.Array]],
        constraint_residual: Union[ConstraintResidual, jax.Array],
    ) -> jax.Array:
        scalar_body_a_conf = state.conf.scalar[body_ids[0]]
        scalar_body_b_conf = state.conf.scalar[body_ids[1]]
        gear_ratio = param.scalar_constraint_param.gear_ratio[constraint_ids[0]]
        return jnp.stack([scalar_body_a_conf - gear_ratio * scalar_body_b_conf])

    @jit
    def jacobian(
        param: SimulationParameters,
        state: State,
        body_ids: Tuple[Union[int, jax.Array]],
        constraint_ids: Tuple[Union[int, jax.Array]],
        constraint_residual: Union[ConstraintResidual, jax.Array],
    ) -> jax.Array:
        gear_ratio = param.scalar_constraint_param.gear_ratio[constraint_ids[0]]
        return jnp.stack([1.0, -gear_ratio])

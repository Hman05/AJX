from abc import ABC, abstractmethod

import jax
import ajx.math as math
from flax import struct
from jax import jit
import jax.numpy as jnp

from enum import Enum
from ajx.definitions import Transform, State
from ajx.param import SimulationParameters
from typing import Union, Tuple
from functools import partial
from ajx.constraints.base import Constraint, ConstraintResidual, get_frame_transform


@struct.dataclass
class GearConstraint(Constraint):
    """
    A constraint that couples two "tangential dofs"

    """

    name: str
    scalar_body_a: str
    scalar_body_b: str
    gear_ratio: float  # TODO: Move to parameter
    dof: int = 1
    constraint_residual: ConstraintResidual = 0

    # def __init__(self, name: str, body: str, constraint_residual: ConstraintResidual):
    #     self.name = name
    #     self.body = body
    #     self.constraint_residual = constraint_residual

    def get_num_bodies():
        return 2

    @property
    def bodies(self):
        return (self.scalar_body_a, self.scalar_body_b)

    def get_multiplier_names(self) -> Tuple[str]:
        return "gear"

    @partial(jit, static_argnums=0)
    def func(
        self,
        state: State,
        param: SimulationParameters,
    ):
        scalar_body_a = param.scalar_body_param.names.index(self.scalar_body_a)
        scalar_body_b = param.scalar_body_param.names.index(self.scalar_body_b)
        constraint_id = param.constraint_param.names.index(self.name)
        return GearConstraint.func(
            param,
            state,
            (scalar_body_a, scalar_body_b),
            constraint_id,
            self.constraint_residual,
        )

    @jit
    def func(
        param: SimulationParameters,
        state: State,
        body_ids: Tuple[Union[int, jax.Array]],
        constraint_id: Union[int, jax.Array],
        constraint_residual: Union[ConstraintResidual, jax.Array],
    ) -> jax.Array:
        """
        C
        """
        scalar_body_a_id = body_ids[0]
        scalar_body_b_id = body_ids[1]
        scalar_body_a_conf = state.conf.scalar[scalar_body_a_id]
        scalar_body_b_conf = state.conf.scalar[scalar_body_b_id]
        gear_ratio = param.scalar_constraint_param.gear_ratio[constraint_id]
        return scalar_body_a_conf - gear_ratio * scalar_body_b_conf

    @partial(jit, static_argnums=0)
    def jacobian(
        self,
        param: SimulationParameters,
        state: State,
    ) -> jax.Array:
        scalar_body_a = param.scalar_body_param.names.index(self.scalar_body_a)
        scalar_body_b = param.scalar_body_param.names.index(self.scalar_body_b)
        constraint_id = param.scalar_constraint_param.names.index(self.name)
        return GearConstraint.jacobian(
            param,
            state,
            (scalar_body_a, scalar_body_b),
            constraint_id,
            self.constraint_residual,
        )

    @jit
    def jacobian(
        param: SimulationParameters,
        state: State,
        body_ids: Tuple[Union[int, jax.Array]],
        constraint_id: Union[int, jax.Array],
        constraint_residual: Union[ConstraintResidual, jax.Array],
    ) -> jax.Array:
        scalar_body_a_id = body_ids[0]
        scalar_body_b_id = body_ids[1]
        gear_ratio = param.scalar_constraint_param.gear_ratio[constraint_id]
        jac_gear = jnp.array([1, -gear_ratio])

        return jac_gear

    @partial(jit, static_argnums=0)
    def get_free_degrees(
        self,
        state: State,
        param: SimulationParameters,
    ) -> jax.Array:
        scalar_body_a = param.scalar_body_param.names.index(self.scalar_body_a)
        scalar_body_b = param.scalar_body_param.names.index(self.scalar_body_b)
        constraint_id = param.scalar_constraint_param.names.index(self.name)

        scalar_body_a_conf = state.conf.scalar[scalar_body_a]
        scalar_body_b_conf = state.conf.scalar[scalar_body_b]
        gear_ratio = param.scalar_constraint_param.gear_ratio[constraint_id]
        return scalar_body_a_conf - gear_ratio * scalar_body_b_conf

from abc import ABC, abstractmethod

import jax
import ajx.math as math
from jax import jit
import jax.numpy as jnp

from enum import Enum
from ajx.definitions import Transform, State
from ajx.param import SimulationParameters
from typing import Union, Tuple
from functools import partial
from ajx.constraints.base import Constraint, ConstraintType, get_frame_transform
from ajx.constraints import TwoBodyConstraint


@jit
def get_frame_transform(frame, constraint_id, body_pos, body_rotation):
    d0 = frame.position[constraint_id]
    frame_rot0 = frame.rotation[constraint_id]
    frame_rot = math.quat_mul(body_rotation, frame_rot0)
    d = math.rotate_vector(body_rotation, d0)
    u = math.rotation_matrix(frame_rot)[:, 0]
    v = math.rotation_matrix(frame_rot)[:, 1]
    w = math.rotation_matrix(frame_rot)[:, 2]

    return d, u, v, w


class TwoBodyShaftConstraint(Constraint):
    """
    A constraint that restricts the relative motion between two bodies and a shaft.

    Constraint types currently supported:
     - Hinge
     - Primsatic
    """

    name: str
    body_a: str
    body_b: str
    shaft: str
    parent_constraint: str
    constraint_type: ConstraintType
    is_attached_to_world: bool = False
    dof: int = 1

    def get_num_bodies():
        return 3

    @property
    def bodies(self):
        return (self.body_a, self.body_b, self.shaft)

    @property
    def names(self):
        return (self.name, self.parent_constraint)

    def __init__(
        self,
        name: str,
        parent_constraint: TwoBodyConstraint,
        shaft: str,
    ):
        self.name = name
        self.body_a = parent_constraint.body_a
        self.body_b = parent_constraint.body_b
        self.parent_constraint = parent_constraint.name
        self.shaft = shaft
        self.constraint_type = parent_constraint.constraint_type

    def get_multiplier_names(self) -> Tuple[str]:
        if self.constraint_type == ConstraintType.HINGE.value:
            return ("nx", "ny", "nz", "n_bend", "n_torsion", "t")
        elif self.constraint_type == ConstraintType.PRISMATIC.value:
            return ("nu", "nw", "n_bend1", "n_torsion", "n_bend2", "t")
        return ()

    @partial(jit, static_argnums=0)
    def func(
        self,
        param: SimulationParameters,
        state: State,
    ):
        body_b_id = param.rigid_body_param.names.index(self.body_b)
        body_a_id = param.rigid_body_param.names.index(self.body_a)
        shaft_id = param.rigid_body_param.names.index(self.shaft)
        constraint_id = param.scalar_constraint_param.names.index(self.name)
        parent_constraint_id = param.constraint_param.names.index(
            self.parent_constraint
        )
        return TwoBodyShaftConstraint.func(
            param,
            state,
            (body_a_id, body_b_id, shaft_id),
            (constraint_id, parent_constraint_id),
            self.constraint_type,
        )

    @jit
    def func(
        param: SimulationParameters,
        state: State,
        body_ids: Tuple[Union[int, jax.Array]],
        constraint_ids: Tuple[Union[int, jax.Array]],
        constraint_type: Union[ConstraintType, jax.Array],
    ) -> jax.Array:
        """
        C
        """
        constraint_id = constraint_ids[0]
        parent_constraint_id = constraint_ids[1]
        body_a_id, body_b_id, shaft_id = body_ids
        body_b_pos = state.conf.pos[body_b_id]
        body_b_rot = state.conf.rot[body_b_id]
        body_a_pos = state.conf.pos[body_a_id]
        body_a_rot = state.conf.rot[body_a_id]
        shaft_conf = state.conf.scalar[shaft_id]

        d_a, u_a, v_a, w_a = get_frame_transform(
            param.constraint_param.frame_a, parent_constraint_id, body_a_pos, body_a_rot
        )

        d_b, u_b, v_b, w_b = get_frame_transform(
            param.constraint_param.frame_b, parent_constraint_id, body_b_pos, body_b_rot
        )

        r_a = body_a_pos + d_a
        r_b = body_b_pos + d_b

        spherical = r_a - r_b
        dot1_1 = jnp.dot(u_a, v_b)
        dot1_2 = jnp.dot(u_a, w_b)
        dot1_3 = jnp.dot(w_a, v_b)
        dot2_1 = jnp.dot(u_a, r_a - r_b)
        dot2_2 = jnp.dot(w_a, r_a - r_b)

        # TODO: Copy and paste code. Computations are done twice...
        frame_a_pos0 = param.constraint_param.frame_a.position[parent_constraint_id]
        frame_a_rot0 = param.constraint_param.frame_a.rotation[parent_constraint_id]
        frame_a_rot = math.quat_mul(body_a_rot, frame_a_rot0)
        d_a = math.rotate_vector(body_a_rot, frame_a_pos0)

        frame_b_pos0 = param.constraint_param.frame_b.position[parent_constraint_id]
        frame_b_rot0 = param.constraint_param.frame_b.rotation[parent_constraint_id]
        frame_b_rot = math.quat_mul(body_b_rot, frame_b_rot0)
        d_b = math.rotate_vector(body_b_rot, frame_b_pos0)

        # For prismatic
        r_a = body_a_pos + d_a
        r_b = body_b_pos + d_b
        r_delta = r_a - r_b
        v_a = math.rotation_matrix(frame_a_rot)[:, 1]
        gear_ratio = param.scalar_constraint_param.gear_ratio[constraint_id]
        free_prismatic = jnp.dot(v_a, r_delta) - gear_ratio * shaft_conf

        # For hinge
        frame_a_rot_inv = math.conjugate(frame_a_rot)
        delta_rotation = math.quat_mul(frame_a_rot_inv, frame_b_rot)
        axis_angle = math.to_rotation_vector(delta_rotation)
        axis = jnp.array([1.0, 0.0, 0.0])
        free_hinge = -jnp.dot(axis_angle, axis) - gear_ratio * shaft_conf

        hinge_constraint = jnp.block([free_hinge[None]]) * (
            constraint_type == ConstraintType.HINGE.value
        )
        prismatic_constraint = jnp.block([free_prismatic[None]]) * (
            constraint_type == ConstraintType.PRISMATIC.value
        )
        return hinge_constraint + prismatic_constraint

    @jit
    def jacobian(
        param: SimulationParameters,
        state: State,
        body_ids: Tuple[Union[int, jax.Array]],
        constraint_ids: Tuple[Union[int, jax.Array]],
        constraint_type: Union[ConstraintType, jax.Array],
    ) -> jax.Array:
        constraint_id = constraint_ids[0]
        parent_constraint_id = constraint_ids[1]
        body_a_id, body_b_id, shaft_id = body_ids
        body_b_pos = state.conf.pos[body_b_id]
        body_b_rot = state.conf.rot[body_b_id]
        body_a_pos = state.conf.pos[body_a_id]
        body_a_rot = state.conf.rot[body_a_id]
        gear_ratio = param.scalar_constraint_param.gear_ratio[constraint_id]

        d_a, u_a, v_a, w_a = get_frame_transform(
            param.constraint_param.frame_a, parent_constraint_id, body_a_pos, body_a_rot
        )
        d_b, u_b, v_b, w_b = get_frame_transform(
            param.constraint_param.frame_b, parent_constraint_id, body_b_pos, body_b_rot
        )

        r_a = body_a_pos + d_a
        r_b = body_b_pos + d_b

        dot2_3_a = jnp.block([v_a[None], jnp.cross(v_a, r_a - r_b)])
        dot2_3_b = jnp.block([-v_a[None], jnp.zeros([1, 3])])
        u_a_tangent_a = jnp.block([jnp.zeros([1, 3]), u_a])
        u_a_tangent_b = jnp.block([jnp.zeros([1, 3]), -u_a])

        jac_hinge = jnp.concatenate(
            [
                jnp.concatenate([u_a_tangent_a]),
                jnp.concatenate([u_a_tangent_b]),
                jnp.array([-gear_ratio]),
            ],
            axis=None,
        ) * (constraint_type == 0)
        jac_prismatic = jnp.concatenate(
            [
                jnp.concatenate([dot2_3_a]),
                jnp.concatenate([dot2_3_b]),
                jnp.array([-gear_ratio]),
            ],
            axis=None,
        ) * (constraint_type == 1)

        return jac_hinge + jac_prismatic

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
from ajx.constraints.base import Constraint, ConstraintType, get_frame_transform


@struct.dataclass
class OneBodyConstraint(Constraint):
    """
    A constraint that restricts the relative motion between two bodies.

    Constraint types currently supported:
     - Hinge
     - Primsatic
    """

    name: str
    body: str
    constraint_type: ConstraintType

    @property
    def parent_constraint(self):
        return self.name

    def get_num_bodies():
        return 1

    @property
    def is_attached_to_world(self):
        return True

    @property
    def dof(self):
        return 6

    @property
    def bodies(self):
        return (self.body,)

    @property
    def names(self):
        return (self.name,)

    # def __init__(self, name: str, body: str, constraint_type: ConstraintType):
    #     self.name = name
    #     self.body = body
    #     self.constraint_type = constraint_type

    def get_multiplier_names(self) -> Tuple[str]:
        if self.constraint_type == ConstraintType.HINGE.value:
            return ("nx", "ny", "nz", "n_bend", "n_torsion", "t")
        elif self.constraint_type == ConstraintType.PRISMATIC.value:
            return ("nu", "nw", "n_bend1", "n_torsion", "n_bend2", "t")
        return ()

    @partial(jit, static_argnums=0)
    def func(
        self,
        state: State,
        param: SimulationParameters,
    ):
        body_id = param.rigid_body_param.names.index(self.body)
        constraint_id = param.constraint_param.names.index(self.name)
        return OneBodyConstraint.func(
            param, state, (body_id,), constraint_id, self.constraint_type
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
        body_id = body_ids[0]
        constraint_id = constraint_ids[0]
        body_b_pos = state.conf.pos[body_id]
        body_b_rot = state.conf.rot[body_id]
        body_a_pos = jnp.array([0.0, 0.0, 0.0])
        body_a_rot = jnp.array([1.0, 0.0, 0.0, 0.0])

        d_a, u_a, v_a, w_a = get_frame_transform(
            param.constraint_param.frame_a, constraint_id, body_a_pos, body_a_rot
        )

        d_b, u_b, v_b, w_b = get_frame_transform(
            param.constraint_param.frame_b, constraint_id, body_b_pos, body_b_rot
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
        frame_a_pos0 = param.constraint_param.frame_a.position[constraint_id]
        frame_a_rot0 = param.constraint_param.frame_a.rotation[constraint_id]
        frame_a_rot = math.quat_mul(body_a_rot, frame_a_rot0)
        d_a = math.rotate_vector(body_a_rot, frame_a_pos0)

        frame_b_pos0 = param.constraint_param.frame_b.position[constraint_id]
        frame_b_rot0 = param.constraint_param.frame_b.rotation[constraint_id]
        frame_b_rot = math.quat_mul(body_b_rot, frame_b_rot0)
        d_b = math.rotate_vector(body_b_rot, frame_b_pos0)

        # For prismatic
        r_a = body_a_pos + d_a
        r_b = body_b_pos + d_b
        r_delta = r_a - r_b
        v_a = math.rotation_matrix(frame_a_rot)[:, 1]
        free_prismatic = jnp.dot(v_a, r_delta)

        # For hinge
        frame_a_rot_inv = math.conjugate(frame_a_rot)
        delta_rotation = math.quat_mul(frame_a_rot_inv, frame_b_rot)
        axis_angle = math.to_rotation_vector(delta_rotation)
        axis = jnp.array([1.0, 0.0, 0.0])
        free_hinge = -jnp.dot(axis_angle, axis)

        hinge_constraint = jnp.block(
            [spherical, dot1_1[None], dot1_2[None], free_hinge[None]]
        ) * (constraint_type == ConstraintType.HINGE.value)
        prismatic_constraint = jnp.block(
            [dot2_1, dot2_2, dot1_1, dot1_2, dot1_3, free_prismatic[None]]
        ) * (constraint_type == ConstraintType.PRISMATIC.value)
        return hinge_constraint + prismatic_constraint

    @partial(jit, static_argnums=0)
    def jacobian(
        self,
        param: SimulationParameters,
        state: State,
    ) -> jax.Array:
        body_id = param.rigid_body_param.names.index(self.body)
        constraint_id = param.constraint_param.names.index(self.name)
        return OneBodyConstraint.jacobian(
            param, state, (body_id,), constraint_id, self.constraint_type
        )

    @jit
    def jacobian(
        param: SimulationParameters,
        state: State,
        body_ids: Tuple[Union[int, jax.Array]],
        constraint_ids: Tuple[Union[int, jax.Array]],
        constraint_type: Union[ConstraintType, jax.Array],
    ) -> jax.Array:
        body_id = body_ids[0]
        constraint_id = constraint_ids[0]
        body_b_pos = state.conf.pos[body_id]
        body_b_rot = state.conf.rot[body_id]
        body_a_pos = jnp.array([0.0, 0.0, 0.0])
        body_a_rot = jnp.array([1.0, 0.0, 0.0, 0.0])

        d_a, u_a, v_a, w_a = get_frame_transform(
            param.constraint_param.frame_a, constraint_id, body_a_pos, body_a_rot
        )
        d_b, u_b, v_b, w_b = get_frame_transform(
            param.constraint_param.frame_b, constraint_id, body_b_pos, body_b_rot
        )

        spherical_b = jnp.block([-jnp.eye(3), math.skew(d_b)])
        dot1_1_b = jnp.block([jnp.zeros([1, 3]), -jnp.cross(u_a, v_b)])
        dot1_2_b = jnp.block([jnp.zeros([1, 3]), -jnp.cross(u_a, w_b)])
        dot1_3_b = jnp.block([jnp.zeros([1, 3]), -jnp.cross(w_a, v_b)])

        dot2_1_b = jnp.block([-u_a[None], jnp.zeros([1, 3])])
        dot2_2_b = jnp.block([-w_a[None], jnp.zeros([1, 3])])
        dot2_3_b = jnp.block([-v_a[None], jnp.zeros([1, 3])])
        u_a_tangent_b = jnp.block([jnp.zeros([1, 3]), -u_a])

        jac_hinge = jnp.concatenate(
            [spherical_b, dot1_1_b, dot1_2_b, u_a_tangent_b], axis=None
        ) * (constraint_type == 0)
        jac_prismatic = jnp.concatenate(
            [dot2_1_b, dot2_2_b, dot1_1_b, dot1_2_b, dot1_3_b, dot2_3_b], axis=None
        ) * (constraint_type == 1)

        return jac_hinge + jac_prismatic

    @partial(jit, static_argnums=0)
    def get_free_degrees(
        self,
        state: State,
        param: SimulationParameters,
    ) -> jax.Array:
        body_b_id = param.rigid_body_param.names.index(self.body)
        constraint_id = param.constraint_param.names.index(self.name)

        body_b_pos = state.conf.pos[body_b_id]
        body_b_rot = state.conf.rot[body_b_id]
        body_a_pos = jnp.array([0.0, 0.0, 0.0])
        body_a_rot = jnp.array([1.0, 0.0, 0.0, 0.0])

        frame_a_pos0 = param.constraint_param.frame_a.position[constraint_id]
        frame_a_rot0 = param.constraint_param.frame_a.rotation[constraint_id]
        frame_a_rot = math.quat_mul(body_a_rot, frame_a_rot0)
        d_a = math.rotate_vector(body_a_rot, frame_a_pos0)

        frame_b_pos0 = param.constraint_param.frame_b.position[constraint_id]
        frame_b_rot0 = param.constraint_param.frame_b.rotation[constraint_id]
        frame_b_rot = math.quat_mul(body_b_rot, frame_b_rot0)
        d_b = math.rotate_vector(body_b_rot, frame_b_pos0)

        # For prismatic
        r_a = body_a_pos + d_a
        r_b = body_b_pos + d_b
        r_delta = r_a - r_b
        v_a = math.rotation_matrix(frame_a_rot)[:, 1]
        x = jnp.dot(v_a, r_delta)
        free_prismatic = x * (self.constraint_type == 1)

        # For hinge
        frame_a_rot_inv = math.conjugate(frame_a_rot)
        delta_rotation = math.quat_mul(frame_a_rot_inv, frame_b_rot)
        axis_angle = math.to_rotation_vector(delta_rotation)
        axis = jnp.array([1.0, 0.0, 0.0])
        theta = jnp.dot(axis_angle, axis)
        free_hinge = theta * (self.constraint_type == 0)
        return free_hinge + free_prismatic

    def place_other(
        self, param: SimulationParameters, body_transform: Transform, x: float
    ) -> Transform:
        """
        Returns the configuration of the next body within a kinematic tree containing this joint.

        Parameters
        ----------
        param: Dict
            The system's parameters stored as a jax pytree with a dictionary at the top level.
        body_transform: Transform
            The configuration of the previous body/frame in the kinematic chain.
        x: float
            Value of the free degree displacement between the frames (Assumes prismatic or hinge).
        Returns:
        -------
        Transform:
            The configuration of the next body.
        """
        i = param.constraint_param.names.index(self.name)
        cp = param.constraint_param
        d0_a = cp.frame_a.position[i]
        frame_a_rot0 = cp.frame_a.rotation[i]
        d_a = math.rotate_vector(body_transform.rot, d0_a)
        frame_a_position = body_transform.pos + d_a
        frame_a_rot = math.quat_mul(body_transform.rot, frame_a_rot0)
        v_a = math.rotation_matrix(frame_a_rot)[:, 1]

        # Hinge
        frame_b_rot0 = cp.frame_b.rotation[i]
        frame_b_rot_as_seen_from_a = math.quat_from_axis_angle(
            jnp.array([1.0, 0.0, 0.0]), x
        )
        frame_b_rot = math.quat_mul(frame_a_rot, frame_b_rot_as_seen_from_a)
        hinge_body_b_rotation = math.quat_mul(frame_b_rot, math.conjugate(frame_b_rot0))
        d0_b = cp.frame_b.position[i]
        d_b = math.rotate_vector(hinge_body_b_rotation, d0_b)
        hinge_body_b_position = frame_a_position - d_b

        # Prismatic
        d_b = v_a * x
        frame_b_pos = frame_a_position - d_b
        frame_b_rot = frame_a_rot
        frame_b_rot0 = cp.frame_b.rotation[i]
        d0_b = cp.frame_b.position[i]
        frame_b_rot0_inv = math.conjugate(frame_b_rot0)
        prismatic_body_b_rotation = math.quat_mul(frame_b_rot, frame_b_rot0_inv)
        prismatic_body_b_position = frame_b_pos - d_a

        body_b_rotation = hinge_body_b_rotation * (
            self.constraint_type == ConstraintType.HINGE.value
        ) + prismatic_body_b_rotation * (
            self.constraint_type == ConstraintType.PRISMATIC.value
        )
        body_b_position = hinge_body_b_position * (
            self.constraint_type == ConstraintType.HINGE.value
        ) + prismatic_body_b_position * (
            self.constraint_type == ConstraintType.PRISMATIC.value
        )

        return Transform(body_b_position, body_b_rotation)

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
from ajx.constraints.base import (
    Constraint,
    ConstraintType,
    get_frame_transform,
    get_frame_transform_ext,
)


class TwoBodyConstraint(Constraint):
    """
    A constraint that restricts the relative motion between two bodies.

    Constraint types currently supported:
     - Hinge
     - Prismatic

    TODO: Enough to consider this as a rotational lock + translational lock?
     - Spherical to constrain translation -> Hinge
     - Dot 2 to constraint translation -> Prismatic
    """

    name: str
    body_a: str
    body_b: str
    constraint_type: ConstraintType
    is_attached_to_world: bool = False

    @classmethod
    def get_num_bodies(cls):
        return 2

    def get_operand_sizes():
        return (6, 6)

    @classmethod
    def get_constrained_degrees(cls):
        return 6

    def get_parameter_group_names():
        return ("constraint_param",)

    def get_body_group_names():
        return ("rigid_body_param", "rigid_body_param")

    def get_gvel_names():
        return ("data", "data")

    @property
    def bodies(self):
        return (self.body_a, self.body_b)

    @property
    def names(self):
        return (self.name,)

    def __init__(
        self, name: str, body_a: str, body_b: str, constraint_type: ConstraintType
    ):
        self.name = name
        self.body_a = body_a
        self.body_b = body_b
        self.constraint_type = constraint_type

    def get_multiplier_names(self) -> Tuple[str]:
        if self.constraint_type == ConstraintType.HINGE.value:
            return ("nx", "ny", "nz", "n_bend1", "n_bend2", "n_torsion")
        elif self.constraint_type == ConstraintType.PRISMATIC.value:
            return ("nu", "nw", "n_bend1", "n_torsion", "n_bend2", "nv")
        elif self.constraint_type == ConstraintType.SE3.value:
            return ("nu", "nv", "nw", "nru", "nrv", "nrw")
        return ()

    def compute_offset(
        default_offset: jax.Array, target: jax.Array, constraint_type: ConstraintType
    ):
        rotational_degrees = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=bool) * (
            constraint_type == ConstraintType.PRISMATIC.value
        )
        linear_degrees = jnp.logical_not(rotational_degrees)
        linear_degrees = jnp.logical_not(rotational_degrees)

        linear_offset = (default_offset - target) * linear_degrees
        roational_offset = (
            (default_offset - target + jnp.pi) % (2 * jnp.pi) - jnp.pi
        ) * rotational_degrees

        return linear_offset + roational_offset

    # @partial(jit, static_argnums=0)
    def funca(
        self,
        param: SimulationParameters,
        state: State,
    ):
        body_b_id = param.rigid_body_param.names.index(self.body_b)
        body_a_id = param.rigid_body_param.names.index(self.body_a)
        constraint_id = param.constraint_param.names.index(self.name)
        return TwoBodyConstraint.func(
            param, state, (body_a_id, body_b_id), (constraint_id,), self.constraint_type
        )

    # @jit
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
        body_a_id, body_b_id = body_ids
        body_b_pos = state.conf.pos[body_b_id]
        body_b_rot = state.conf.rot[body_b_id]
        body_a_pos = state.conf.pos[body_a_id]
        body_a_rot = state.conf.rot[body_a_id]

        d_a, u_a, v_a, w_a, q_a = get_frame_transform_ext(
            param.constraint_param.frame_a, constraint_id, body_a_pos, body_a_rot
        )

        d_b, u_b, v_b, w_b, q_b = get_frame_transform_ext(
            param.constraint_param.frame_b, constraint_id, body_b_pos, body_b_rot
        )

        r_a = body_a_pos + d_a
        r_b = body_b_pos + d_b

        spherical = r_a - r_b
        R_a = math.rotation_matrix(q_a).T
        co_spherical = R_a @ spherical  # math.rotate_vector(q_a, spherical)
        dot1_1 = jnp.dot(u_a, v_b)
        dot1_2 = jnp.dot(u_a, w_b)

        so3_err = math.quat_residual(q_a, q_b)

        # TODO: Copy and paste code. Computations are done twice...
        frame_a_pos0 = param.constraint_param.frame_a.position[constraint_id]
        frame_a_rot0 = param.constraint_param.frame_a.rotation[constraint_id]
        frame_a_rot = math.quat_mul(body_a_rot, frame_a_rot0)
        d_a = math.rotate_vector(body_a_rot, frame_a_pos0)

        frame_b_pos0 = param.constraint_param.frame_b.position[constraint_id]
        frame_b_rot0 = param.constraint_param.frame_b.rotation[constraint_id]
        frame_b_rot = math.quat_mul(body_b_rot, frame_b_rot0)
        d_b = math.rotate_vector(body_b_rot, frame_b_pos0)

        # For hinge
        frame_a_rot_inv = math.conjugate(frame_a_rot)
        delta_rotation = math.quat_mul(frame_a_rot_inv, frame_b_rot)
        axis_angle = math.to_rotation_vector(delta_rotation)
        axis = jnp.array([1.0, 0.0, 0.0])
        free_hinge = jnp.dot(axis_angle, axis)

        hinge_constraint = jnp.block(
            [spherical, dot1_1[None], dot1_2[None], free_hinge[None]]
        ) * (constraint_type == ConstraintType.HINGE.value)
        prismatic_constraint = jnp.block(
            [co_spherical, dot1_1[None], dot1_2[None], free_hinge[None]]
        ) * (constraint_type == ConstraintType.PRISMATIC.value)
        se3_constraint = jnp.block([co_spherical, so3_err]) * (
            constraint_type == ConstraintType.SE3.value
        )
        return hinge_constraint + prismatic_constraint + se3_constraint

    @jit
    def jacobian(
        param: SimulationParameters,
        state: State,
        body_ids: Tuple[Union[int, jax.Array]],
        constraint_ids: Tuple[Union[int, jax.Array]],
        constraint_type: Union[ConstraintType, jax.Array],
    ) -> jax.Array:
        constraint_id = constraint_ids[0]
        body_a_id, body_b_id = body_ids
        body_b_pos = state.conf.pos[body_b_id]
        body_b_rot = state.conf.rot[body_b_id]
        body_a_pos = state.conf.pos[body_a_id]
        body_a_rot = state.conf.rot[body_a_id]

        d_a, u_a, v_a, w_a, q_a = get_frame_transform_ext(
            param.constraint_param.frame_a, constraint_id, body_a_pos, body_a_rot
        )
        d_b, u_b, v_b, w_b, q_b = get_frame_transform_ext(
            param.constraint_param.frame_b, constraint_id, body_b_pos, body_b_rot
        )

        r_a = body_a_pos + d_a
        r_b = body_b_pos + d_b

        spherical_a = jnp.block([jnp.eye(3), -math.skew(d_a)])
        spherical_b = jnp.block([-jnp.eye(3), math.skew(d_b)])

        R_a = math.rotation_matrix(math.conjugate(q_a))

        co_spherical_a = jnp.block(
            [R_a, R_a @ math.skew(r_a - r_b) - R_a @ math.skew(d_a)]
        )

        co_spherical_b = jnp.block([-R_a, R_a @ math.skew(d_b)])
        dot1_1_a = jnp.block([jnp.zeros([1, 3]), jnp.cross(u_a, v_b)])
        dot1_1_b = jnp.block([jnp.zeros([1, 3]), -jnp.cross(u_a, v_b)])
        dot1_2_a = jnp.block([jnp.zeros([1, 3]), jnp.cross(u_a, w_b)])
        dot1_2_b = jnp.block([jnp.zeros([1, 3]), -jnp.cross(u_a, w_b)])
        u_a_tangent_a = -jnp.block([jnp.zeros([1, 3]), u_a])
        u_a_tangent_b = -jnp.block([jnp.zeros([1, 3]), -u_a])

        def inc_quat_residual(inc_a, inc_b, q_a, q_b):
            q_inc_a = math.from_rotation_vector(inc_a[3:])
            q_inc_b = math.from_rotation_vector(inc_b[3:])
            q_a_new = math.quat_mul(q_inc_a, q_a)
            q_b_new = math.quat_mul(q_inc_b, q_b)
            return math.quat_residual(q_a_new, q_b_new)

        so3_err_a = jax.jacfwd(inc_quat_residual, argnums=0)(
            jnp.zeros([6]), jnp.zeros([6]), q_a, q_b
        )
        so3_err_b = jax.jacfwd(inc_quat_residual, argnums=1)(
            jnp.zeros([6]), jnp.zeros([6]), q_a, q_b
        )

        jac_hinge = jnp.concatenate(
            [
                jnp.concatenate([spherical_a, dot1_1_a, dot1_2_a, u_a_tangent_a]),
                jnp.concatenate([spherical_b, dot1_1_b, dot1_2_b, u_a_tangent_b]),
            ],
            axis=None,
        ) * (constraint_type == 0)
        jac_prismatic = jnp.concatenate(
            [
                jnp.concatenate([co_spherical_a, dot1_1_a, dot1_2_a, u_a_tangent_a]),
                jnp.concatenate([co_spherical_b, dot1_1_b, dot1_2_b, u_a_tangent_b]),
            ],
            axis=None,
        ) * (constraint_type == 1)
        jac_se3 = jnp.concatenate(
            [
                jnp.concatenate([co_spherical_a, so3_err_a]),
                jnp.concatenate([co_spherical_b, so3_err_b]),
            ],
            axis=None,
        ) * (constraint_type == 2)

        return jac_hinge + jac_prismatic + jac_se3

    # @partial(jit, static_argnums=0)
    def get_free_degrees(
        self,
        state: State,
        param: SimulationParameters,
    ) -> jax.Array:
        body_b_id = param.rigid_body_param.names.index(self.body_b)
        body_a_id = param.rigid_body_param.names.index(self.body_a)
        constraint_id = param.constraint_param.names.index(self.name)

        body_b_pos = state.conf.pos[body_b_id]
        body_b_rot = state.conf.rot[body_b_id]
        body_a_pos = state.conf.pos[body_a_id]
        body_a_rot = state.conf.rot[body_a_id]

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
        u_a = math.rotation_matrix(frame_a_rot)[:, 0]
        x = jnp.dot(u_a, r_delta)
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
        self,
        free_degree: int,
        param: SimulationParameters,
        body_a_transform: Transform,
        x: float,
    ) -> Transform:
        """
        Returns the configuration of the next body within a kinematic tree containing this joint.

        Parameters
        ----------
        param: Dict
            The system's parameters stored as a jax pytree with a dictionary at the top level.
        body_a_transform: Transform
            The transform of the previous body/frame in the kinematic chain.
        x: float
            Value of the free degree displacement between the frames (Assumes prismatic or hinge).
        Returns:
        -------
        Transform:
            The transform of the next body.
        """
        i = param.constraint_param.names.index(self.name)
        cp = param.constraint_param
        d0_a = cp.frame_a.position[i]
        frame_a_rot0 = cp.frame_a.rotation[i]
        d_a = math.rotate_vector(body_a_transform.rot, d0_a)
        frame_a_position = body_a_transform.pos + d_a
        frame_a_rot = math.quat_mul(body_a_transform.rot, frame_a_rot0)
        u_a = math.rotation_matrix(frame_a_rot)[:, 0]
        v_a = math.rotation_matrix(frame_a_rot)[:, 1]
        w_a = math.rotation_matrix(frame_a_rot)[:, 2]

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
        d_b = u_a * x
        frame_b_pos = frame_a_position - d_b
        frame_b_rot = frame_a_rot
        frame_b_rot0 = cp.frame_b.rotation[i]
        d0_b = cp.frame_b.position[i]
        frame_b_rot0_inv = math.conjugate(frame_b_rot0)
        prismatic_body_b_rotation = math.quat_mul(frame_b_rot, frame_b_rot0_inv)
        prismatic_body_b_position = frame_b_pos + d_a
        body_b_rotation = hinge_body_b_rotation * (
            free_degree == 5
        ) + prismatic_body_b_rotation * (free_degree == 0)
        body_b_position = hinge_body_b_position * (
            free_degree == 5
        ) + prismatic_body_b_position * (free_degree == 0)

        return Transform(body_b_position, body_b_rotation)

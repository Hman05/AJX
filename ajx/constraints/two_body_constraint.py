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
    ConstraintResidual,
    get_frame_transform,
    get_frame_transform_ext,
    place_body_b,
    wrapped_offset,
)


class TwoBodyConstraint(Constraint):
    """
    A constraint that restricts the relative motion between two bodies.
    """

    name: str
    body_a: str
    body_b: str
    constraint_residual: ConstraintResidual
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
        self,
        name: str,
        body_a: str,
        body_b: str,
        constraint_residual: ConstraintResidual,
    ):
        self.name = name
        self.body_a = body_a
        self.body_b = body_b
        self.constraint_residual = constraint_residual

    def get_multiplier_names(self) -> Tuple[str]:
        if self.constraint_residual == ConstraintResidual.AXIAL_WORLD_SPHERICAL.value:
            return ("nx", "ny", "nz", "n_bend1", "n_bend2", "n_torsion")
        elif self.constraint_residual == ConstraintResidual.AXIAL_LOCAL_SPHERICAL.value:
            return ("nu", "nw", "nv", "n_bend1", "n_torsion", "n_bend2")
        elif self.constraint_residual == ConstraintResidual.BEND_TWIST.value:
            return ("nu", "nv", "nw", "n_bend1", "n_bend2", "n_twist")
        return ()

    def compute_offset(
        default_offset: jax.Array,
        target: jax.Array,
        constraint_residual: ConstraintResidual,
    ):
        return wrapped_offset(default_offset, target, constraint_residual)

    def object_func(
        self,
        state: State,
        param: SimulationParameters,
    ):
        body_b_id = param.rigid_body_param.names.index(self.body_b)
        body_a_id = param.rigid_body_param.names.index(self.body_a)
        constraint_id = param.constraint_param.names.index(self.name)
        return TwoBodyConstraint.func(
            param,
            state,
            (body_a_id, body_b_id),
            (constraint_id,),
            self.constraint_residual,
        )

    def object_jacobian(
        self,
        state: State,
        param: SimulationParameters,
    ):
        body_b_id = param.rigid_body_param.names.index(self.body_b)
        body_a_id = param.rigid_body_param.names.index(self.body_a)
        constraint_id = param.constraint_param.names.index(self.name)
        return TwoBodyConstraint.jacobian(
            param,
            state,
            (body_a_id, body_b_id),
            (constraint_id,),
            self.constraint_residual,
        )

    @jit
    def func(
        param: SimulationParameters,
        state: State,
        body_ids: Tuple[Union[int, jax.Array]],
        constraint_ids: Tuple[Union[int, jax.Array]],
        constraint_residual: Union[ConstraintResidual, jax.Array],
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
        dot1_0 = jnp.dot(u_a, u_b)
        dot1_1 = jnp.dot(u_a, v_b)
        dot1_2 = jnp.dot(u_a, w_b)
        dot3_2 = jnp.dot(w_a, v_b)

        cross1_0 = jnp.linalg.norm(jnp.cross(u_a, u_b))
        cross1_1 = jnp.linalg.norm(jnp.cross(u_a, v_b))
        cross1_2 = jnp.linalg.norm(jnp.cross(u_a, w_b))
        cross3_2 = jnp.linalg.norm(jnp.cross(w_a, v_b))
        bend0 = jnp.atan2(cross1_0, dot1_0)
        bend1 = jnp.atan2(cross1_1, dot1_1) - jnp.pi / 2
        bend2 = jnp.atan2(cross1_2, dot1_2) - jnp.pi / 2
        twist = jnp.atan2(cross3_2, dot3_2) - jnp.pi / 2

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
        ) * (constraint_residual == ConstraintResidual.AXIAL_WORLD_SPHERICAL.value)
        prismatic_constraint = jnp.block(
            [co_spherical, dot1_1[None], dot1_2[None], free_hinge[None]]
        ) * (constraint_residual == ConstraintResidual.AXIAL_LOCAL_SPHERICAL.value)
        bend_twist_constraint = jnp.block([co_spherical, bend1, bend2, twist]) * (
            constraint_residual == ConstraintResidual.BEND_TWIST.value
        )
        return hinge_constraint + prismatic_constraint + bend_twist_constraint

    @jit
    def jacobian(
        param: SimulationParameters,
        state: State,
        body_ids: Tuple[Union[int, jax.Array]],
        constraint_ids: Tuple[Union[int, jax.Array]],
        constraint_residual: Union[ConstraintResidual, jax.Array],
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

        def angular_jacobian(a, b):
            epsilon = 1e-12
            d_cross = jnp.cross(a, b)
            d_cross_norm = jnp.linalg.norm(d_cross)

            Ga = -d_cross[None] / (d_cross_norm + epsilon)
            Gb = d_cross[None] / (d_cross_norm + epsilon)
            Ga_ext = jnp.concatenate([jnp.zeros([1, 3]), Ga], axis=1)
            Gb_ext = jnp.concatenate([jnp.zeros([1, 3]), Gb], axis=1)
            return Ga_ext, Gb_ext

        bend0_jac_a, bend0_jac_b = angular_jacobian(u_a, u_b)
        filler_jac_a, filler_jac_b = jnp.zeros([1, 6]), jnp.zeros([1, 6])
        bend1_jac_a, bend1_jac_b = angular_jacobian(u_a, v_b)
        bend2_jac_a, bend2_jac_b = angular_jacobian(u_a, w_b)
        twist_jac_a, twist_jac_b = angular_jacobian(w_a, v_b)

        jac_hinge = jnp.concatenate(
            [
                jnp.concatenate([spherical_a, dot1_1_a, dot1_2_a, u_a_tangent_a]),
                jnp.concatenate([spherical_b, dot1_1_b, dot1_2_b, u_a_tangent_b]),
            ],
            axis=None,
        ) * (constraint_residual == ConstraintResidual.AXIAL_WORLD_SPHERICAL.value)
        jac_prismatic = jnp.concatenate(
            [
                jnp.concatenate([co_spherical_a, dot1_1_a, dot1_2_a, u_a_tangent_a]),
                jnp.concatenate([co_spherical_b, dot1_1_b, dot1_2_b, u_a_tangent_b]),
            ],
            axis=None,
        ) * (constraint_residual == ConstraintResidual.AXIAL_LOCAL_SPHERICAL.value)
        jac_twist_bend = jnp.concatenate(
            [
                jnp.concatenate(
                    [co_spherical_a, bend1_jac_a, bend2_jac_a, twist_jac_a]
                ),
                jnp.concatenate(
                    [co_spherical_b, bend1_jac_b, bend2_jac_b, twist_jac_b]
                ),
            ],
            axis=None,
        ) * (constraint_residual == ConstraintResidual.BEND_TWIST.value)

        return jac_hinge + jac_prismatic + jac_twist_bend

    def place_other(
        self,
        param: SimulationParameters,
        body_a_transform: Transform,
        degrees: Union[int, Tuple[int]] = (),
        values: Union[float, jax.Array] = (),
    ) -> Transform:
        """
        Returns the transform of body b that puts the given degrees of this
        constraint at the given values, and all other degrees at zero.

        Parameters
        ----------
        param: SimulationParameters
            The system's parameters.
        body_a_transform: Transform
            The transform of body a.
        degrees: int or Tuple[int]
            The degrees (rows of the constraint residual) to set. These can be the
            linear degrees 0, 1 and 2, and the hinge degree 5 of the axial residuals.
        values: float or jax.Array
            The value of each degree.
        Returns:
        -------
        Transform:
            The transform of body b.
        """
        i = param.constraint_param.names.index(self.name)
        cp = param.constraint_param
        return place_body_b(
            self.constraint_residual,
            cp.frame_a,
            cp.frame_b,
            i,
            body_a_transform,
            degrees,
            values,
        )

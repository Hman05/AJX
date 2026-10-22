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
from ajx.constraints.base import (
    Constraint,
    ConstraintResidual,
    get_frame_transform,
    get_frame_transform_ext,
    place_body_b,
)


@struct.dataclass
class OneBodyConstraint(Constraint):
    """
    A constraint that restricts the relative motion between a rigid body and the world.
    """

    name: str = struct.field(pytree_node=False)
    body: str = struct.field(pytree_node=False)
    constraint_residual: ConstraintResidual = struct.field(pytree_node=False)

    @classmethod
    def get_num_bodies(cls):
        return 1

    def get_operand_sizes():
        return (6,)

    @classmethod
    def get_constrained_degrees(cls):
        return 6

    def get_parameter_group_names():
        return ("constraint_param",)

    def get_body_group_names():
        return ("rigid_body_param",)

    def get_gvel_names():
        return ("data",)

    @property
    def bodies(self):
        return (self.body,)

    @property
    def names(self):
        return (self.name,)

    # def __init__(self, name: str, body: str, constraint_residual: ConstraintResidual):
    #     self.name = name
    #     self.body = body
    #     self.constraint_residual = constraint_residual

    def get_multiplier_names(self) -> Tuple[str]:
        if self.constraint_residual == ConstraintResidual.AXIAL_WORLD_SPHERICAL.value:
            return ("nx", "ny", "nz", "n_bend", "n_torsion", "t")
        elif self.constraint_residual == ConstraintResidual.AXIAL_LOCAL_SPHERICAL.value:
            return ("nu", "nw", "n_bend1", "n_torsion", "n_bend2", "t")
        elif self.constraint_residual == ConstraintResidual.SE3.value:
            return ("nu", "nv", "nw", "nru", "nrv", "nrw")
        elif self.constraint_residual == ConstraintResidual.BEND_TWIST.value:
            return ("nu", "nv", "nw", "n_bend1", "n_bend2", "n_twist")
        return ()

    def compute_offset(
        default_offset: jax.Array,
        target: jax.Array,
        constraint_residual: ConstraintResidual,
    ):
        hinge_rotational_degrees = jnp.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=bool
        ) * (constraint_residual == ConstraintResidual.AXIAL_WORLD_SPHERICAL.value)
        prismatic_rotational_degrees = jnp.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=bool
        ) * (constraint_residual == ConstraintResidual.AXIAL_LOCAL_SPHERICAL.value)
        se3_rotational_degrees = jnp.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=bool
        ) * (constraint_residual == ConstraintResidual.SE3.value)
        rotational_degrees = (
            hinge_rotational_degrees
            + prismatic_rotational_degrees
            + se3_rotational_degrees
        )
        linear_degrees = jnp.logical_not(rotational_degrees)

        linear_offset = (default_offset - target) * linear_degrees
        roational_offset = (
            (default_offset - target + jnp.pi) % (2 * jnp.pi) - jnp.pi
        ) * rotational_degrees

        return linear_offset + roational_offset

    def object_func(
        self,
        state: State,
        param: SimulationParameters,
    ):
        body_id = param.rigid_body_param.names.index(self.body)
        constraint_id = param.constraint_param.names.index(self.name)
        return OneBodyConstraint.func(
            param, state, (body_id,), (constraint_id,), self.constraint_residual
        )

    def object_jacobian(
        self,
        state: State,
        param: SimulationParameters,
    ):
        body_id = param.rigid_body_param.names.index(self.body)
        constraint_id = param.constraint_param.names.index(self.name)
        return OneBodyConstraint.jacobian(
            param, state, (body_id,), (constraint_id,), self.constraint_residual
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
        body_id = body_ids[0]
        constraint_id = constraint_ids[0]
        body_b_pos = state.conf.pos[body_id]
        body_b_rot = state.conf.rot[body_id]
        body_a_pos = jnp.array([0.0, 0.0, 0.0])
        body_a_rot = jnp.array([1.0, 0.0, 0.0, 0.0])

        d_a, u_a, v_a, w_a, q_a = get_frame_transform_ext(
            param.constraint_param.frame_a, constraint_id, body_a_pos, body_a_rot
        )

        d_b, u_b, v_b, w_b, q_b = get_frame_transform_ext(
            param.constraint_param.frame_b, constraint_id, body_b_pos, body_b_rot
        )

        r_a = body_a_pos + d_a
        r_b = body_b_pos + d_b

        spherical = r_a - r_b
        co_spherical = math.rotate_vector(math.conjugate(q_a), spherical)
        dot1_1 = jnp.dot(u_a, v_b)
        dot1_2 = jnp.dot(u_a, w_b)
        dot3_2 = jnp.dot(w_a, v_b)

        cross1_1 = jnp.linalg.norm(jnp.cross(u_a, v_b))
        cross1_2 = jnp.linalg.norm(jnp.cross(u_a, w_b))
        cross3_2 = jnp.linalg.norm(jnp.cross(w_a, v_b))
        bend1 = jnp.atan2(cross1_1, dot1_1) - jnp.pi / 2
        bend2 = jnp.atan2(cross1_2, dot1_2) - jnp.pi / 2
        twist = jnp.atan2(cross3_2, dot3_2) - jnp.pi / 2

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

        # For prismatic
        r_a = body_a_pos + d_a
        r_b = body_b_pos + d_b

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
        se3_constraint = jnp.block([co_spherical, so3_err]) * (
            constraint_residual == ConstraintResidual.SE3.value
        )
        bend_twist_constraint = jnp.block([co_spherical, bend1, bend2, twist]) * (
            constraint_residual == ConstraintResidual.BEND_TWIST.value
        )
        return (
            hinge_constraint
            + prismatic_constraint
            + se3_constraint
            + bend_twist_constraint
        )

    @partial(jit, static_argnums=0)
    def jacobian(
        self,
        param: SimulationParameters,
        state: State,
    ) -> jax.Array:
        body_id = param.rigid_body_param.names.index(self.body)
        constraint_id = param.constraint_param.names.index(self.name)
        return OneBodyConstraint.jacobian(
            param, state, (body_id,), constraint_id, self.constraint_residual
        )

    @jit
    def jacobian(
        param: SimulationParameters,
        state: State,
        body_ids: Tuple[Union[int, jax.Array]],
        constraint_ids: Tuple[Union[int, jax.Array]],
        constraint_residual: Union[ConstraintResidual, jax.Array],
    ) -> jax.Array:
        body_id = body_ids[0]
        constraint_id = constraint_ids[0]
        body_b_pos = state.conf.pos[body_id]
        body_b_rot = state.conf.rot[body_id]
        body_a_pos = jnp.array([0.0, 0.0, 0.0])
        body_a_rot = jnp.array([1.0, 0.0, 0.0, 0.0])

        d_a, u_a, v_a, w_a, q_a = get_frame_transform_ext(
            param.constraint_param.frame_a, constraint_id, body_a_pos, body_a_rot
        )
        d_b, u_b, v_b, w_b, q_b = get_frame_transform_ext(
            param.constraint_param.frame_b, constraint_id, body_b_pos, body_b_rot
        )

        spherical_b = jnp.block([-jnp.eye(3), math.skew(d_b)])
        dot1_1_b = jnp.block([jnp.zeros([1, 3]), -jnp.cross(u_a, v_b)])
        dot1_2_b = jnp.block([jnp.zeros([1, 3]), -jnp.cross(u_a, w_b)])

        R_a = math.rotation_matrix(math.conjugate(q_a))
        co_spherical_b = jnp.block([-R_a, R_a @ math.skew(d_b)])
        u_a_tangent_b = jnp.block([jnp.zeros([1, 3]), u_a])

        def angular_jacobian(a, b):
            epsilon = 1e-12
            d_cross = jnp.cross(a, b)
            d_cross_norm = jnp.linalg.norm(d_cross)

            Gb = d_cross[None] / (d_cross_norm + epsilon)
            Gb_ext = jnp.concatenate([jnp.zeros([1, 3]), Gb], axis=1)
            return Gb_ext

        bend1_jac_b = angular_jacobian(u_a, v_b)
        bend2_jac_b = angular_jacobian(u_a, w_b)
        twist_jac_b = angular_jacobian(w_a, v_b)

        def inc_quat_residual(inc_a, inc_b, q_a, q_b):
            q_inc_a = math.from_rotation_vector(inc_a[3:])
            q_inc_b = math.from_rotation_vector(inc_b[3:])
            q_a_new = math.quat_mul(q_inc_a, q_a)
            q_b_new = math.quat_mul(q_inc_b, q_b)
            return math.quat_residual(q_a_new, q_b_new)

        so3_err_b = jax.jacfwd(inc_quat_residual, argnums=1)(
            jnp.zeros([6]), jnp.zeros([6]), q_a, q_b
        )

        jac_hinge = jnp.concatenate(
            [spherical_b, dot1_1_b, dot1_2_b, u_a_tangent_b], axis=None
        ) * (constraint_residual == 0)
        jac_prismatic = jnp.concatenate(
            [co_spherical_b, dot1_1_b, dot1_2_b, u_a_tangent_b], axis=None
        ) * (constraint_residual == 1)
        jac_se3 = jnp.concatenate([co_spherical_b, so3_err_b], axis=None) * (
            constraint_residual == 2
        )
        jac_twist_bend = jnp.concatenate(
            [co_spherical_b, bend1_jac_b, bend2_jac_b, twist_jac_b], axis=None
        ) * (constraint_residual == ConstraintResidual.BEND_TWIST.value)

        return jac_hinge + jac_prismatic + jac_se3 + jac_twist_bend

    def place_other(
        self,
        param: SimulationParameters,
        degrees: Union[int, Tuple[int]] = (),
        values: Union[float, jax.Array] = (),
    ) -> Transform:
        """
        Returns the transform of the body that puts the given degrees of this
        constraint at the given values, and all other degrees at zero.

        Parameters
        ----------
        param: SimulationParameters
            The system's parameters.
        degrees: int or Tuple[int]
            The degrees (rows of the constraint residual) to set. These can be the
            linear degrees 0, 1 and 2, and the hinge degree 5 of the axial residuals.
        values: float or jax.Array
            The value of each degree.
        Returns:
        -------
        Transform:
            The transform of the body.
        """
        i = param.constraint_param.names.index(self.name)
        cp = param.constraint_param
        return place_body_b(
            self.constraint_residual,
            cp.frame_a,
            cp.frame_b,
            i,
            Transform.identity(),
            degrees,
            values,
        )

    def place_frame_a(self, state: State, param: SimulationParameters):
        body_id = param.rigid_body_param.names.index(self.body)
        constraint_id = param.constraint_param.names.index(self.name)

        world_body_pos = state.conf.pos[body_id]
        world_body_rot = state.conf.rot[body_id]

        body_frame_b_pos = param.constraint_param.frame_b.position[constraint_id]
        body_frame_b_rot = param.constraint_param.frame_b.rotation[constraint_id]

        world_frame_b_rot = math.quat_mul(body_frame_b_rot, world_body_rot)
        world_frame_b_pos = world_body_pos + math.rotate_vector(
            world_frame_b_rot, body_frame_b_pos
        )

        world_frame_a_rot = world_frame_b_rot
        world_frame_a_pos = world_frame_b_pos

        return Transform(world_frame_a_pos, world_frame_a_rot)

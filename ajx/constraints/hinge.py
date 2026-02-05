import jax.numpy as jnp
from flax import struct
from jax import jit
from functools import partial

from ajx.constraints.base import Constraint
from ajx.definitions import Configuration
from typing import Optional
import ajx.math as math
from ajx.definitions import State


@struct.dataclass
class HingeJoint(Constraint):
    name: str
    body_a: Optional[str]
    body_b: str

    @property
    def is_attached_to_world(self):
        return self.body_a is None

    @property
    def dof(self):
        return 6

    @property
    def multiplier_suffixes(self):
        return ["nx", "ny", "nz", "n_bend", "n_torsion", "t"]

    @partial(jit, static_argnums=0)
    def c(self, param, state: State):
        body_b_id = param.rigid_body_param.names.index(self.body_b)
        body_b_pos = state.conf.pos[body_b_id]
        body_b_rot = state.conf.rot[body_b_id]

        if self.is_attached_to_world:
            body_a_pos = jnp.zeros(3)
            body_a_rot = jnp.array([1.0, 0.0, 0.0, 0.0])
        else:
            body_a_id = param.rigid_body_param.names.index(self.body_a)
            body_a_pos = state.conf.pos[body_a_id]
            body_a_rot = state.conf.rot[body_a_id]

        d_a, u_a, v_a, w_a = Constraint.get_frame_transform(
            param.get_cp(self.name).frame_a, body_a_pos, body_a_rot
        )

        d_b, u_b, v_b, w_b = Constraint.get_frame_transform(
            param.get_cp(self.name).frame_b, body_b_pos, body_b_rot
        )

        r_a = body_a_pos + d_a
        r_b = body_b_pos + d_b

        spherical = r_a - r_b
        dot1_1 = jnp.dot(u_a, v_b)
        dot1_2 = jnp.dot(u_a, w_b)

        return jnp.concatenate([spherical, dot1_1[None], dot1_2[None]])

    @partial(jit, static_argnums=0)
    def jac(self, param, state):
        body_b_id = param.rigid_body_param.names.index(self.body_b)

        body_b_pos = state.conf.pos[body_b_id]
        body_b_rot = state.conf.rot[body_b_id]

        if self.is_attached_to_world:
            body_a_pos = jnp.zeros(3)
            body_a_rot = jnp.array([1.0, 0.0, 0.0, 0.0])
        else:
            body_a_id = param.rigid_body_param.names.index(self.body_a)
            body_a_pos = state.conf.pos[body_a_id]
            body_a_rot = state.conf.rot[body_a_id]

        d_a, u_a, v_a, w_a = Constraint.get_frame_transform(
            param.get_cp(self.name).frame_a, body_a_pos, body_a_rot
        )
        d_b, u_b, v_b, w_b = Constraint.get_frame_transform(
            param.get_cp(self.name).frame_b, body_b_pos, body_b_rot
        )

        jac = {}

        jac[body_b_id] = jnp.block(
            [
                [-jnp.eye(3), math.skew(d_b)],
                [jnp.zeros([1, 3]), -jnp.cross(u_a, v_b)],
                [jnp.zeros([1, 3]), -jnp.cross(u_a, w_b)],
            ]
        )
        if self.is_attached_to_world:
            return jac

        jac[body_a_id] = jnp.block(
            [
                [jnp.eye(3), -math.skew(d_a)],
                [jnp.zeros([1, 3]), jnp.cross(u_a, v_b)],
                [jnp.zeros([1, 3]), jnp.cross(u_a, w_b)],
            ]
        )
        return jac

    def jac_and_proj(
        param,
        state,
        body_a_id,
        body_b_id,
        constraint_id,
    ):
        body_b_pos = state.conf.pos[body_b_id]
        body_b_rot = state.conf.rot[body_b_id]

        is_attached_to_world = body_a_id == -1
        natw = jnp.logical_not(is_attached_to_world)
        quat_default = jnp.array([1.0, 0.0, 0.0, 0.0]) * is_attached_to_world

        # Note that jax clamps indices that are out of range.
        # We utilize this here to avoid branching (body_a_id could be -1)
        body_a_pos = state.conf.pos[body_a_id] * natw
        body_a_rot = state.conf.rot[body_a_id] * natw + quat_default

        d_a, u_a, v_a, w_a = Constraint.get_frame_transform(
            param.constraint_param[constraint_id].frame_a, body_a_pos, body_a_rot
        )
        d_b, u_b, v_b, w_b = Constraint.get_frame_transform(
            param.constraint_param[constraint_id].frame_b, body_b_pos, body_b_rot
        )

        # Order: jac_a, jac_b
        jac = jnp.stack(
            [
                jnp.block(
                    [
                        [jnp.eye(3), -math.skew(d_a)],
                        [jnp.zeros([1, 3]), jnp.cross(u_a, v_b)],
                        [jnp.zeros([1, 3]), jnp.cross(u_a, w_b)],
                        [jnp.zeros([1, 3]), u_a],
                    ]
                ),
                jnp.block(
                    [
                        [-jnp.eye(3), math.skew(d_b)],
                        [jnp.zeros([1, 3]), -jnp.cross(u_a, v_b)],
                        [jnp.zeros([1, 3]), -jnp.cross(u_a, w_b)],
                        [jnp.zeros([1, 3]), -u_a],
                    ]
                ),
            ]
        )
        return jac

    def c2(
        param,
        state,
        body_a_id,
        body_b_id,
        constraint_id,
    ):
        body_b_pos = state.conf.pos[body_b_id]
        body_b_rot = state.conf.rot[body_b_id]

        is_attached_to_world = body_a_id == -1
        natw = jnp.logical_not(is_attached_to_world)
        quat_default = jnp.array([1.0, 0.0, 0.0, 0.0]) * is_attached_to_world

        # Note that jax clamps indices that are out of range.
        # We utilize this here to avoid branching (body_a_id could be -1)
        body_a_pos = state.conf.pos[body_a_id] * natw
        body_a_rot = state.conf.rot[body_a_id] * natw + quat_default

        d_a, u_a, v_a, w_a = Constraint.get_frame_transform(
            param.constraint_param[constraint_id].frame_a, body_a_pos, body_a_rot
        )

        d_b, u_b, v_b, w_b = Constraint.get_frame_transform(
            param.constraint_param[constraint_id].frame_b, body_b_pos, body_b_rot
        )

        r_a = body_a_pos + d_a
        r_b = body_b_pos + d_b

        spherical = r_a - r_b
        dot1_1 = jnp.dot(u_a, v_b)
        dot1_2 = jnp.dot(u_a, w_b)

        return jnp.concatenate([spherical, dot1_1[None], dot1_2[None]])

    @partial(jit, static_argnums=0)
    def tangential_projection(self, param, state):
        body_b_id = param.rigid_body_param.names.index(self.body_b)
        if self.is_attached_to_world:
            body_a_rot = jnp.array([1.0, 0.0, 0.0, 0.0])
        else:
            body_a_id = param.rigid_body_names.index(self.body_a)
            body_a_rot = state.conf.rot[body_a_id]

        frame_a_rot0 = param.get_cp(self.name).frame_a.rotation
        frame_a_rot = math.quat_mul(body_a_rot, frame_a_rot0)
        u_a = math.rotation_matrix(frame_a_rot)[:, 0]
        D = {}

        D[body_b_id] = jnp.block([[jnp.zeros([3]), -u_a]])

        if self.is_attached_to_world:
            return D
        D[body_a_id] = jnp.block([[jnp.zeros([3]), u_a]])
        return D

    def tangential_projection2(
        param,
        state,
        body_a_id,
        body_b_id,
        constraint_id,
    ):
        is_attached_to_world = body_a_id == -1
        natw = jnp.logical_not(is_attached_to_world)
        quat_default = jnp.array([1.0, 0.0, 0.0, 0.0]) * is_attached_to_world

        # Note that jax clamps indices that are out of range.
        # We utilize this here to avoid branching (body_a_id could be -1)
        body_a_rot = state.conf.rot[body_a_id] * natw + quat_default

        frame_a_rot0 = param.constraint_param[constraint_id].frame_a.rotation
        frame_a_rot = math.quat_mul(body_a_rot, frame_a_rot0)
        u_a = math.rotation_matrix(frame_a_rot)[:, 0]
        D = {}

        # Order: jac_a, jac_b
        jac = jnp.stack(
            [jnp.block([[jnp.zeros([3]), u_a]]), jnp.block([[jnp.zeros([3]), -u_a]])]
        )
        return jac

    @partial(jit, static_argnums=0)
    def tangential_projection_decomposed(self, param, state):
        body_b_id = param.rigid_body_names.index(self.body_b)
        if self.is_attached_to_world:
            body_a_rotation = jnp.array([1.0, 0.0, 0.0, 0.0])
        else:
            body_a_id = param.rigid_body_names.index(self.body_a)
            body_a_rotation = state[body_a_id].position[3:]

        frame_a_rot0 = param[self.name].frame_a.rotation
        frame_a_rot = math.quat_mul(body_a_rotation, frame_a_rot0)
        u_a = math.rotation_matrix(frame_a_rot)[:, 0]
        D = {}
        D[body_b_id] = jnp.block(
            [
                [jnp.zeros([3]), -u_a],
                [jnp.zeros([3]), u_a],
            ]
        )
        if self.is_attached_to_world:
            return D
        D[body_a_id] = jnp.block(
            [
                [jnp.zeros([3]), u_a],
                [jnp.zeros([3]), -u_a],
            ]
        )
        return D

    def place_other(
        self, param, body_a_transform: Configuration, theta: float
    ) -> Configuration:
        """
        Returns the configuration of the next body within a kinematic tree containing this hinge joint.

        Parameters
        ----------
        param: Dict
            The system's parameters stored as a jax pytree with a dictionary at the top level.
        body_a_transform: Configuration
            The configuration of the previous body/frame in the kinematic chain.
        theta: float
            The rotational displacement between the frames.
        Returns:
        -------
        Configuration:
            The configuration of the next body.
        """
        cp = param.get_cp(self.name)
        d0_a = cp.frame_a.position
        frame_a_rot0 = cp.frame_a.rotation
        d_a = math.rotate_vector(body_a_transform.rot, d0_a)
        frame_a_position = body_a_transform.pos + d_a

        frame_a_rot = math.quat_mul(body_a_transform.rot, frame_a_rot0)

        frame_b_rot0 = cp.frame_b.rotation

        frame_b_rot_as_seen_from_a = math.quat_from_axis_angle(
            jnp.array([1.0, 0.0, 0.0]), theta
        )
        frame_b_rot = math.quat_mul(frame_a_rot, frame_b_rot_as_seen_from_a)

        body_b_rotation = math.quat_mul(frame_b_rot, math.conjugate(frame_b_rot0))

        d0_b = cp.frame_b.position
        d_b = math.rotate_vector(body_b_rotation, d0_b)
        body_b_position = frame_a_position - d_b
        return Configuration(body_b_position, body_b_rotation)

    @partial(jit, static_argnums=0)
    def get_free_degrees(self, state, param):
        body_b_id = param.rigid_body_param.names.index(self.body_b)
        body_b_rot = state.conf.rot[body_b_id]
        if self.is_attached_to_world:
            body_a_rotation = jnp.array([1.0, 0.0, 0.0, 0.0])
        else:
            body_a_id = param.rigid_body_param.names.index(self.body_a)
            body_a_rotation = state.conf.rot[body_a_id]

        frame_a_rot0 = param.get_cp(self.name).frame_a.rotation
        frame_a_rot = math.quat_mul(body_a_rotation, frame_a_rot0)

        frame_b_rot0 = param.get_cp(self.name).frame_b.rotation
        frame_b_rot = math.quat_mul(body_b_rot, frame_b_rot0)

        frame_a_rot_inv = math.conjugate(frame_a_rot)
        delta_rotation = math.quat_mul(frame_a_rot_inv, frame_b_rot)
        axis_angle = math.to_rotation_vector(delta_rotation)
        axis = jnp.array([1.0, 0.0, 0.0])
        theta = jnp.dot(axis_angle, axis)
        return theta

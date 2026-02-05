import jax.numpy as jnp
from flax import struct
from jax import jit
from functools import partial

from ajx.constraints.base import Constraint
from typing import Optional
import ajx.math as math
from ajx.definitions import Configuration
from ajx.definitions import State


@struct.dataclass
class PrismaticJoint(Constraint):
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
        return ["nu", "nw", "n_bend1", "n_torsion", "n_bend2", "t"]

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

        dot2_1 = jnp.dot(u_a, r_a - r_b)
        dot2_2 = jnp.dot(w_a, r_a - r_b)
        dot1_1 = jnp.dot(u_a, v_b)
        dot1_2 = jnp.dot(u_a, w_b)
        dot1_3 = jnp.dot(w_a, v_b)

        return jnp.stack([dot2_1, dot2_2, dot1_1, dot1_2, dot1_3])

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

        dot2_1 = jnp.dot(u_a, r_a - r_b)
        dot2_2 = jnp.dot(w_a, r_a - r_b)
        dot1_1 = jnp.dot(u_a, v_b)
        dot1_2 = jnp.dot(u_a, w_b)
        dot1_3 = jnp.dot(w_a, v_b)

        return jnp.stack([dot2_1, dot2_2, dot1_1, dot1_2, dot1_3])

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

        r_a = body_a_pos + d_a
        r_b = body_b_pos + d_b

        jac = {}

        jac[body_b_id] = jnp.block(
            [
                [-u_a, jnp.zeros([1, 3])],
                [-w_a, jnp.zeros([1, 3])],
                [jnp.zeros([1, 3]), -jnp.cross(u_a, v_b)],
                [jnp.zeros([1, 3]), -jnp.cross(u_a, w_b)],
                [jnp.zeros([1, 3]), -jnp.cross(w_a, v_b)],
            ]
        )
        if self.is_attached_to_world:
            return jac

        jac[body_a_id] = jnp.block(
            [
                [u_a, jnp.cross(u_a, r_a - r_b)],
                [w_a, jnp.cross(w_a, r_a - r_b)],
                [jnp.zeros([1, 3]), jnp.cross(u_a, v_b)],
                [jnp.zeros([1, 3]), jnp.cross(u_a, w_b)],
                [jnp.zeros([1, 3]), jnp.cross(w_a, v_b)],
            ]
        )
        return jac

    @partial(jit, static_argnums=0)
    def tangential_projection(self, param, state):
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

        D = {}

        D[body_b_id] = jnp.block(
            [
                [-v_a, jnp.zeros([1, 3])],
            ]
        )
        if self.is_attached_to_world:
            return D

        D[body_a_id] = jnp.block(
            [
                [v_a, jnp.cross(v_a, r_a - r_b)],
            ]
        )
        return D

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

        r_a = body_a_pos + d_a
        r_b = body_b_pos + d_b

        # Order: jac_a, jac_b
        jac = jnp.stack(
            [
                jnp.block(
                    [
                        [u_a, jnp.cross(u_a, r_a - r_b)],
                        [w_a, jnp.cross(w_a, r_a - r_b)],
                        [jnp.zeros([1, 3]), jnp.cross(u_a, v_b)],
                        [jnp.zeros([1, 3]), jnp.cross(u_a, w_b)],
                        [jnp.zeros([1, 3]), jnp.cross(w_a, v_b)],
                        [v_a, jnp.cross(v_a, r_a - r_b)],
                    ]
                ),
                jnp.block(
                    [
                        [-u_a, jnp.zeros([1, 3])],
                        [-w_a, jnp.zeros([1, 3])],
                        [jnp.zeros([1, 3]), -jnp.cross(u_a, v_b)],
                        [jnp.zeros([1, 3]), -jnp.cross(u_a, w_b)],
                        [jnp.zeros([1, 3]), -jnp.cross(w_a, v_b)],
                        [-v_a, jnp.zeros([1, 3])],
                    ]
                ),
            ]
        )
        return jac

    def tangential_projection2(
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

        # Order: jac_a, jac_b
        jac = jnp.stack(
            [
                jnp.block(
                    [
                        [v_a, jnp.cross(v_a, r_a - r_b)],
                    ]
                ),
                jnp.block(
                    [
                        [-v_a, jnp.zeros([1, 3])],
                    ]
                ),
            ]
        )
        return jac

    def tangential_projection_decomposed(self, param, state):
        return {}

    @partial(jit, static_argnums=0)
    def get_free_degrees(self, state, param):
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

        r_delta = r_a - r_b

        x = jnp.dot(v_a, r_delta)

        return x

    @partial(jit, static_argnums=0)
    def place_other(
        self, param, body_a_transform: Configuration, x: float
    ) -> Configuration:
        """
        Returns the configuration of the next body within a kinematic tree containing this prismatic joint.

        Parameters
        ----------
        param: Dict
            The system's parameters stored as a jax pytree with a dictionary at the top level.
        body_a_transform: Configuration
            The configuration of the previous body/frame in the kinematic chain.
        x: float
            The displacement between the frames.
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

        v_a = math.rotation_matrix(frame_a_rot)[:, 1]

        # d_a, u_a, v_a, w_a = Constraint.get_frame_transform(
        #     cp.frame_a, body_a_transform.pos, body_a_transform.rot
        # )

        d_b = v_a * x
        frame_b_pos = frame_a_position - d_b
        frame_b_rot = frame_a_rot

        frame_b_rot0 = cp.frame_b.rotation
        d0_b = cp.frame_b.position
        frame_b_rot0_inv = math.conjugate(frame_b_rot0)

        body_b_rot = math.quat_mul(frame_b_rot, frame_b_rot0_inv)
        body_b_pos = frame_b_pos - d_a  # math.rotate_vector(frame_b_rot, d0_b)

        return Configuration(body_b_pos, body_b_rot)

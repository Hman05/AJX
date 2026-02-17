from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import vmap
import ajx.math as math
from typing import List, Tuple, Any, Dict, Sequence
from flax import struct
from ajx.tree_util import ParameterNode


@struct.dataclass
class RigidBody:
    name: Tuple[str]
    geometry: Tuple[str]


@struct.dataclass
class Transform(ParameterNode):
    """Dataclass for a transform by position and rotation"""

    pos: jax.Array
    rot: jax.Array

    def to_configuration(self):
        return Configuration(self.pos[None], self.rot[None])


@struct.dataclass
class Configuration(ParameterNode):
    """Dataclass for the full configuration of a system"""

    pos: jax.Array
    rot: jax.Array

    def retract(self, update: jax.Array) -> Configuration:
        assert update.size == self.tangent_size()
        n_bodies = self.pos.shape[0]
        if len(self.pos.shape) == 3:
            n_timesteps = self.pos.shape[0]
            return vmap(Configuration.retract)(self, update.reshape(n_timesteps, -1))
        if len(self.pos.shape) == 2:
            update = update.reshape(n_bodies, 6)
            delta_pos = update[:, :3]
            delta_rot = update[:, 3:]
            quaternion_delta = vmap(math.from_rotation_vector)(delta_rot)
            new_pos = self.pos + delta_pos
            new_rot = vmap(math.quat_mul)(quaternion_delta, self.rot)
        return Configuration(new_pos, new_rot)

    def tangent_size(self):
        assert len(self.pos.shape) == len(self.rot.shape)
        if len(self.pos.shape) == 3:
            n_timesteps = self.pos.shape[0]
            n_bodies = self.pos.shape[1]
            return 6 * n_timesteps * n_bodies
        elif len(self.pos.shape) == 2:
            n_bodies = self.pos.shape[0]
            return 6 * n_bodies
        raise Exception


@struct.dataclass
class GeneralizedVelocity(ParameterNode):
    data: jax.Array

    @property
    def vel(self):
        return self.data[..., :3]

    @property
    def ang(self):
        return self.data[..., 3:]


@struct.dataclass
class State(ParameterNode):
    conf: Configuration
    gvel: GeneralizedVelocity


@struct.dataclass
class Frame(ParameterNode):
    position: jnp.array
    rotation: jnp.array

    def to_frames(self):
        return Frames(self.position[None], self.rotation[None])


@struct.dataclass
class Frames(ParameterNode):
    position: jnp.array
    rotation: jnp.array


@struct.dataclass
class ConstraintParameters(ParameterNode):
    # Fixed
    names: Tuple[str] = struct.field(pytree_node=False)

    # Dynamic
    frame_a: Frames
    frame_b: Frames
    compliance: jax.Array  # Viscous compliance for velocity constrained dofs
    damping: jax.Array  # Ignored for velocity constrained dofs
    target: jax.Array  # Target velocity for velocity constrained dofs
    is_velocity: jax.Array  # Bools set to true for velocity constrained dofs

    @classmethod
    def create(
        cls,
        frame_a: Frame,
        frame_b: Frame,
        compliance: float,
        damping: float,
        b: float,
        name: str,
    ):
        holonomic_compliance = jnp.array([compliance] * 5)[None]
        holonomic_damping = jnp.array([damping] * 5)[None]
        viscous_compliance = jnp.array([1.0 / b])[None]
        ignored_damping = jnp.array([0.0])[None]
        target = jnp.zeros(6)[None]
        compliance = jnp.concatenate([holonomic_compliance, viscous_compliance], axis=1)
        damping = jnp.concatenate([holonomic_damping, ignored_damping], axis=1)
        is_velocity = jnp.array([False, False, False, False, False, True], dtype=bool)[
            None
        ]
        names = (name,)
        return cls(
            names,
            frame_a=frame_a.to_frames(),
            frame_b=frame_b.to_frames(),
            compliance=compliance,
            damping=damping,
            target=target,
            is_velocity=is_velocity,
        )

    @classmethod
    def create_empty(cls):
        return cls(
            names=(),
            frame_a=Frames(jnp.array([0, 3]), jnp.array([0, 4])),
            frame_b=Frames(jnp.array([0, 3]), jnp.array([0, 4])),
            compliance=jnp.array([0, 6]),
            damping=jnp.array([0, 6]),
            target=jnp.array([0, 6]),
            is_velocity=jnp.array([0, 6]),
        )

    @classmethod
    def create_locked(
        cls,
        frame_a: Frame,
        frame_b: Frame,
        compliance: float,
        damping: float,
        offset: float,
        name: str,
    ):
        frame_a_data = frame_a.concat()
        frame_b_data = frame_b.concat()
        holonomic_compliance = jnp.array([compliance] * 6)
        damping = jnp.array([damping] * 6)
        target = jnp.zeros(5)
        offset = jnp.array([offset])
        new_data = jnp.concatenate(
            [frame_a_data, frame_b_data, holonomic_compliance, damping, target, offset]
        )
        return cls(name, new_data)

    def insert(self, src):
        new = self.copy()
        for constraint_name, src2 in src.items():
            if constraint_name in [":", "all"]:
                idx = jnp.s_[:]
            elif constraint_name in self.names:
                idx = self.names.index(constraint_name)
            else:
                msg = f"The provided source ({constraint_name}) does not index destination correctly"
                raise Exception(msg)

            if src2 is None:
                continue
            for prop, val in src2.items():
                if prop == "compliance":
                    new.compliance = val
                elif prop == "compliance04":
                    new.compliance = new.compliance.at[..., idx, :4].set(val)
                elif prop == "compliance5":
                    new.compliance = new.compliance.at[..., idx, 5].set(val)
                elif prop == "damping":
                    new.damping = val
                elif prop == "damping04":
                    new.damping = new.damping.at[..., idx, :4].set(val)
                elif prop == "damping5":
                    new.damping = new.damping.at[..., idx, 5].set(val)
                elif prop == "target":
                    new.target = val
                elif prop == "target04":
                    new.target = new.target.at[..., idx, :4].set(val)
                elif prop == "target5":
                    new.target = new.target.at[..., idx, 5].set(val)
                else:
                    msg = f"The provided source ({prop}) does not index destination correctly"
                    raise Exception(msg)
        return new


@struct.dataclass
class RigidBodyParameters(ParameterNode):
    # Fixed
    names: Tuple[str] = struct.field(pytree_node=False)

    # 2D Array
    mass: jax.Array  # [nb, 1]
    mc: jax.Array  # [nb, 3]
    inertia: jax.Array  # [nb, 6]

    def get_inertia_matrix(self):
        # Assumes vmap...
        inertia = jnp.zeros([3, 3])
        triu = jnp.triu_indices(3)
        tril = (triu[1], triu[0])
        inertia = inertia.at[triu].set(self.inertia)
        inertia = inertia.at[tril].set(self.inertia)
        return inertia

    @classmethod
    def create(cls, mass: float, inertia_diag: jax.Array, name: str):
        mass = jnp.array(mass).reshape([1])[None]
        mc = jnp.array([0.0, 0.0, 0.0])[None]
        diag_indices = jnp.array([0, 3, 5])
        inertia = jnp.zeros(6).at[diag_indices].set(inertia_diag)[None]
        names = (name,)
        return cls(names, mass, mc, inertia)

    def insert(self, src):
        new = self.copy()
        if src is None:
            return new
        if isinstance(src, jax.Array):
            # TODO: Lazy fix for empty dict
            return new
        for rb_name, src2 in src.items():
            if not rb_name in self.names:
                msg = f"The provided source ({rb_name}) does not index destination correctly"
                raise Exception(msg)
            idx = self.names.index(rb_name)
            for prop, val in src2.items():
                if prop == "mass":
                    new.data = new.data.at[idx, 0].set(val)
                elif prop == "mc_x":
                    new.data = new.data.at[idx, 1].set(val)
                elif prop == "mc_y":
                    new.data = new.data.at[idx, 2].set(val)
                elif prop == "mc_z":
                    new.data = new.data.at[idx, 3].set(val)
                elif prop == "inertia_xx":
                    new.data = new.data.at[idx, 4].set(val)
                elif prop == "inertia_xy":
                    new.data = new.data.at[idx, 5].set(val)
                elif prop == "inertia_xz":
                    new.data = new.data.at[idx, 6].set(val)
                elif prop == "inertia_yy":
                    new.data = new.data.at[idx, 7].set(val)
                elif prop == "inertia_yz":
                    new.data = new.data.at[idx, 8].set(val)
                elif prop == "inertia_zz":
                    new.data = new.data.at[idx, 9].set(val)
                else:
                    msg = f"The provided source ({prop}) does not index destination correctly"
                    raise Exception(msg)
        return new

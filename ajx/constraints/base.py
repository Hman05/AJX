from abc import ABC, abstractmethod

import ajx.math as math
from jax import jit
import jax.numpy as jnp

from enum import Enum


class ConstraintType(Enum):
    HINGE = 0
    PRISMATIC = 1


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


@jit
def get_frame_transform_ext(frame, constraint_id, body_pos, body_rotation):
    d0 = frame.position[constraint_id]
    frame_rot0 = frame.rotation[constraint_id]
    frame_rot = math.quat_mul(body_rotation, frame_rot0)
    d = math.rotate_vector(body_rotation, d0)
    u = math.rotation_matrix(frame_rot)[:, 0]
    v = math.rotation_matrix(frame_rot)[:, 1]
    w = math.rotation_matrix(frame_rot)[:, 2]

    return d, u, v, w, frame_rot


class Constraint(ABC):
    pass

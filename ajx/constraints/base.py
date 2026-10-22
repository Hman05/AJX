from abc import ABC, abstractmethod

import ajx.math as math
import jax
from jax import jit
import jax.numpy as jnp

from enum import Enum
from ajx.definitions import Transform


class ConstraintResidual(Enum):
    AXIAL_WORLD_SPHERICAL = 0
    AXIAL_LOCAL_SPHERICAL = 1
    SE3 = 2
    BEND_TWIST = 3


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


# The degrees of a constraint are the rows of its residual. place_other can set the
# linear degrees, which are the offset r_a - r_b from the origin of frame b to the
# origin of frame a, in world coordinates for AXIAL_WORLD_SPHERICAL and in frame a
# coordinates for the other residuals. It can also set the hinge degree of the
# axial residuals, which is the angle of frame b about the x-axis of frame a.
LINEAR_DEGREES = (0, 1, 2)
HINGE_DEGREE = 5
AXIAL_RESIDUALS = (
    ConstraintResidual.AXIAL_WORLD_SPHERICAL,
    ConstraintResidual.AXIAL_LOCAL_SPHERICAL,
)


def is_wrapped_degree(constraint_residual, degree: int) -> bool:
    """
    Whether a degree is an angle that wraps around at +-pi, which is the hinge
    degree of the axial residuals
    """
    return (
        ConstraintResidual(constraint_residual) in AXIAL_RESIDUALS
        and degree == HINGE_DEGREE
    )


def wrapped_degrees(constraint_residual) -> jax.Array:
    """
    Mask of the degrees that wrap around at +-pi (see is_wrapped_degree). The
    residual type may be traced.
    """
    axial_values = jnp.array([residual.value for residual in AXIAL_RESIDUALS])
    is_axial = jnp.any(axial_values == constraint_residual)
    return jnp.zeros(6, dtype=bool).at[HINGE_DEGREE].set(is_axial)


def wrapped_offset(default_offset, target, constraint_residual) -> jax.Array:
    """
    The offset default_offset - target of the residual from its target, wrapped to
    [-pi, pi) for the degrees that wrap around
    """
    offset = default_offset - target
    wrapped = (offset + jnp.pi) % (2 * jnp.pi) - jnp.pi
    return jnp.where(wrapped_degrees(constraint_residual), wrapped, offset)


def place_body_b(
    constraint_residual,
    frame_a,
    frame_b,
    constraint_id,
    body_a_transform,
    degrees,
    values,
) -> Transform:
    """
    Returns the transform of body b that puts the given degrees at the given values,
    and all other degrees of the constraint at zero.
    """
    constraint_residual = ConstraintResidual(constraint_residual)
    degrees = (degrees,) if isinstance(degrees, int) else tuple(degrees)
    values = jnp.atleast_1d(jnp.asarray(values))
    assert values.shape == (len(degrees),), "Expected one value per degree"
    settable_degrees = LINEAR_DEGREES
    if constraint_residual in AXIAL_RESIDUALS:
        settable_degrees += (HINGE_DEGREE,)
    for degree in degrees:
        if degree not in settable_degrees:
            raise ValueError(
                f"Degree {degree} of {constraint_residual.name} cannot be placed, "
                f"only degrees {settable_degrees}"
            )

    linear = jnp.zeros(3, dtype=values.dtype)
    angle = jnp.zeros((), dtype=values.dtype)
    for degree, value in zip(degrees, values):
        if degree == HINGE_DEGREE:
            angle = value
        else:
            linear = linear.at[degree].set(value)

    frame_a_rot = math.quat_mul(body_a_transform.rot, frame_a.rotation[constraint_id])
    frame_a_pos = body_a_transform.pos + math.rotate_vector(
        body_a_transform.rot, frame_a.position[constraint_id]
    )
    if constraint_residual != ConstraintResidual.AXIAL_WORLD_SPHERICAL:
        linear = math.rotate_vector(frame_a_rot, linear)
    frame_b_pos = frame_a_pos - linear
    frame_b_rot = math.quat_mul(
        frame_a_rot, math.quat_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), angle)
    )

    body_b_rot = math.quat_mul(
        frame_b_rot, math.conjugate(frame_b.rotation[constraint_id])
    )
    body_b_pos = frame_b_pos - math.rotate_vector(
        body_b_rot, frame_b.position[constraint_id]
    )
    return Transform(body_b_pos, body_b_rot)


class Constraint(ABC):
    def get_degree(self, state, param, degree):
        """
        Returns the value of a degree of the constraint, or of several degrees if
        degree is a sequence. The degrees are the rows of the constraint residual.
        """
        return self.object_func(state, param)[jnp.asarray(degree, dtype=int)]

import jax.numpy as jnp
from jax import jit, custom_jvp
import jax


@jit
def skew(vector: jax.Array) -> jax.Array:
    """
    Get skew-symmetric matrix corresponding to vector
    """
    return jnp.array(
        [
            [0, -vector[2], vector[1]],
            [vector[2], 0, -vector[0]],
            [-vector[1], vector[0], 0],
        ]
    )


def quat_mul(a: jax.Array, b: jax.Array) -> jax.Array:
    """
    Quaternion multiplication
    """
    a_s = a[0]
    a_v = a[1:]
    b_s = b[0]
    b_v = b[1:]

    out_s = a_s * b_s - jnp.dot(a_v, b_v)
    out_v = a_s * b_v + b_s * a_v + jnp.cross(a_v, b_v)

    return jnp.concatenate([out_s[None], out_v])


def rotate_vector(quat: jax.Array, vector: jax.Array) -> jax.Array:
    """
    Rotate a vector by rotation represented as a quaternion.
    """
    p = jnp.concatenate([jnp.zeros(1), vector])
    qstar = conjugate(quat)
    p_prime = quat_mul(quat, quat_mul(p, qstar))
    return p_prime[1:]


def rotation_matrix(quat: jax.Array) -> jax.Array:
    """
    Create a rotation matrix from quaternion
    """
    return (
        jnp.eye(3) + 2 * quat[0] * skew(quat[1:]) + 2 * skew(quat[1:]) @ skew(quat[1:])
    )


def conjugate(quat: jax.Array) -> jax.Array:
    """
    Get the quaternion conjugate
    """
    return quat * jnp.array([1, -1, -1, -1])


def normalize(quat: jax.Array) -> jax.Array:
    return quat / jnp.linalg.norm(quat)


def quat_from_axis_angle(normalized_axis: jax.Array, angle: jax.Array) -> jax.Array:
    """
    Get quaternion rotation from a scaled axis angle rotation.
    """
    cos_half = jnp.cos(0.5 * angle)
    sin_half = jnp.sin(0.5 * angle)
    return jnp.concatenate([cos_half[None], sin_half * normalized_axis])


# Yet to be fully tested what proper limits should be
series_expansion_limit = 1e-1
series_expansion_limit_jvp = 1e-1


@custom_jvp
def from_rotation_vector(rotation_vector: jax.Array) -> jax.Array:
    """
    Get the quaternion representation of a rotation_vector.
    """

    angle = jnp.linalg.norm(rotation_vector)
    cos_half = jnp.cos(0.5 * angle)
    eps = 1e-20
    extra_for_safe_division = 1.0 * (angle == 0)
    normalized_axis = rotation_vector / (angle + extra_for_safe_division)
    sin_half = jnp.sin(0.5 * angle)
    v = rotation_vector
    return jnp.concatenate(
        [
            cos_half[None],
            v / 2 - v / 48 * angle**2 + v / 3840 * angle**4 - v / 645120 * angle**6,
        ]
    ) * (angle <= series_expansion_limit) + jnp.concatenate(
        [
            cos_half[None],
            sin_half * normalized_axis,
        ]
    ) * (
        angle > series_expansion_limit
    )


@from_rotation_vector.defjvp
def _quaternion_from_rotation_vector(primal, tangent):
    (v,) = primal
    (v_dot,) = tangent
    angle = jnp.linalg.norm(v)
    eps = (angle == 0) * 1e-8
    angle_safe = angle + eps
    normalized_axis = v / angle_safe
    cos_half = jnp.cos(0.5 * angle)
    sin_half = jnp.sin(0.5 * angle)
    primal_out = jnp.concatenate(
        [
            cos_half[None],
            v / 2 - v / 48 * angle**2 + v / 3840 * angle**4 - v / 645120 * angle**6,
        ]
    ) * (angle <= series_expansion_limit) + jnp.concatenate(
        [
            cos_half[None],
            sin_half * normalized_axis,
        ]
    ) * (
        angle > series_expansion_limit
    )

    vvdot = jnp.dot(v, v_dot)
    angle2 = angle**2 + eps
    angle3 = angle**3 + eps
    q_dot_s2 = 0.5 * vvdot / angle_safe * jnp.sin(0.5 * angle)
    q_dot_v2 = (v_dot / angle_safe - v * vvdot / angle3) * jnp.sin(
        0.5 * angle
    ) + 0.5 * v * vvdot / angle2 * jnp.cos(0.5 * angle)

    q_dot_s = -1 / 4 * vvdot + 1 / 96 * vvdot * angle**2
    q_dot_v = (
        0.5 * v_dot
        - 1 / 24 * vvdot * v
        - 1 / 48 * v_dot * angle**2
        + 1 / 960 * vvdot * v * angle**2
    )

    tangent_out = jnp.concatenate([q_dot_s[None], q_dot_v]) * (
        angle <= series_expansion_limit_jvp
    ) + jnp.concatenate([q_dot_s2[None], q_dot_v2]) * (
        angle > series_expansion_limit_jvp
    )

    return primal_out, tangent_out


@jit
def quat_residual(q1, q2):
    """
    Get the right-invariant residual between two rotations (unit quaternions)
    as a rotation vector
    """
    q12 = quat_mul(q1, conjugate(q2))
    return to_rotation_vector(q12)


@jit
def to_axis_angle(quat: jax.Array):
    """
    Get a scaled axis angle representation of quaternion rotation.
    """
    norm = 2 * jnp.arctan2(jnp.linalg.norm(quat[1:]), quat[0])
    sin_norm_half = jnp.sin(norm / 2)
    # norm = 2*jnp.arctan2(jnp.linalg.norm(q[1:]), q[0])
    # sin_norm_half = jnp.sin(norm/2)
    extra_for_safe_division = (sin_norm_half == 0.0) * 1.0
    axis = quat[1:] / (sin_norm_half + extra_for_safe_division)
    angle = norm
    return axis, angle


@custom_jvp
def to_rotation_vector(quat: jax.Array):
    """
    Get a scaled axis angle representation of quaternion rotation.
    We write the resulting rotation vector theta as the product of an axis
    n and angle gamma.
    """
    gamma = 2 * jnp.arctan2(jnp.linalg.norm(quat[1:]), quat[0])
    sin_gamma_half = jnp.sin(gamma / 2)
    eps = (sin_gamma_half == 0.0) * 1.0e-8
    u = quat[1:] / (sin_gamma_half + eps) * gamma
    return u


@to_rotation_vector.defjvp
def to_rotation_vector_jvp(primal, tangent):
    q = primal[0]
    q_s = primal[0][0]
    q_v = primal[0][1:]
    qs_dot = tangent[0][0]
    qv_dot = tangent[0][1:]
    n, gamma = to_axis_angle(q)
    primal_out = n * gamma
    eps = (jnp.dot(q_v, n) == 0.0) * 1.0e-8
    gamma_dot = -2 * qs_dot / (jnp.dot(q_v, n) + eps)
    n_dot = (qv_dot - 0.5 * n * q_s * gamma_dot) / (jnp.dot(q_v, n) + eps)
    theta_dot_large = gamma_dot * n + gamma * n_dot
    theta_dot_small = 2 * qv_dot
    theta_dot = jnp.where(gamma < 1e-8, theta_dot_small, theta_dot_large)
    return primal_out, theta_dot

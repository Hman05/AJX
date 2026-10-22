import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ajx.math as math

IDENTITY = jnp.array([1.0, 0.0, 0.0, 0.0])
X_AXIS = jnp.array([1.0, 0.0, 0.0])
Y_AXIS = jnp.array([0.0, 1.0, 0.0])
Z_AXIS = jnp.array([0.0, 0.0, 1.0])

# Magnitudes on both sides of math.series_expansion_limit (0.1) and close to pi
ROTATION_ANGLES = [1e-6, 1e-3, 0.05, 0.0999, 0.1001, 0.5, 1.5, 3.0]
# Magnitudes between pi and 2 pi, where q has negative scalar part
LARGE_ROTATION_ANGLES = [3.5, 5.0, 6.0]


def random_unit_quaternions(n, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    # Use the hemisphere with non-negative scalar part so rotation angles are in [0, pi]
    q *= np.sign(q[:, :1])
    return jnp.array(q)


def random_unit_vectors(n, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    return jnp.array(v / np.linalg.norm(v, axis=1, keepdims=True))


QUATERNIONS = random_unit_quaternions(8)
VECTORS = random_unit_vectors(8, seed=1)


def assert_same_rotation(q1, q2, atol=1e-10):
    """Quaternions q and -q represent the same rotation."""
    assert min(jnp.abs(q1 - q2).max(), jnp.abs(q1 + q2).max()) < atol


# skew


def test_skew_matches_cross_product():
    a, b = VECTORS[0], VECTORS[1]
    np.testing.assert_allclose(math.skew(a) @ b, jnp.cross(a, b), atol=1e-14)


def test_skew_is_antisymmetric():
    S = math.skew(VECTORS[0])
    np.testing.assert_allclose(S, -S.T, atol=1e-14)


# Quaternion algebra


@pytest.mark.parametrize("q", QUATERNIONS)
def test_quat_mul_identity(q):
    np.testing.assert_allclose(math.quat_mul(IDENTITY, q), q, atol=1e-14)
    np.testing.assert_allclose(math.quat_mul(q, IDENTITY), q, atol=1e-14)


def test_quat_mul_is_associative():
    a, b, c = QUATERNIONS[:3]
    np.testing.assert_allclose(
        math.quat_mul(math.quat_mul(a, b), c),
        math.quat_mul(a, math.quat_mul(b, c)),
        atol=1e-14,
    )


@pytest.mark.parametrize("q", QUATERNIONS)
def test_quat_mul_with_conjugate_is_identity(q):
    np.testing.assert_allclose(
        math.quat_mul(q, math.conjugate(q)), IDENTITY, atol=1e-14
    )


def test_quat_mul_composes_rotations():
    a, b = QUATERNIONS[:2]
    v = VECTORS[0]
    np.testing.assert_allclose(
        math.rotate_vector(math.quat_mul(a, b), v),
        math.rotate_vector(a, math.rotate_vector(b, v)),
        atol=1e-14,
    )


def test_normalize_gives_unit_quaternion():
    q = jnp.array([1.0, 2.0, -3.0, 4.0])
    q_normalized = math.normalize(q)
    np.testing.assert_allclose(jnp.linalg.norm(q_normalized), 1.0, atol=1e-14)
    np.testing.assert_allclose(q_normalized * jnp.linalg.norm(q), q, atol=1e-14)


# Rotations of vectors and rotation matrices


@pytest.mark.parametrize(
    "axis, vector, expected",
    [
        (Z_AXIS, X_AXIS, Y_AXIS),
        (X_AXIS, Y_AXIS, Z_AXIS),
        (Y_AXIS, Z_AXIS, X_AXIS),
    ],
)
def test_quat_from_axis_angle_quarter_turn(axis, vector, expected):
    q = math.quat_from_axis_angle(axis, jnp.array(0.5 * jnp.pi))
    np.testing.assert_allclose(math.rotate_vector(q, vector), expected, atol=1e-14)


@pytest.mark.parametrize("q", QUATERNIONS)
def test_rotate_vector_matches_rotation_matrix(q):
    v = VECTORS[0]
    np.testing.assert_allclose(
        math.rotate_vector(q, v), math.rotation_matrix(q) @ v, atol=1e-14
    )


@pytest.mark.parametrize("q", QUATERNIONS)
def test_rotation_matrix_is_proper_rotation(q):
    R = math.rotation_matrix(q)
    np.testing.assert_allclose(R @ R.T, jnp.eye(3), atol=1e-14)
    np.testing.assert_allclose(jnp.linalg.det(R), 1.0, atol=1e-14)


@pytest.mark.parametrize("q", QUATERNIONS)
def test_matrix_to_quaternion_inverts_rotation_matrix(q):
    assert_same_rotation(math.matrix_to_quaternion(math.rotation_matrix(q)), q)


@pytest.mark.parametrize(
    "axis",
    [X_AXIS, Y_AXIS, Z_AXIS],
    ids=["largest_xx", "largest_yy", "largest_zz"],
)
def test_matrix_to_quaternion_negative_trace_branches(axis):
    # Rotations close to pi have negative trace, and the largest diagonal entry
    # of R is the one belonging to the rotation axis
    q = math.quat_from_axis_angle(axis, jnp.array(2.8))
    R = math.rotation_matrix(q)
    assert jnp.trace(R) < 0
    assert_same_rotation(math.matrix_to_quaternion(R), q)


@pytest.mark.parametrize("x", VECTORS)
def test_make_rotation_matrix_has_given_first_column(x):
    R = math.make_rotation_matrix(x)
    np.testing.assert_allclose(R[:, 0], x, atol=1e-14)
    np.testing.assert_allclose(R @ R.T, jnp.eye(3), atol=1e-14)
    np.testing.assert_allclose(jnp.linalg.det(R), 1.0, atol=1e-14)


def test_make_rotation_matrix_normalizes_and_handles_x_axis():
    R = math.make_rotation_matrix(jnp.array([2.0, 0.0, 0.0]))
    np.testing.assert_allclose(R[:, 0], X_AXIS, atol=1e-14)
    np.testing.assert_allclose(R @ R.T, jnp.eye(3), atol=1e-14)


# Rotation vectors


@pytest.mark.parametrize("angle", ROTATION_ANGLES)
@pytest.mark.parametrize("axis", VECTORS[:3])
def test_from_rotation_vector_matches_axis_angle(axis, angle):
    np.testing.assert_allclose(
        math.from_rotation_vector(angle * axis),
        math.quat_from_axis_angle(axis, jnp.array(angle)),
        atol=1e-14,
    )


def test_from_rotation_vector_of_zero_is_identity():
    np.testing.assert_allclose(math.from_rotation_vector(jnp.zeros(3)), IDENTITY)


@pytest.mark.parametrize("angle", ROTATION_ANGLES)
@pytest.mark.parametrize("axis", VECTORS[:3])
def test_rotation_vector_roundtrip(axis, angle):
    v = angle * axis
    np.testing.assert_allclose(
        math.to_rotation_vector(math.from_rotation_vector(v)), v, atol=1e-12
    )


def test_to_rotation_vector_of_identity_is_zero():
    np.testing.assert_allclose(math.to_rotation_vector(IDENTITY), jnp.zeros(3))


@pytest.mark.parametrize("q", QUATERNIONS)
def test_to_axis_angle_matches_to_rotation_vector(q):
    axis, angle = math.to_axis_angle(q)
    np.testing.assert_allclose(jnp.linalg.norm(axis), 1.0, atol=1e-14)
    np.testing.assert_allclose(axis * angle, math.to_rotation_vector(q), atol=1e-14)


@pytest.mark.parametrize("q", QUATERNIONS)
def test_quat_residual_of_equal_rotations_is_zero(q):
    np.testing.assert_allclose(math.quat_residual(q, q), jnp.zeros(3), atol=1e-12)


@pytest.mark.parametrize("angle", ROTATION_ANGLES)
def test_quat_residual_recovers_relative_rotation(angle):
    q2 = QUATERNIONS[0]
    v = angle * VECTORS[0]
    q1 = math.quat_mul(math.from_rotation_vector(v), q2)
    np.testing.assert_allclose(math.quat_residual(q1, q2), v, atol=1e-12)


@pytest.mark.parametrize("q", QUATERNIONS)
def test_to_rotation_vector_is_invariant_to_quaternion_sign(q):
    np.testing.assert_allclose(
        math.to_rotation_vector(-q), math.to_rotation_vector(q), atol=1e-12
    )


@pytest.mark.parametrize("q", QUATERNIONS)
def test_quat_residual_is_invariant_to_quaternion_sign(q):
    np.testing.assert_allclose(math.quat_residual(q, -q), jnp.zeros(3), atol=1e-12)


@pytest.mark.parametrize("angle", LARGE_ROTATION_ANGLES)
def test_to_rotation_vector_returns_shortest_angle(angle):
    axis = VECTORS[0]
    q = math.quat_from_axis_angle(axis, jnp.array(angle))
    np.testing.assert_allclose(
        math.to_rotation_vector(q), (angle - 2 * jnp.pi) * axis, atol=1e-12
    )


@pytest.mark.parametrize("angle", ROTATION_ANGLES + LARGE_ROTATION_ANGLES)
@pytest.mark.parametrize("axis", VECTORS[:3])
def test_to_rotation_vector_s3_inverts_from_rotation_vector(axis, angle):
    v = angle * axis
    np.testing.assert_allclose(
        math.to_rotation_vector_s3(math.from_rotation_vector(v)), v, atol=1e-12
    )


@pytest.mark.parametrize("angle", ROTATION_ANGLES)
def test_to_rotation_vector_s3_depends_on_quaternion_sign(angle):
    # -q is reached by continuing the rotation about the same axis to 2 pi - angle
    axis = VECTORS[0]
    q = math.quat_from_axis_angle(axis, jnp.array(angle))
    np.testing.assert_allclose(
        math.to_rotation_vector_s3(-q), (angle - 2 * jnp.pi) * axis, atol=1e-12
    )


def test_only_to_rotation_vector_s3_is_continuous_past_pi():
    axis = VECTORS[0]
    q_before = math.from_rotation_vector((jnp.pi - 1e-6) * axis)
    q_after = math.from_rotation_vector((jnp.pi + 1e-6) * axis)

    s3_jump = math.to_rotation_vector_s3(q_after) - math.to_rotation_vector_s3(q_before)
    np.testing.assert_allclose(s3_jump, 2e-6 * axis, atol=1e-10)

    shortest_jump = math.to_rotation_vector(q_after) - math.to_rotation_vector(q_before)
    np.testing.assert_allclose(shortest_jump, -2 * jnp.pi * axis, atol=1e-5)


# Custom JVPs


def central_difference_jacobian(f, x, eps=1e-6):
    columns = []
    for i in range(x.shape[0]):
        dx = jnp.zeros_like(x).at[i].set(eps)
        columns.append((f(x + dx) - f(x - dx)) / (2 * eps))
    return jnp.stack(columns, axis=-1)


@pytest.mark.parametrize("angle", ROTATION_ANGLES + LARGE_ROTATION_ANGLES)
@pytest.mark.parametrize("axis", VECTORS[:3])
def test_from_rotation_vector_jvp_matches_finite_differences(axis, angle):
    v = angle * axis
    np.testing.assert_allclose(
        jax.jacfwd(math.from_rotation_vector)(v),
        central_difference_jacobian(math.from_rotation_vector, v),
        atol=1e-7,
    )


def test_from_rotation_vector_jvp_at_zero():
    J = jax.jacfwd(math.from_rotation_vector)(jnp.zeros(3))
    expected = jnp.concatenate([jnp.zeros((1, 3)), 0.5 * jnp.eye(3)])
    np.testing.assert_allclose(J, expected, atol=1e-14)


@pytest.mark.parametrize(
    "to_rotation_vector",
    [math.to_rotation_vector, math.to_rotation_vector_s3],
    ids=["shortest", "s3"],
)
@pytest.mark.parametrize("angle", ROTATION_ANGLES + LARGE_ROTATION_ANGLES)
@pytest.mark.parametrize("axis", VECTORS[:3])
def test_to_rotation_vector_jvp_on_unit_sphere(axis, angle, to_rotation_vector):
    # The quaternion is only meaningful on the unit sphere, so differentiate
    # through a normalization that keeps the perturbed quaternion on it
    q = math.quat_from_axis_angle(axis, jnp.array(angle))

    def perturbed_rotation_vector(delta):
        return to_rotation_vector(math.normalize(q + delta))

    zero = jnp.zeros(4)
    np.testing.assert_allclose(
        jax.jacfwd(perturbed_rotation_vector)(zero),
        central_difference_jacobian(perturbed_rotation_vector, zero),
        atol=1e-7,
    )


@pytest.mark.parametrize("angle", ROTATION_ANGLES)
def test_quat_residual_jacobian_matches_finite_differences(angle):
    q2 = QUATERNIONS[0]
    q1 = math.quat_mul(math.from_rotation_vector(angle * VECTORS[0]), q2)

    def residual_after_perturbation(delta):
        return math.quat_residual(
            math.quat_mul(math.from_rotation_vector(delta), q1), q2
        )

    zero = jnp.zeros(3)
    np.testing.assert_allclose(
        jax.jacfwd(residual_after_perturbation)(zero),
        central_difference_jacobian(residual_after_perturbation, zero),
        atol=1e-7,
    )


def test_rotation_vector_gradients_are_finite_at_zero():
    def loss(v):
        return jnp.sum(math.to_rotation_vector(math.from_rotation_vector(v)) ** 2)

    assert jnp.all(jnp.isfinite(jax.grad(loss)(jnp.zeros(3))))


# Interpolation


def test_slerp_endpoints():
    q0, q1 = QUATERNIONS[:2]
    np.testing.assert_allclose(math.slerp(q0, q1, np.float64(0.0)), q0, atol=1e-14)
    assert_same_rotation(math.slerp(q0, q1, np.float64(1.0)), q1)


def test_slerp_midpoint_halves_the_rotation():
    axis = VECTORS[0]
    q0 = IDENTITY
    q1 = math.quat_from_axis_angle(axis, jnp.array(1.2))
    np.testing.assert_allclose(
        math.slerp(q0, q1, np.float64(0.5)),
        math.quat_from_axis_angle(axis, jnp.array(0.6)),
        atol=1e-14,
    )


def test_slerp_takes_shortest_path():
    axis = VECTORS[0]
    q1 = math.quat_from_axis_angle(axis, jnp.array(1.2))
    # -q1 is the same rotation, so interpolating towards it should give the same result
    np.testing.assert_allclose(
        math.slerp(IDENTITY, -q1, np.float64(0.5)),
        math.slerp(IDENTITY, q1, np.float64(0.5)),
        atol=1e-14,
    )


def test_slerp_of_nearly_equal_quaternions_is_unit():
    q0 = QUATERNIONS[0]
    q1 = math.normalize(q0 + 1e-10)
    result = math.slerp(q0, q1, np.float64(0.3))
    np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-14)
    assert_same_rotation(result, q0, atol=1e-9)


def test_quat_interp_matches_slerp_and_clamps():
    q0 = IDENTITY
    q1 = math.quat_from_axis_angle(Z_AXIS, jnp.array(1.0))
    q2 = math.quat_from_axis_angle(Z_AXIS, jnp.array(2.0))
    xp = [0.0, 1.0, 3.0]
    fp = jnp.stack([q0, q1, q2])

    result = math.quat_interp([-1.0, 0.5, 2.0, 5.0], xp, fp)

    expected = jnp.stack(
        [
            q0,
            math.quat_from_axis_angle(Z_AXIS, jnp.array(0.5)),
            math.quat_from_axis_angle(Z_AXIS, jnp.array(1.5)),
            q2,
        ]
    )
    np.testing.assert_allclose(result, expected, atol=1e-14)


# Euler angles


@pytest.mark.parametrize(
    "yaw, pitch, roll",
    [
        (0.0, 0.0, 0.0),
        (0.3, 0.0, 0.0),
        (0.0, -0.4, 0.0),
        (0.0, 0.0, 1.1),
        (0.7, -0.2, 2.5),
    ],
)
def test_quat_to_euler_inverts_zyx_composition(yaw, pitch, roll):
    q = math.quat_mul(
        math.quat_from_axis_angle(Z_AXIS, jnp.array(yaw)),
        math.quat_mul(
            math.quat_from_axis_angle(Y_AXIS, jnp.array(pitch)),
            math.quat_from_axis_angle(X_AXIS, jnp.array(roll)),
        ),
    )
    np.testing.assert_allclose(math.quat_to_euler(q), [yaw, pitch, roll], atol=1e-12)

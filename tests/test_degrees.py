import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ajx.math as math
from ajx.constraints import (
    ConstraintResidual,
    OneBodyConstraint,
    TwoBodyConstraint,
    TwoBodyShaftConstraint,
)
from ajx.constraints.base import DEGREES, DegreeKind, is_wrapped_degree
from ajx.definitions import (
    Configuration,
    ConstraintParameters,
    Frame,
    GeneralizedVelocity,
    RigidBodyParameters,
    State,
    Transform,
)
from ajx.param import SimulationParameters, create_parameter_node
from ajx.sensors import DegreeEncoder, OffsetParameters

DegreeTestSparseParam = create_parameter_node("DegreeTestSparseParam", ())

AXIAL_RESIDUALS = [
    ConstraintResidual.AXIAL_WORLD_SPHERICAL,
    ConstraintResidual.AXIAL_LOCAL_SPHERICAL,
]
OTHER_RESIDUALS = [ConstraintResidual.BEND_TWIST]
CONSTRAINT_TYPES = [OneBodyConstraint, TwoBodyConstraint]


def random_transform(rng):
    rot = math.normalize(jnp.array(rng.normal(size=4)))
    return Transform(jnp.array(rng.normal(size=3)), rot)


def make_param(rng):
    # Rotated frames with offsets on both bodies, so that placing along the wrong
    # axis or about the wrong point gives different degrees
    frame_a = random_transform(rng)
    frame_b = random_transform(rng)
    return SimulationParameters(
        jnp.zeros(3),
        RigidBodyParameters.concatenate(
            [
                RigidBodyParameters.create(1.0, jnp.ones(3), "a"),
                RigidBodyParameters.create(1.0, jnp.ones(3), "b"),
            ]
        ),
        ConstraintParameters.create(
            5,
            Frame(frame_a.pos, frame_a.rot),
            Frame(frame_b.pos, frame_b.rot),
            1e-8,
            0.02,
            1e8,
            "joint",
        ),
        DegreeTestSparseParam(),
    )


def make_constraint(Constraint, residual):
    if Constraint is OneBodyConstraint:
        return OneBodyConstraint(
            name="joint", body="b", constraint_residual=residual.value
        )
    return TwoBodyConstraint("joint", "a", "b", residual.value)


def make_state(body_a, body_b):
    return State(
        Configuration(
            jnp.stack([body_a.pos, body_b.pos]),
            jnp.stack([body_a.rot, body_b.rot]),
            jnp.zeros(0),
        ),
        GeneralizedVelocity(jnp.zeros((2, 6)), jnp.zeros(0)),
    )


def place(constraint, param, body_a, degrees, values):
    if isinstance(constraint, OneBodyConstraint):
        body_b = constraint.place_other(param, degrees, values)
    else:
        body_b = constraint.place_other(param, body_a, degrees, values)
    return make_state(body_a, body_b)


def random_values(rng, degrees):
    # Hinge angles within (-pi, pi), where they are not wrapped
    return jnp.array([rng.uniform(-3.0, 3.0) for _ in degrees])


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize(
    "degrees", [(), (5,), (0,), (1,), (0, 1, 2), (0, 5), (2, 5)], ids=str
)
@pytest.mark.parametrize("residual", AXIAL_RESIDUALS, ids=lambda r: r.name)
@pytest.mark.parametrize("Constraint", CONSTRAINT_TYPES, ids=lambda c: c.__name__)
def test_place_other_sets_degrees_of_axial_residuals(
    Constraint, residual, degrees, seed
):
    rng = np.random.default_rng(seed)
    param = make_param(rng)
    constraint = make_constraint(Constraint, residual)
    values = random_values(rng, degrees)
    state = place(constraint, param, random_transform(rng), degrees, values)

    expected = jnp.zeros(6).at[jnp.array(degrees, dtype=int)].set(values)
    np.testing.assert_allclose(
        constraint.object_func(state, param), expected, atol=1e-10
    )
    np.testing.assert_allclose(
        constraint.get_degree(state, param, degrees), values, atol=1e-10
    )


@pytest.mark.parametrize("degrees", [(), (0,), (0, 1, 2)], ids=str)
@pytest.mark.parametrize("residual", OTHER_RESIDUALS, ids=lambda r: r.name)
@pytest.mark.parametrize("Constraint", CONSTRAINT_TYPES, ids=lambda c: c.__name__)
def test_place_other_sets_linear_degrees_of_other_residuals(
    Constraint, residual, degrees
):
    rng = np.random.default_rng(2)
    param = make_param(rng)
    constraint = make_constraint(Constraint, residual)
    values = random_values(rng, degrees)
    state = place(constraint, param, random_transform(rng), degrees, values)

    expected = jnp.zeros(6).at[jnp.array(degrees, dtype=int)].set(values)
    np.testing.assert_allclose(
        constraint.object_func(state, param), expected, atol=1e-10
    )


@pytest.mark.parametrize(
    "residual, degree",
    [(r, 3) for r in AXIAL_RESIDUALS]
    + [(r, 4) for r in AXIAL_RESIDUALS]
    + [(r, 5) for r in OTHER_RESIDUALS],
    ids=str,
)
@pytest.mark.parametrize("Constraint", CONSTRAINT_TYPES, ids=lambda c: c.__name__)
def test_place_other_rejects_degrees_it_cannot_place(Constraint, residual, degree):
    rng = np.random.default_rng(3)
    param = make_param(rng)
    constraint = make_constraint(Constraint, residual)
    with pytest.raises(ValueError):
        place(constraint, param, random_transform(rng), degree, 0.1)


def test_get_degree_of_a_single_degree_is_a_scalar():
    rng = np.random.default_rng(4)
    param = make_param(rng)
    constraint = make_constraint(
        TwoBodyConstraint, ConstraintResidual.AXIAL_WORLD_SPHERICAL
    )
    state = place(constraint, param, random_transform(rng), 5, 0.4)
    degree = constraint.get_degree(state, param, 5)
    assert degree.shape == ()
    np.testing.assert_allclose(degree, 0.4, atol=1e-10)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("residual", list(ConstraintResidual), ids=lambda r: r.name)
def test_one_body_degrees_match_two_body_degrees_relative_to_world(residual, seed):
    # A one-body constraint is a two-body constraint whose body a is the world
    rng = np.random.default_rng(seed)
    param = make_param(rng)
    state = make_state(Transform.identity(), random_transform(rng))
    one_body = make_constraint(OneBodyConstraint, residual)
    two_body = make_constraint(TwoBodyConstraint, residual)
    np.testing.assert_allclose(
        one_body.object_func(state, param),
        two_body.object_func(state, param),
        atol=1e-12,
    )


def assert_jacobian_matches_autodiff(constraint, param, state, rows):
    # The Jacobian is the derivative of the residual with respect to the body
    # velocities, which move the bodies as Configuration.retract
    def residual_after_retract(delta):
        return constraint.object_func(
            state.replace(conf=state.conf.retract(delta)), param
        )

    expected = jax.jacfwd(residual_after_retract)(jnp.zeros(12))[rows]
    jacobian = constraint.object_jacobian(state, param)
    if isinstance(constraint, OneBodyConstraint):
        jacobian = jacobian.reshape(6, 6)[rows]
        np.testing.assert_allclose(jacobian, expected[:, 6:], atol=1e-9)
    else:
        jacobian = jacobian.reshape(2, 6, 6)[:, rows]
        np.testing.assert_allclose(jacobian[0], expected[:, :6], atol=1e-9)
        np.testing.assert_allclose(jacobian[1], expected[:, 6:], atol=1e-9)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("residual", list(ConstraintResidual), ids=lambda r: r.name)
@pytest.mark.parametrize("Constraint", CONSTRAINT_TYPES, ids=lambda c: c.__name__)
def test_jacobian_matches_autodiff(Constraint, residual, seed):
    rng = np.random.default_rng(seed)
    param = make_param(rng)
    state = make_state(random_transform(rng), random_transform(rng))
    constraint = make_constraint(Constraint, residual)
    # The Jacobian of the hinge degree is only exact when the hinge axes are
    # aligned, which random poses are not
    rows = slice(0, 5) if residual in AXIAL_RESIDUALS else slice(0, 6)
    assert_jacobian_matches_autodiff(constraint, param, state, rows)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("residual", AXIAL_RESIDUALS, ids=lambda r: r.name)
@pytest.mark.parametrize("Constraint", CONSTRAINT_TYPES, ids=lambda c: c.__name__)
def test_hinge_jacobian_matches_autodiff_when_axes_are_aligned(
    Constraint, residual, seed
):
    rng = np.random.default_rng(seed)
    param = make_param(rng)
    constraint = make_constraint(Constraint, residual)
    degrees = (0, 1, 2, 5)
    values = random_values(rng, degrees)
    state = place(constraint, param, random_transform(rng), degrees, values)
    assert_jacobian_matches_autodiff(constraint, param, state, slice(0, 6))


# Encoders measuring degrees

EncoderTestSparseParam = create_parameter_node(
    "EncoderTestSparseParam", ("offset_param",)
)


def with_offset_param(param, name, offset, scale):
    return param.replace(
        sparse_param=EncoderTestSparseParam(
            OffsetParameters((name,), jnp.array([offset]), jnp.array([scale]))
        )
    )


@pytest.mark.parametrize("degree", [0, 5])
def test_degree_encoder_observes_scaled_and_offset_degree(degree):
    rng = np.random.default_rng(5)
    param = with_offset_param(make_param(rng), "encoder", 0.1, 2.0)
    constraint = make_constraint(
        TwoBodyConstraint, ConstraintResidual.AXIAL_LOCAL_SPHERICAL
    )
    state = place(constraint, param, random_transform(rng), degree, 0.4)
    encoder = DegreeEncoder("encoder", constraint, degree)
    np.testing.assert_allclose(
        encoder.observe(state, None, param), [2.0 * 0.4 + 0.1], atol=1e-10
    )


@pytest.mark.parametrize("residual", list(ConstraintResidual), ids=lambda r: r.name)
@pytest.mark.parametrize("degree", [0, 3, 5])
def test_degree_encoder_wraps_only_hinge_degrees(residual, degree):
    constraint = make_constraint(TwoBodyConstraint, residual)
    encoder = DegreeEncoder("encoder", constraint, degree)
    is_hinge = residual in AXIAL_RESIDUALS and degree == 5
    target, prediction = jnp.array([3.0]), jnp.array([-3.0])
    expected = 6.0 - 2 * jnp.pi if is_hinge else 6.0
    np.testing.assert_allclose(encoder.residual(target, prediction), [expected])
    assert encoder.observable_names == (["theta"] if is_hinge else ["x"])


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("residual", list(ConstraintResidual), ids=lambda r: r.name)
def test_place_frame_a_puts_all_degrees_at_zero(residual, seed):
    rng = np.random.default_rng(seed)
    param = make_param(rng)
    constraint = make_constraint(OneBodyConstraint, residual)
    state = make_state(random_transform(rng), random_transform(rng))
    frame_a = constraint.place_frame_a(state, param)
    cp = param.constraint_param
    param = param.replace(
        constraint_param=cp.replace(
            frame_a=cp.frame_a.replace(
                position=cp.frame_a.position.at[0].set(frame_a.pos),
                rotation=cp.frame_a.rotation.at[0].set(frame_a.rot),
            )
        )
    )
    np.testing.assert_allclose(constraint.object_func(state, param), 0.0, atol=1e-10)


@pytest.mark.parametrize("residual", list(ConstraintResidual), ids=lambda r: r.name)
@pytest.mark.parametrize("Constraint", CONSTRAINT_TYPES, ids=lambda c: c.__name__)
def test_compute_offset_wraps_only_hinge_degrees(Constraint, residual):
    # Offsets of 6 from a target of -3 to a value of 3, which wrap to 6 - 2 pi
    offset = Constraint.compute_offset(
        jnp.full(6, 3.0), jnp.full(6, -3.0), residual.value
    )
    expected = [
        6.0 - 2 * jnp.pi if is_wrapped_degree(residual, degree) else 6.0
        for degree in range(6)
    ]
    np.testing.assert_allclose(offset, expected, atol=1e-12)


# The degree table describes the residuals


def place_frame_b(constraint, param, body_a, frame_b_rel):
    """State with frame b at the transform frame_b_rel relative to frame a"""
    cp = param.constraint_param
    if isinstance(constraint, OneBodyConstraint):
        body_a = Transform.identity()
    frame_a = body_a.multiply(Transform(cp.frame_a.position[0], cp.frame_a.rotation[0]))
    frame_b = frame_a.multiply(frame_b_rel)
    frame_b0 = Transform(cp.frame_b.position[0], cp.frame_b.rotation[0])
    return make_state(body_a, frame_b.multiply(frame_b0.inverse()))


@pytest.mark.parametrize("residual", list(ConstraintResidual), ids=lambda r: r.name)
@pytest.mark.parametrize("Constraint", CONSTRAINT_TYPES, ids=lambda c: c.__name__)
def test_linear_degrees_are_components_of_frame_offset(Constraint, residual):
    rng = np.random.default_rng(6)
    param = make_param(rng)
    constraint = make_constraint(Constraint, residual)
    # A frame b with offset and rotation relative to frame a
    state = place_frame_b(
        constraint,
        param,
        random_transform(rng),
        Transform(
            jnp.array(rng.normal(size=3)),
            math.from_rotation_vector(jnp.array(rng.normal(size=3)) * 0.3),
        ),
    )
    cp = param.constraint_param
    body_a = Transform(state.conf.pos[0], state.conf.rot[0])
    if Constraint is OneBodyConstraint:
        body_a = Transform.identity()
    body_b = Transform(state.conf.pos[1], state.conf.rot[1])
    frame_a = body_a.multiply(Transform(cp.frame_a.position[0], cp.frame_a.rotation[0]))
    frame_b = body_b.multiply(Transform(cp.frame_b.position[0], cp.frame_b.rotation[0]))
    offset = frame_a.pos - frame_b.pos
    if residual != ConstraintResidual.AXIAL_WORLD_SPHERICAL:
        offset = math.rotate_vector(math.conjugate(frame_a.rot), offset)

    values = constraint.object_func(state, param)
    for i, degree in enumerate(DEGREES[residual]):
        if degree.kind == DegreeKind.LINEAR:
            np.testing.assert_allclose(values[i], offset[degree.axis], atol=1e-10)


@pytest.mark.parametrize("angle", [-1.2, -0.4, 0.4, 1.2])
@pytest.mark.parametrize("residual", list(ConstraintResidual), ids=lambda r: r.name)
@pytest.mark.parametrize("Constraint", CONSTRAINT_TYPES, ids=lambda c: c.__name__)
def test_rotational_degrees_measure_rotations_about_their_axes(
    Constraint, residual, angle
):
    rng = np.random.default_rng(7)
    param = make_param(rng)
    constraint = make_constraint(Constraint, residual)
    body_a = random_transform(rng)
    for i, degree in enumerate(DEGREES[residual]):
        if degree.kind == DegreeKind.LINEAR:
            continue
        rotation = math.from_rotation_vector(jnp.zeros(3).at[degree.axis].set(angle))
        state = place_frame_b(
            constraint, param, body_a, Transform(jnp.zeros(3), rotation)
        )
        value = degree.sign * (
            angle if degree.kind == DegreeKind.ANGLE else jnp.sin(angle)
        )
        # A rotation about the axis of one degree leaves all other degrees at zero
        np.testing.assert_allclose(
            constraint.object_func(state, param),
            jnp.zeros(6).at[i].set(value),
            atol=1e-10,
        )


@pytest.mark.parametrize("angle", [-3.0, -2.0, 2.0, 3.0])
@pytest.mark.parametrize("residual", AXIAL_RESIDUALS, ids=lambda r: r.name)
def test_wrapping_hinge_degrees_measure_angles_up_to_pi(residual, angle):
    rng = np.random.default_rng(8)
    param = make_param(rng)
    constraint = make_constraint(TwoBodyConstraint, residual)
    rotation = math.from_rotation_vector(jnp.array([angle, 0.0, 0.0]))
    state = place_frame_b(
        constraint, param, random_transform(rng), Transform(jnp.zeros(3), rotation)
    )
    assert DEGREES[residual][5].wraps
    np.testing.assert_allclose(
        constraint.get_degree(state, param, 5), angle, atol=1e-10
    )


@pytest.mark.parametrize("residual", list(ConstraintResidual), ids=lambda r: r.name)
@pytest.mark.parametrize("Constraint", CONSTRAINT_TYPES, ids=lambda c: c.__name__)
def test_multiplier_names_are_degree_names(Constraint, residual):
    constraint = make_constraint(Constraint, residual)
    names = constraint.get_multiplier_names()
    assert names == tuple(degree.name for degree in DEGREES[residual])
    assert len(names) == Constraint.get_constrained_degrees()


def test_shaft_constraint_names_its_single_multiplier():
    parent = make_constraint(
        TwoBodyConstraint, ConstraintResidual.AXIAL_WORLD_SPHERICAL
    )
    shaft = TwoBodyShaftConstraint("shaft", parent, "shaft_body")
    assert len(shaft.get_multiplier_names()) == shaft.get_constrained_degrees()

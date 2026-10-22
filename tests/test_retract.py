import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ajx.math as math
from ajx.definitions import (
    ConstraintParameters,
    Configuration,
    Frame,
    Frames,
    RigidBodyParameters,
    Transform,
)
from ajx.param import create_parameter_node


def random_configuration(n_bodies, n_scalars, seed=0):
    rng = np.random.default_rng(seed)
    pos = jnp.array(rng.normal(size=(n_bodies, 3)))
    rot = jax.vmap(math.normalize)(jnp.array(rng.normal(size=(n_bodies, 4))))
    scalar = jnp.array(rng.normal(size=(n_scalars,)))
    return Configuration(pos, rot, scalar)


def random_tangent(size, seed=1, scale=0.3):
    rng = np.random.default_rng(seed)
    return jnp.array(scale * rng.normal(size=(size,)))


@pytest.mark.parametrize("n_bodies, n_scalars", [(1, 0), (3, 0), (3, 2), (0, 2)])
def test_configuration_log_map_inverts_retract(n_bodies, n_scalars):
    conf = random_configuration(n_bodies, n_scalars)
    delta = random_tangent(conf.tangent_size())
    np.testing.assert_allclose(conf.retract(delta).log_map(conf), delta, atol=1e-10)


@pytest.mark.parametrize("n_bodies, n_scalars", [(1, 0), (3, 2)])
def test_configuration_retract_inverts_log_map(n_bodies, n_scalars):
    conf = random_configuration(n_bodies, n_scalars, seed=0)
    other = conf.retract(random_tangent(conf.tangent_size(), seed=2))
    restored = conf.retract(other.log_map(conf))
    np.testing.assert_allclose(restored.pos, other.pos, atol=1e-10)
    np.testing.assert_allclose(restored.rot, other.rot, atol=1e-10)
    np.testing.assert_allclose(restored.scalar, other.scalar, atol=1e-10)


def test_configuration_log_map_matches_transform_log_map_per_body():
    conf = random_configuration(3, 0)
    other = conf.retract(random_tangent(conf.tangent_size()))
    per_body = jax.vmap(Transform.log_map)(
        other.as_vectorized_transform(), conf.as_vectorized_transform()
    )
    np.testing.assert_allclose(other.log_map(conf), per_body.reshape(-1), atol=1e-12)


def two_body_parameters():
    return RigidBodyParameters.concatenate(
        [
            RigidBodyParameters.create(1.0, jnp.array([1.0, 2.0, 3.0]), "b0"),
            RigidBodyParameters.create(2.0, jnp.array([4.0, 5.0, 6.0]), "b1"),
        ]
    )


def test_tree_retract_adds_to_named_component():
    param = two_body_parameters()
    new = param.tree_retract({"mass": {"b1": jnp.array(0.1)}})
    np.testing.assert_allclose(new.mass, [1.0, 2.1])


def test_tree_retract_adds_to_indexed_component_entry():
    param = two_body_parameters()
    new = param.tree_retract({"inertia": {"b0": {0: 0.5}}})
    np.testing.assert_allclose(new.inertia[0], [1.5, 0.0, 0.0, 2.0, 0.0, 3.0])
    np.testing.assert_allclose(new.inertia[1], param.inertia[1])


def test_tree_retract_adds_to_whole_array():
    param = two_body_parameters()
    new = param.tree_retract({"mass": jnp.array([0.1, 0.2])})
    np.testing.assert_allclose(new.mass, [1.1, 2.2])


def test_tree_retract_resolves_second_axis_names():
    param = two_body_parameters()
    new = param.tree_retract({"inertia": {"b1": {"xx": 1.0, "zz": -1.0}}})
    np.testing.assert_allclose(new.inertia[1], [5.0, 0.0, 0.0, 5.0, 0.0, 5.0])
    np.testing.assert_allclose(new.inertia[0], param.inertia[0])


BodiesNode = create_parameter_node("BodiesNode", ("bodies",))


@pytest.mark.parametrize(
    "delta",
    [
        {"bodies": {"inertia": {"b1": {"xx": 1.0}}}},
        {"bodies.inertia.b1.xx": 1.0},
        {("bodies", "inertia", "b1", "xx"): 1.0},
    ],
    ids=["nested", "dotted", "tuple"],
)
def test_tree_retract_accepts_paths(delta):
    node = BodiesNode(bodies=two_body_parameters())
    new = node.tree_retract(delta)
    np.testing.assert_allclose(new.bodies.inertia[1, 0], 5.0)
    np.testing.assert_allclose(new.bodies.inertia[0], node.bodies.inertia[0])


@pytest.mark.parametrize(
    "path, current", [("bodies.inertia.b1.xx", 4.0), ("bodies.mass.b0", 1.0)]
)
def test_tree_retract_matches_tree_replace_with_added_value(path, current):
    node = BodiesNode(bodies=two_body_parameters())
    retracted = node.tree_retract({path: 0.25})
    replaced = node.tree_replace({path: current + 0.25})
    np.testing.assert_allclose(retracted.bodies.flatten(), replaced.bodies.flatten())


def random_frames(n_frames, seed=0):
    conf = random_configuration(n_frames, 0, seed)
    return Frames(conf.pos, conf.rot)


def frame_at(frames, i):
    return Frame(frames.position[i], frames.rotation[i])


def test_frame_retract_matches_transform():
    frame = frame_at(random_frames(1), 0)
    delta = random_tangent(6)
    new = frame.retract(delta)
    expected = Transform(frame.position, frame.rotation).retract(delta)
    assert frame.tangent_size() == 6
    np.testing.assert_allclose(new.position, expected.pos, atol=1e-12)
    np.testing.assert_allclose(new.rotation, expected.rot, atol=1e-12)
    np.testing.assert_allclose(new.log_map(frame), delta, atol=1e-10)


@pytest.mark.parametrize("n_frames", [1, 3])
def test_frames_log_map_inverts_retract(n_frames):
    frames = random_frames(n_frames)
    delta = random_tangent(frames.tangent_size())
    new = frames.retract(delta)
    assert frames.tangent_size() == 6 * n_frames
    np.testing.assert_allclose(jnp.linalg.norm(new.rotation, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(new.log_map(frames), delta, atol=1e-10)


def test_frames_retract_applies_per_frame():
    frames = random_frames(3)
    delta = random_tangent(frames.tangent_size())
    new = frames.retract(delta)
    for i in range(3):
        expected = frame_at(frames, i).as_transform().retract(delta[6 * i : 6 * i + 6])
        np.testing.assert_allclose(new.position[i], expected.pos, atol=1e-12)
        np.testing.assert_allclose(new.rotation[i], expected.rot, atol=1e-12)


def test_tree_retract_on_constraint_frames_stays_on_manifold():
    frame = Frame(jnp.zeros(3), jnp.array([1.0, 0.0, 0.0, 0.0]))
    param = ConstraintParameters.create_locked(frame, frame, 1e-8, 1e-8, 0.0, 0.0, "c")
    new = param.tree_retract({"frame_a": jnp.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.2])})
    np.testing.assert_allclose(new.frame_a.position, [[0.1, 0.0, 0.0]], atol=1e-12)
    np.testing.assert_allclose(
        new.frame_a.rotation, [[jnp.cos(0.1), 0.0, 0.0, jnp.sin(0.1)]], atol=1e-12
    )

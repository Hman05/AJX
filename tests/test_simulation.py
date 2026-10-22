from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ajx.math as math
from ajx.constraints import ConstraintResidual, GearConstraint, TwoBodyConstraint
from ajx.definitions import (
    Configuration,
    ConstraintParameters,
    Frame,
    GeneralizedVelocity,
    RigidBodyParameters,
    ScalarBody,
    ScalarBodyParameters,
    ScalarConstraintParameters,
    State,
    Transform,
)
from ajx.example_environments import CartPole, FreeBody, Furuta, Pendulum
from ajx.example_environments.dlo_attached import DLOAttached, DLOAttachedSettings
from ajx.example_environments.double_pendulum import (
    DoublePendulum,
    DoublePendulumSettings,
)
from ajx.param import SimulationParameters, create_parameter_node
from ajx.simulation import Simulation, SimulationSettings, Solver

TIMESTEP = 0.01
AGREEMENT_STEPS = 20
PGS_ITERATIONS = 200


# Environments


def make_pendulum(settings):
    env = Pendulum(settings, has_quadratic_damping=False)
    return env, env.state_from_angle(1.0, env.default_param), None


def make_cartpole(settings):
    env = CartPole(settings)
    return env, env.state_from_angles(0.0, 1.0, env.default_param), jnp.zeros(1)


def make_furuta(settings):
    env = Furuta(settings)
    return env, env.state_from_angles(0.3, 1.0, env.default_param), jnp.zeros(1)


def make_free_body(settings):
    env = FreeBody(settings)
    return env, env.state_from_angular_velocity(jnp.array([0.1, 2.0, 0.3])), None


def make_double_pendulum(settings):
    env = DoublePendulum(settings, DoublePendulumSettings())
    return env, env.state_from_angles(1.0, 0.5, env.default_param), jnp.zeros(1)


def make_dlo_attached(settings):
    env = DLOAttached(
        settings,
        DLOAttachedSettings(
            n_bodies=5,
            body_length=0.05,
            mass_density=1000.0,
            constraint_residual=ConstraintResidual.BEND_TWIST.value,
            hinge_motor_attachment=False,
        ),
    )
    return env, env.state_from_angles(env.default_param), jnp.zeros(2)


ENVIRONMENTS = {
    "pendulum": make_pendulum,
    "cartpole": make_cartpole,
    "furuta": make_furuta,
    "free_body": make_free_body,
    "double_pendulum": make_double_pendulum,
    "dlo_attached": make_dlo_attached,
}


@lru_cache(maxsize=None)
def rollout(env_name, solver, ldlt_refinement_steps=0, n_steps=AGREEMENT_STEPS):
    settings = SimulationSettings(
        TIMESTEP,
        use_gyroscopic=True,
        solver=solver,
        pgs_iterations=PGS_ITERATIONS,
        ldlt_refinement_steps=ldlt_refinement_steps,
    )
    env, state, control = ENVIRONMENTS[env_name](settings)
    for _ in range(n_steps):
        state, _ = env.step(state, control, env.default_param)
    return state


# Solver agreement


SOLVER_CASES = [
    pytest.param(Solver.SPARSE_LINEAR, 0, 1e-10, id="sparse_linear"),
    pytest.param(Solver.SPARSE_LINEAR, 2, 1e-10, id="sparse_linear_refined"),
    pytest.param(Solver.DENSE_PGS, 0, 1e-5, id="dense_pgs"),
    pytest.param(Solver.SPARSE_PGS, 0, 1e-5, id="sparse_pgs"),
]


@pytest.mark.parametrize("solver, refinement_steps, atol", SOLVER_CASES)
@pytest.mark.parametrize("env_name", ENVIRONMENTS)
def test_solver_agrees_with_dense_linear(env_name, solver, refinement_steps, atol):
    reference = rollout(env_name, Solver.DENSE_LINEAR)
    state = rollout(env_name, solver, refinement_steps)

    np.testing.assert_allclose(state.gvel.data, reference.gvel.data, atol=atol)
    np.testing.assert_allclose(state.conf.pos, reference.conf.pos, atol=atol)
    np.testing.assert_allclose(state.conf.rot, reference.conf.rot, atol=atol)


@pytest.mark.parametrize("env_name", ENVIRONMENTS)
def test_quaternions_stay_normalized(env_name):
    state = rollout(env_name, Solver.DENSE_LINEAR)
    np.testing.assert_allclose(jnp.linalg.norm(state.conf.rot, axis=1), 1.0, atol=1e-12)


# Physical behaviour


def world_inertia(param, rot, body=0):
    R = math.rotation_matrix(rot)
    inertia = jax.vmap(lambda rb: rb.get_inertia_matrix())(param.rigid_body_param)
    return R @ inertia[body] @ R.T


def single_body_energy(state, param):
    vel, ang = state.gvel.data[0, :3], state.gvel.data[0, 3:]
    mass = param.rigid_body_param.mass[0]
    kinetic = 0.5 * mass * vel @ vel
    kinetic += 0.5 * ang @ world_inertia(param, state.conf.rot[0]) @ ang
    potential = -mass * param.gravity @ state.conf.pos[0]
    return kinetic + potential


def free_body_state(pos, rot, vel, ang):
    return State(
        Configuration(pos[None], rot[None]),
        GeneralizedVelocity(jnp.concatenate([vel, ang])[None]),
    )


@pytest.mark.parametrize("solver", [Solver.DENSE_LINEAR, Solver.SPARSE_LINEAR])
def test_free_fall_matches_symplectic_euler(solver):
    env = FreeBody(SimulationSettings(TIMESTEP, solver=solver))
    gravity = jnp.array([0.0, -9.82, 0.0])
    param = env.default_param.tree_replace({"gravity": gravity})
    pos0 = jnp.array([0.3, 0.0, 0.1])
    vel0 = jnp.array([1.0, 2.0, -0.5])
    state = free_body_state(pos0, math.Rotations.identity, vel0, jnp.zeros(3))

    n = 50
    for _ in range(n):
        state, _ = env.step(state, None, param)

    # v_{k+1} = v_k + h g and x_{k+1} = x_k + h v_{k+1}
    h = TIMESTEP
    expected_vel = vel0 + n * h * gravity
    expected_pos = pos0 + n * h * vel0 + h**2 * gravity * n * (n + 1) / 2
    np.testing.assert_allclose(state.gvel.data[0, :3], expected_vel, atol=1e-12)
    np.testing.assert_allclose(state.conf.pos[0], expected_pos, atol=1e-12)


@pytest.mark.parametrize("use_gyroscopic", [False, True])
def test_spin_about_principal_axis_is_steady(use_gyroscopic):
    env = FreeBody(SimulationSettings(TIMESTEP, use_gyroscopic=use_gyroscopic))
    ang0 = jnp.array([0.0, 0.0, 2.0])
    state = env.state_from_angular_velocity(ang0)
    for _ in range(100):
        state, _ = env.step(state, None, env.default_param)
    np.testing.assert_allclose(state.gvel.data[0, 3:], ang0, atol=1e-12)


def test_angular_velocity_is_constant_without_gyroscopic_term():
    env = FreeBody(SimulationSettings(TIMESTEP, use_gyroscopic=False))
    ang0 = jnp.array([0.1, 0.2, 2.0])
    state = env.state_from_angular_velocity(ang0)
    for _ in range(100):
        state, _ = env.step(state, None, env.default_param)
    np.testing.assert_allclose(state.gvel.data[0, 3:], ang0, atol=1e-12)


def test_angular_momentum_drift_is_first_order_in_timestep():
    def momentum_drift(h):
        env = FreeBody(SimulationSettings(h, use_gyroscopic=True))
        param = env.default_param
        state = env.state_from_angular_velocity(jnp.array([0.1, 0.2, 2.0]))

        def momentum(state):
            return world_inertia(param, state.conf.rot[0]) @ state.gvel.data[0, 3:]

        momentum0 = momentum(state)
        for _ in range(int(round(2.0 / h))):
            state, _ = env.step(state, None, param)
        return jnp.linalg.norm(momentum(state) - momentum0) / jnp.linalg.norm(momentum0)

    drift_coarse = momentum_drift(0.01)
    drift_fine = momentum_drift(0.005)
    assert drift_coarse < 1e-3
    assert 1.8 < drift_coarse / drift_fine < 2.2


def pendulum_pivot_error(state):
    # The hinge attaches the world origin to this point on the pendulum
    attachment = jnp.array([0.0, 0.091, 0.0])
    return jnp.linalg.norm(
        state.conf.pos[0] + math.rotate_vector(state.conf.rot[0], attachment)
    )


def test_hanging_pendulum_stays_at_rest():
    env = Pendulum(SimulationSettings(TIMESTEP), has_quadratic_damping=False)
    param = env.default_param
    state = env.state_from_angle(0.0, param)
    for _ in range(300):
        state, _ = env.step(state, None, param)
    assert jnp.abs(state.gvel.data).max() < 1e-10
    assert pendulum_pivot_error(state) < 1e-4


def test_swinging_pendulum_keeps_pivot_and_does_not_gain_energy():
    env = Pendulum(SimulationSettings(TIMESTEP), has_quadratic_damping=False)
    param = env.default_param
    state = env.state_from_angle(1.0, param)
    energy0 = single_body_energy(state, param)

    energies, pivot_errors = [], []
    for _ in range(300):
        state, _ = env.step(state, None, param)
        energies.append(single_body_energy(state, param))
        pivot_errors.append(pendulum_pivot_error(state))

    assert max(pivot_errors) < 1e-3
    # Energy is dissipated by the viscous hinge, but compliant constraints let it
    # fluctuate slightly from step to step
    assert max(energies) < energy0 + 1e-4
    assert energies[-1] < energy0


# Gear between two scalar bodies

GearSparseParam = create_parameter_node("GearSparseParam", ())


def make_gear_system(solver, gear_ratio, inertia_a, inertia_b, ang0):
    simulation = Simulation(
        SimulationSettings(TIMESTEP, solver=solver),
        (),
        (GearConstraint("gear", "shaft_a", "shaft_b"),),
        scalar_body_list=(ScalarBody("shaft_a"), ScalarBody("shaft_b")),
    )
    param = SimulationParameters(
        jnp.zeros(3),
        RigidBodyParameters.create_empty(),
        ConstraintParameters.create_empty(),
        GearSparseParam(),
        scalar_body_param=ScalarBodyParameters.concatenate(
            [
                ScalarBodyParameters.create(inertia_a, True, "shaft_a"),
                ScalarBodyParameters.create(inertia_b, True, "shaft_b"),
            ]
        ),
        scalar_constraint_param=ScalarConstraintParameters.create(
            0.0, 0.0, gear_ratio, True, 1e-8, 2 * TIMESTEP, "gear"
        ),
    )
    state = State(
        Configuration(jnp.zeros((0, 3)), jnp.zeros((0, 4)), jnp.zeros(2)),
        GeneralizedVelocity(jnp.zeros((0, 6)), ang0),
        multipliers=jnp.zeros(1),
        residual=jnp.zeros(1),
    )
    return simulation, param, state


@pytest.mark.parametrize(
    "solver",
    [Solver.DENSE_LINEAR, Solver.SPARSE_LINEAR, Solver.DENSE_PGS, Solver.SPARSE_PGS],
    ids=lambda s: s.name.lower(),
)
def test_gear_couples_shafts_and_conserves_momentum(solver):
    gear_ratio, inertia_a, inertia_b = 2.0, 1.0, 2.0
    ang0 = jnp.array([1.0, -0.5])
    simulation, param, state = make_gear_system(
        solver, gear_ratio, inertia_a, inertia_b, ang0
    )
    for _ in range(100):
        state, ((gvel_next, multipliers, residual), _) = simulation.pre_step(
            state, jnp.zeros(0), param
        )
        state = simulation.post_step(state, gvel_next, multipliers, residual)

    # The gear impulse acts as +lambda on shaft a and -r lambda on shaft b, so
    # I_a w_a + I_b w_b / r is conserved, and w_a = r w_b once engaged
    momentum = inertia_a * ang0[0] + inertia_b * ang0[1] / gear_ratio
    ang_b = momentum / (inertia_a * gear_ratio + inertia_b / gear_ratio)
    np.testing.assert_allclose(
        state.gvel.scalar, [gear_ratio * ang_b, ang_b], atol=1e-6
    )


def test_gear_jacobian_matches_autodiff():
    simulation, param, state = make_gear_system(
        Solver.DENSE_LINEAR, 2.0, 1.0, 2.0, jnp.zeros(2)
    )
    state = state.replace(conf=state.conf.replace(scalar=jnp.array([0.3, -0.2])))

    def residual(scalar):
        conf = state.conf.replace(scalar=scalar)
        return GearConstraint.func(param, state.replace(conf=conf), (0, 1), (0,), 0)

    np.testing.assert_allclose(
        GearConstraint.jacobian(param, state, (0, 1), (0,), 0),
        jax.jacfwd(residual)(state.conf.scalar)[0],
    )


# Placing bodies from free degrees

PlacementSparseParam = create_parameter_node("PlacementSparseParam", ())


@pytest.mark.parametrize("x", [-0.4, 0.0, 0.7])
def test_prismatic_place_other_matches_free_degree(x):
    # A rotated joint frame on a rotated parent body, so that a placement along
    # the wrong axis would be measured as a different displacement
    frame_rot = math.quat_from_axis_angle(jnp.array([0.3, -0.5, 0.8]) / jnp.sqrt(0.98), 1.1)
    prismatic = TwoBodyConstraint(
        name="prismatic",
        body_a="a",
        body_b="b",
        constraint_residual=ConstraintResidual.AXIAL_LOCAL_SPHERICAL.value,
    )
    param = SimulationParameters(
        jnp.zeros(3),
        RigidBodyParameters.concatenate(
            [
                RigidBodyParameters.create(1.0, jnp.ones(3), "a"),
                RigidBodyParameters.create(1.0, jnp.ones(3), "b"),
            ]
        ),
        ConstraintParameters.create(
            0,
            Frame(jnp.array([0.2, -0.1, 0.4]), frame_rot),
            Frame(jnp.array([0.0, 0.3, -0.2]), frame_rot),
            1e-8,
            2 * TIMESTEP,
            1e8,
            "prismatic",
        ),
        PlacementSparseParam(),
    )
    body_a = Transform(
        jnp.array([1.0, 2.0, -0.5]),
        math.quat_from_axis_angle(jnp.array([-0.6, 0.2, 0.4]) / jnp.sqrt(0.56), 0.9),
    )
    body_b = prismatic.place_other(0, param, body_a, x)
    state = State(
        Configuration(
            jnp.stack([body_a.pos, body_b.pos]),
            jnp.stack([body_a.rot, body_b.rot]),
            jnp.zeros(0),
        ),
        GeneralizedVelocity(jnp.zeros((2, 6)), jnp.zeros(0)),
    )
    np.testing.assert_allclose(prismatic.get_free_degrees(state, param), x, atol=1e-12)

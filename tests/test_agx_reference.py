"""
Compare AJX against AGX Dynamics, which AJX is modelled on.

Both use SPOOK with the same compliance and damping parameters, but the joints and
solvers are not implemented identically, so the trajectory tests compare with
tolerances and check that the difference vanishes as the timestep shrinks.

AGX does not apply gyroscopic forces to rigid bodies (a free body keeps a constant
angular velocity), so only simulations where gyroscopic forces vanish or are
disabled in AJX are compared.

The tests are skipped unless AGX can be imported, e.g. after running
    source /opt/Algoryx/AGX-<version>/setup_env.bash
"""

from functools import lru_cache

import jax.numpy as jnp
import numpy as np
import pytest

agx = pytest.importorskip("agx")
agxSDK = pytest.importorskip("agxSDK")
agxDriveTrain = pytest.importorskip("agxDriveTrain")
agxPowerLine = pytest.importorskip("agxPowerLine")

import ajx.math as math
from ajx.constraints import (
    ConstraintResidual,
    GearConstraint,
    OneBodyConstraint,
    TwoBodyConstraint,
)
from ajx.definitions import (
    ConstraintParameters,
    Configuration,
    Frame,
    GeneralizedVelocity,
    RigidBody,
    RigidBodyParameters,
    ScalarBody,
    ScalarBodyParameters,
    ScalarConstraintParameters,
    State,
)
from ajx.example_environments import FreeBody
from ajx.param import SimulationParameters, create_parameter_node
from ajx.simulation import Simulation, SimulationSettings

AGX_INIT = agx.AutoInit()

AGXReferenceSparseParam = create_parameter_node("AGXReferenceSparseParam", ())

GRAVITY = jnp.array([0.0, -9.82, 0.0])
X_AXIS = jnp.array([1.0, 0.0, 0.0])

# Links of a planar pendulum chain swinging about the x-axis
LINK_LENGTH = 0.5
LINK_MASS = 1.0
LINK_INERTIA = jnp.array([0.03, 0.002, 0.03])
LINK_TOP = jnp.array([0.0, 0.5 * LINK_LENGTH, 0.0])
LINK_BOTTOM = jnp.array([0.0, -0.5 * LINK_LENGTH, 0.0])


# AGX helpers


def to_agx_vec3(v):
    return agx.Vec3(*(float(x) for x in v))


def agx_simulation(timestep, gravity=GRAVITY):
    simulation = agxSDK.Simulation()
    simulation.setTimeStep(timestep)
    simulation.setUniformGravity(to_agx_vec3(gravity))
    return simulation


def agx_rigid_body(
    simulation, mass, inertia_diag, pos, rot, vel=jnp.zeros(3), ang=jnp.zeros(3)
):
    body = agx.RigidBody()
    mass_properties = body.getMassProperties()
    mass_properties.setAutoGenerateMask(0)
    mass_properties.setMass(float(mass))
    mass_properties.setInertiaTensor(to_agx_vec3(inertia_diag))
    body.setPosition(to_agx_vec3(pos))
    # AJX stores quaternions as [w, x, y, z], agx.Quat takes (x, y, z, w)
    body.setRotation(agx.Quat(*(float(x) for x in rot[1:]), float(rot[0])))
    body.setVelocity(to_agx_vec3(vel))
    body.setAngularVelocity(to_agx_vec3(ang))
    simulation.add(body)
    return body


def agx_position(body):
    p = body.getPosition()
    return np.array([p.x(), p.y(), p.z()])


def agx_rotation(body):
    q = body.getRotation()
    return np.array([q.w(), q.x(), q.y(), q.z()])


# AJX helpers


def ajx_step(simulation, state, param):
    state, ((gvel_next, multipliers, residual), _) = simulation.pre_step(
        state, jnp.zeros(0), param
    )
    return simulation.post_step(state, gvel_next, multipliers, residual)


# Pendulum chain built in both simulators


def build_pendulum_chain(n_links, timestep, compliance, viscous_damping, angles):
    """
    A chain of links hanging from the world origin, connected by hinges about the
    x-axis. Each hinge has the given compliance and SPOOK damping 2h on its locked
    degrees of freedom. If viscous_damping is not None, the free hinge rotation is
    damped by a torque -viscous_damping * angular velocity.
    """
    # To place the links at the given absolute angles, starting from the origin
    positions, rotations, pivots = [], [], []
    pivot = jnp.zeros(3)
    for angle in angles[:n_links]:
        rot = math.quat_from_axis_angle(X_AXIS, jnp.array(angle))
        pos = pivot - math.rotate_vector(rot, LINK_TOP)
        pivots.append(pivot)
        positions.append(pos)
        rotations.append(rot)
        pivot = pos + math.rotate_vector(rot, LINK_BOTTOM)

    # AJX models viscous damping as compliance 1/b on the free hinge velocity
    b = 1e-12 if viscous_damping is None else viscous_damping
    names = [f"link{i}" for i in range(n_links)]
    constraints, constraint_params = [], []
    for i in range(n_links):
        if i == 0:
            constraint = OneBodyConstraint(
                name="hinge0",
                body=names[0],
                constraint_residual=ConstraintResidual.AXIAL_WORLD_SPHERICAL.value,
            )
            frame_a = Frame(jnp.zeros(3), math.Rotations.identity)
        else:
            constraint = TwoBodyConstraint(
                f"hinge{i}",
                names[i - 1],
                names[i],
                ConstraintResidual.AXIAL_WORLD_SPHERICAL.value,
            )
            frame_a = Frame(LINK_BOTTOM, math.Rotations.identity)
        constraints.append(constraint)
        constraint_params.append(
            ConstraintParameters.create(
                free_degree=5,
                frame_a=frame_a,
                frame_b=Frame(LINK_TOP, math.Rotations.identity),
                compliance=compliance,
                damping=2 * timestep,
                b=b,
                name=f"hinge{i}",
            )
        )

    ajx_sim = Simulation(
        SimulationSettings(timestep),
        tuple(RigidBody(name, ()) for name in names),
        tuple(constraints),
    )
    ajx_param = SimulationParameters(
        GRAVITY,
        RigidBodyParameters.concatenate(
            [RigidBodyParameters.create(LINK_MASS, LINK_INERTIA, n) for n in names]
        ),
        ConstraintParameters.concatenate(constraint_params),
        AGXReferenceSparseParam(),
    )
    ajx_state = State(
        Configuration(jnp.stack(positions), jnp.stack(rotations)),
        GeneralizedVelocity(jnp.zeros((n_links, 6))),
        multipliers=jnp.zeros(6 * n_links),
        residual=jnp.zeros(6 * n_links),
    )

    agx_sim = agx_simulation(timestep)
    agx_bodies = [
        agx_rigid_body(agx_sim, LINK_MASS, LINK_INERTIA, pos, rot)
        for pos, rot in zip(positions, rotations)
    ]
    for i in range(n_links):
        hinge_frame = agx.HingeFrame()
        hinge_frame.setAxis(to_agx_vec3(X_AXIS))
        hinge_frame.setCenter(to_agx_vec3(pivots[i]))
        if i == 0:
            hinge = agx.Hinge(hinge_frame, agx_bodies[0])
        else:
            hinge = agx.Hinge(hinge_frame, agx_bodies[i - 1], agx_bodies[i])
        hinge.setCompliance(compliance)
        hinge.setDamping(2 * timestep)
        if viscous_damping is not None:
            motor = hinge.getMotor1D()
            motor.setEnable(True)
            motor.setSpeed(0.0)
            motor.setCompliance(1.0 / viscous_damping)
        agx_sim.add(hinge)

    return ajx_sim, ajx_param, ajx_state, agx_sim, agx_bodies


@lru_cache(maxsize=None)
def max_chain_position_difference(
    n_links,
    timestep,
    duration=2.0,
    compliance=1e-8,
    viscous_damping=None,
    angles=(1.0, 0.5),
):
    ajx_sim, ajx_param, ajx_state, agx_sim, agx_bodies = build_pendulum_chain(
        n_links, timestep, compliance, viscous_damping, angles
    )
    max_difference = 0.0
    for _ in range(int(round(duration / timestep))):
        ajx_state = ajx_step(ajx_sim, ajx_state, ajx_param)
        agx_sim.stepForward()
        agx_positions = np.stack([agx_position(body) for body in agx_bodies])
        difference = np.abs(agx_positions - np.array(ajx_state.conf.pos)).max()
        max_difference = max(max_difference, difference)
    return max_difference


# Tests


@pytest.mark.parametrize("viscous_damping", [None, 0.01], ids=["free", "damped"])
@pytest.mark.parametrize("n_links", [1, 2], ids=["pendulum", "double_pendulum"])
def test_pendulum_chain_matches_agx(n_links, viscous_damping):
    difference = max_chain_position_difference(
        n_links, 0.01, viscous_damping=viscous_damping
    )
    assert difference < 1e-3


@pytest.mark.parametrize("n_links", [1, 2], ids=["pendulum", "double_pendulum"])
def test_pendulum_chain_difference_to_agx_is_second_order(n_links):
    coarse = max_chain_position_difference(n_links, 0.01)
    fine = max_chain_position_difference(n_links, 0.005)
    assert coarse / fine > 3.0


def test_hanging_pendulum_deflection_matches_agx():
    compliance = 1e-4
    ajx_sim, ajx_param, ajx_state, agx_sim, agx_bodies = build_pendulum_chain(
        1, 0.01, compliance, viscous_damping=0.01, angles=(0.0,)
    )
    for _ in range(300):
        ajx_state = ajx_step(ajx_sim, ajx_state, ajx_param)
        agx_sim.stepForward()

    rest_height = -0.5 * LINK_LENGTH
    ajx_deflection = rest_height - ajx_state.conf.pos[0, 1]
    agx_deflection = rest_height - agx_position(agx_bodies[0])[1]

    expected = LINK_MASS * -GRAVITY[1] * compliance
    np.testing.assert_allclose(ajx_deflection, expected, rtol=1e-9)
    np.testing.assert_allclose(agx_deflection, expected, rtol=1e-9)


def test_free_flight_matches_agx():
    timestep = 0.01
    vel0 = jnp.array([1.0, 2.0, -0.5])
    ang0 = jnp.array([0.3, -1.0, 2.0])
    inertia = jnp.array([1.0, 2.0, 4.0])

    env = FreeBody(SimulationSettings(timestep, use_gyroscopic=False))
    param = env.default_param.tree_replace({"gravity": GRAVITY})
    state = State(
        Configuration(jnp.zeros((1, 3)), math.Rotations.identity[None]),
        GeneralizedVelocity(jnp.concatenate([vel0, ang0])[None]),
    )

    agx_sim = agx_simulation(timestep)
    body = agx_rigid_body(
        agx_sim, 1.0, inertia, jnp.zeros(3), math.Rotations.identity, vel0, ang0
    )

    for _ in range(100):
        state, _ = env.step(state, None, param)
        agx_sim.stepForward()

    np.testing.assert_allclose(state.conf.pos[0], agx_position(body), atol=1e-12)
    np.testing.assert_allclose(state.conf.rot[0], agx_rotation(body), atol=1e-12)


# Gear between two shafts built in both simulators


def run_gear(compliance, damping, gear_ratio, inertia_a, inertia_b, ang_a0, steps):
    """
    Two shafts coupled by a holonomic gear, with shaft a initially spinning.
    Returns the shaft angles after each step from AJX and AGX.

    AJX measures the gear violation as q_a - r q_b (input shaft angle), while
    AGX measures it as q_a / r - q_b (output shaft angle). The AGX compliance is
    therefore scaled by 1 / r^2 to describe the same gear.
    """
    timestep = 0.01

    ajx_sim = Simulation(
        SimulationSettings(timestep),
        (),
        (GearConstraint("gear", "shaft_a", "shaft_b"),),
        scalar_body_list=(ScalarBody("shaft_a"), ScalarBody("shaft_b")),
    )
    ajx_param = SimulationParameters(
        jnp.zeros(3),
        RigidBodyParameters.create_empty(),
        ConstraintParameters.create_empty(),
        AGXReferenceSparseParam(),
        scalar_body_param=ScalarBodyParameters.concatenate(
            [
                ScalarBodyParameters.create(inertia_a, True, "shaft_a"),
                ScalarBodyParameters.create(inertia_b, True, "shaft_b"),
            ]
        ),
        scalar_constraint_param=ScalarConstraintParameters.create(
            0.0, 0.0, gear_ratio, True, compliance, damping, "gear"
        ),
    )
    ajx_state = State(
        Configuration(jnp.zeros((0, 3)), jnp.zeros((0, 4)), jnp.zeros(2)),
        GeneralizedVelocity(jnp.zeros((0, 6)), jnp.array([ang_a0, 0.0])),
        multipliers=jnp.zeros(1),
        residual=jnp.zeros(1),
    )

    agx_sim = agx_simulation(timestep, gravity=jnp.zeros(3))
    power_line = agxPowerLine.PowerLine()
    agx_sim.add(power_line)
    shaft_a = agxDriveTrain.Shaft()
    shaft_a.setInertia(inertia_a)
    shaft_b = agxDriveTrain.Shaft()
    shaft_b.setInertia(inertia_b)
    gear = agxDriveTrain.HolonomicGear()
    gear.setGearRatio(gear_ratio)
    gear.setViscousCompliance(compliance / gear_ratio**2)
    gear.setViscousDamping(damping)
    power_line.add(shaft_a)
    shaft_a.connect(gear)
    gear.connect(shaft_b)
    shaft_a.getRotationalDimension().setAngularVelocity(ang_a0)

    ajx_angles, agx_angles = [], []
    for _ in range(steps):
        ajx_state = ajx_step(ajx_sim, ajx_state, ajx_param)
        agx_sim.stepForward()
        ajx_angles.append(np.array(ajx_state.conf.scalar))
        agx_angles.append([shaft_a.getAngle(), shaft_b.getAngle()])
    return np.array(ajx_angles), np.array(agx_angles)


@pytest.mark.parametrize("compliance", [1e-8, 1e-2], ids=["stiff", "compliant"])
def test_gear_matches_agx(compliance):
    ajx_angles, agx_angles = run_gear(
        compliance,
        damping=0.02,
        gear_ratio=2.0,
        inertia_a=1.0,
        inertia_b=2.0,
        ang_a0=1.0,
        steps=100,
    )
    np.testing.assert_allclose(ajx_angles, agx_angles, atol=1e-10)


def test_stiff_gear_shares_momentum_like_agx():
    # The gear impulse lambda acts as +lambda on shaft a and -r lambda on shaft b,
    # so I_a w_a + I_b w_b / r is conserved, and w_a = r w_b once engaged.
    # SPOOK damping removes the initial velocity mismatch over a few steps.
    gear_ratio, inertia_a, inertia_b = 2.0, 1.0, 2.0
    ajx_angles, agx_angles = run_gear(
        1e-8, 0.02, gear_ratio, inertia_a, inertia_b, ang_a0=1.0, steps=100
    )
    ang_b = inertia_a / (inertia_a * gear_ratio + inertia_b / gear_ratio)
    expected = np.array([gear_ratio * ang_b, ang_b])
    for angles in (ajx_angles, agx_angles):
        final_ang = (angles[-1] - angles[-2]) / 0.01
        np.testing.assert_allclose(final_ang, expected, atol=1e-6)

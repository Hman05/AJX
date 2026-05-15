from dataclasses import dataclass, field

import imageio
import jax
import jax.numpy as jnp
import numpy as np
from IPython.display import Video
from flax import struct

import ajx.math as math
from ajx import Transform
from ajx.example_environments.dlo_scoop import (
    CableParameters,
    DLOScoop,
    DLOSettings,
)
from ajx.example_graphics.application import Application
from ajx.example_graphics.environment_scene import EnvironmentScene
from ajx.simulation import SimulationSettings, Solver
from ajx.tree_util import ParameterNode, tangent_jacfwd

# Enable float64 support globally (must be done early).
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)


@dataclass
class ExperimentConfig:
    timestep: float = 0.016667
    horizon: int = 60
    extra_horizon: int = 60

    coulomb_penalty_weight: float = 1e3
    target_velocity_weight: float = 1.0
    target_position_weight: float = 1.0
    control_regularization_weight: float = 0.01
    power_penalty_weight: float = 1.0
    power_max: float = 100.0

    friction_coefficient_eval: float = 0.5
    friction_coefficient_constraint: float = 0.4

    flight_time: float = 0.5
    gravity: jax.Array = field(default_factory=lambda: jnp.array([0.0, 0.0, -9.82]))
    grippermc_to_marker: jax.Array = field(
        default_factory=lambda: jnp.array([0.0478024, 0.0, 0.0])
    )
    target_pos: jax.Array = field(default_factory=lambda: jnp.array([0.3, 0.0, 0.3]))
    bin_pos: jax.Array = field(default_factory=lambda: jnp.array([0.8, 0.0, -0.1]))

    nu: float = 0.333
    youngs_modulus: float = 3e8
    n_segments: int = 4
    dlo_length: float = 0.3
    outer_radius: float = 0.015
    inner_radius: float = 0.013
    density: float = 1000.0

    lm_iterations: int = 15
    lm_damping: float = 1e-4
    render_fps: int = 60
    output_path: str = "animation.mp4"


@struct.dataclass
class ControlTrajectory(ParameterNode):
    data: jax.Array


def relu(x):
    return jnp.maximum(0.0, x)


def compute_target_velocity(config: ExperimentConfig):
    return (
        config.bin_pos - config.target_pos
    ) / config.flight_time - 0.5 * config.gravity * config.flight_time


def build_environment(config: ExperimentConfig):
    target_vel = compute_target_velocity(config)
    env = DLOScoop(
        sim_settings=SimulationSettings(config.timestep, True, Solver.DENSE_LINEAR),
        env_settings=DLOSettings.create(
            n_segments=config.n_segments,
            length=config.dlo_length,
            outer_radius=config.outer_radius,
            inner_radius=config.inner_radius,
            density=config.density,
            pose_estimate_linear_offsets=[],
            loose_end=False,
        ),
        target_pos=config.target_pos,
        target_vel=target_vel,
        bin_pos=config.bin_pos,
    )

    env.camera_pos = jnp.array([0.5, -3.0, 0.0])
    env.camera_rot = math.quat_from_axis_angle(jnp.array([0.0, 0.0, 1.0]), 0.0)

    cable_param = CableParameters(
        youngs_modulus=config.youngs_modulus,
        shear_modulus=config.youngs_modulus / (2 * (1 + config.nu)),
        damping=env.default_param.sparse_param.cable_param.damping,
    )
    env_param = env.default_param.tree_replace(
        src={"sparse_param.cable_param": cable_param}
    )
    initial_state = env.get_neutral_state(env_param)
    return env, env_param, initial_state, target_vel


def coulomb_limit_penalty(state, friction_coefficient, penalty_weight):
    """
    Return residual penalties for violating the contact friction model.
    """
    # The last six multipliers correspond to the cylinder lock contact.
    cylinder_lock_mul = state.multipliers[-6:]
    # Extract the normal force and the 2D tangential force components.
    lambda_t = cylinder_lock_mul[:2]
    lambda_n = cylinder_lock_mul[2]

    # Penalize excessive tangential force relative to the allowed
    # friction cone defined by the supplied friction coefficient.
    lambda_t_norm = jnp.linalg.norm(lambda_t)
    coulomb_residual = penalty_weight * relu(
        lambda_t_norm + friction_coefficient * lambda_n
    )

    # Also penalize positive normal multiplier values so the optimizer
    # prefers the expected sign convention for the contact constraint.
    positive_normal_residual = penalty_weight * relu(lambda_n)

    return jnp.concatenate([coulomb_residual, positive_normal_residual], axis=None)


def get_constraint_jacobian(constraint, env_param, state):
    if hasattr(constraint, "body"):
        body_ids = (env_param.rigid_body_param.names.index(constraint.body),)
    else:
        body_ids = tuple(
            env_param.rigid_body_param.names.index(body) for body in constraint.bodies
        )
    constraint_ids = (env_param.constraint_param.names.index(constraint.name),)
    return constraint.__class__.jacobian(
        env_param,
        state,
        body_ids,
        constraint_ids,
        constraint.constraint_type,
    )


def get_robot_power(state, env, env_param, timestep):
    """Estimate the total mechanical power produced by the robot joints."""
    # There are three constraint for yaw-pitch-roll control.
    link_multipliers = state.multipliers.reshape(-1, 6)[:3]

    # Select the active joint components for the three link constraints.
    pos_yaw_degrees = jnp.array([0, 1, 2, 5])
    hinge_degree = jnp.array([5])
    pos_yaw_multipliers = link_multipliers[0, pos_yaw_degrees]
    pitch_multipliers = link_multipliers[1, hinge_degree]
    roll_multipliers = link_multipliers[2, hinge_degree]

    # Map the constraint multipliers into generalized forces at each joint.
    G1 = env.lock_world_to_hidden1a.object_jacobian(state, env_param)
    f1 = G1.reshape(6, 6)[pos_yaw_degrees].T @ pos_yaw_multipliers / timestep

    G2 = env.lock_hidden1a_to_hidden2a.object_jacobian(state, env_param)
    G2_prime = G2.reshape(2, 6, 6)[:, hinge_degree].transpose(0, 2, 1)
    f2 = G2_prime @ pitch_multipliers / timestep

    G3 = env.lock_hidden2a_to_gripper1.object_jacobian(state, env_param)
    G3_prime = G3.reshape(2, 6, 6)[:, hinge_degree].transpose(0, 2, 1)
    # Convert impulse-like quantities to forces using the timestep.
    f3 = G3_prime @ roll_multipliers / timestep

    # Power is force/torque dotted with the corresponding generalized velocity.
    P1 = jnp.dot(f1, state.gvel.data[0])
    P2_1 = jnp.dot(f2[0], state.gvel.data[0])
    P2_2 = jnp.dot(f2[1], state.gvel.data[1])
    P3_1 = jnp.dot(f3[0], state.gvel.data[1])
    P3_2 = jnp.dot(f3[1], state.gvel.data[2])
    return P1 + P2_1 + P2_2 + P3_1 + P3_2


def power_limit_residual(state, env, env_param, timestep, max_power, penalty_weight):
    """Return a penalty when robot power exceeds the configured limit."""
    total_power = get_robot_power(state, env, env_param, timestep)
    return penalty_weight * relu(total_power - max_power)


def build_residual_function(
    env, env_param, initial_state, target_vel, config: ExperimentConfig
):
    @jax.jit
    def residuals(control_signal):
        control_data = control_signal.data
        per_step_residuals0 = jnp.zeros((config.horizon, 3))

        def body_fn(i, carry):
            state, per_step_residuals = carry
            state = env.step_state(state, control_data[i], env_param)

            contact_residual = coulomb_limit_penalty(
                state,
                friction_coefficient=config.friction_coefficient_constraint,
                penalty_weight=config.coulomb_penalty_weight,
            )
            power_residual = power_limit_residual(
                state,
                env,
                env_param,
                config.timestep,
                max_power=config.power_max,
                penalty_weight=config.power_penalty_weight,
            )
            step_residual = jnp.concatenate(
                [contact_residual, jnp.atleast_1d(power_residual)]
            )
            per_step_residuals = per_step_residuals.at[i].set(step_residual)
            return state, per_step_residuals

        final_state, per_step_residuals = jax.lax.fori_loop(
            0,
            config.horizon,
            body_fn,
            (initial_state, per_step_residuals0),
        )

        terminal_velocity_error = config.target_velocity_weight * (
            final_state.gvel.data[-1, :3] - target_vel
        )
        terminal_position_error = config.target_position_weight * (
            final_state.conf.pos[-1, :3] - config.target_pos
        )
        control_regularization = control_data * config.control_regularization_weight

        return jnp.concatenate(
            [
                per_step_residuals,
                terminal_velocity_error,
                terminal_position_error,
                control_regularization,
            ],
            axis=None,
        )

    return residuals


def levenberg_marquardt(residual, jac_r, joint_param, n_iter=50, damping=1e-3):
    x = joint_param
    lam = damping
    J = None
    for i in range(n_iter):
        J = jac_r(x)  # (m, n)
        r = residual(x)  # (m,)
        cost = 0.5 * jnp.dot(r, r)

        JTJ = J.T @ J
        JTr = J.T @ r

        # LM step: (J^T J + λI) δ = -J^T r
        A = JTJ + lam * jnp.eye(J.shape[1])
        delta = -jnp.linalg.solve(A, JTr)

        x_new = x.retract(delta)
        r_new = residual(x_new)
        cost_new = 0.5 * jnp.dot(r_new, r_new)

        actual_reduction = cost - cost_new
        predicted_reduction = -jnp.dot(JTr, delta) - 0.5 * delta @ (JTJ @ delta)

        if predicted_reduction > 0:
            rho = actual_reduction / predicted_reduction
        else:
            rho = -jnp.inf

        if rho > 0:
            x = x_new
            lam *= max(1 / 3, 1 - (2 * rho - 1) ** 3)
            lam = jnp.maximum(lam, 1e-18)
            accepted = True
        else:
            lam *= 10.0
            accepted = False

        print(
            f"i: {i} | cost: {jnp.linalg.norm(r)} | λ: {lam:.3e} |"
            f"{'accepted' if accepted else 'rejected'}"
        )
        if lam > 1e50:
            break

    return x, J


def released_cylinder_param(env_param):
    return env_param.tree_replace(
        {"constraint_param.compliance.lock_gripper2_to_cylinder": jnp.full((6,), 1e8)},
    )


def render_solution(env, env_param, initial_state, solution, config: ExperimentConfig):
    writer = imageio.get_writer(config.output_path, fps=config.render_fps)

    control_data = solution.data
    control_data = jnp.concatenate(
        [control_data, jnp.zeros_like(control_data[-1])[None]]
    )

    env_step = jax.jit(env.step)
    state = initial_state
    scene = EnvironmentScene(
        env,
        env_param,
        initial_state,
        show_text=False,
        show_fps=False,
        debug_render=False,
    )
    app = Application(scene, config.render_fps, "default", headless=True)

    active_env_param = env_param
    for i in range(config.horizon + config.extra_horizon):
        if i == config.horizon:
            active_env_param = released_cylinder_param(active_env_param)

        state, _observations = env_step(state, control_data[i], active_env_param)
        cylinder_lock_mul = state.multipliers[-6:]
        lambda_n = -cylinder_lock_mul[2]
        lambda_t = cylinder_lock_mul[:2]
        lambda_t_norm = jnp.linalg.norm(lambda_t)
        below_coulomb_limit = (
            lambda_t_norm <= config.friction_coefficient_eval * lambda_n
        )
        if not below_coulomb_limit and i > 0:
            active_env_param = released_cylinder_param(active_env_param)

        scene.state = state
        scene.update_geometry()
        app.graphicsEngine.renderFrame()
        frame = app.get_headless_frame()
        writer.append_data(np.array(frame))

    writer.close()
    return Video(config.output_path)


def main():
    config = ExperimentConfig()
    env, env_param, initial_state, target_vel = build_environment(config)
    residual = build_residual_function(
        env, env_param, initial_state, target_vel, config
    )
    jac_r = jax.jacf(residual)

    control_signal = ControlTrajectory(jnp.ones([config.horizon, 12]))
    solution, _jacobian = levenberg_marquardt(
        residual,
        jac_r,
        control_signal,
        n_iter=config.lm_iterations,
        damping=config.lm_damping,
    )
    return render_solution(env, env_param, initial_state, solution, config)


if __name__ == "__main__":
    main()

import jax

from ajx.example_graphics.environment_scene import EnvironmentScene
from ajx.example_graphics.application import Application
from ajx.example_environments.dlo_scoop import (
    DLOScoop,
    DLOSettings,
    CableParameters,
)
from ajx.simulation import SimulationSettings, Solver
import jax.numpy as jnp
import ajx.math as math
from ajx import Transform
from ajx.tree_util import tangent_jacfwd

# Enable float64 support globally (must be done early).
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)


if __name__ == "__main__":
    timestep = 0.016667
    grippermc_to_marker = jnp.array([0.0478024, 0, 0])

    # Test to optimize path with more strict friction settings
    mu = 0.5
    mu_strict = 0.4

    # Set target-position, from which an object should continue in a trajectory
    # and end up in a bin, at bin-position.
    target_pos = jnp.array([0.3, 0.0, 0.3])
    bin_pos = jnp.array([0.8, 0.0, -0.1])

    # We parameterize the trajectory with a flight-time (s)
    flight_time = 0.5
    g = jnp.array([0.0, 0.0, -9.82])
    # The resultant target-velocity (at target-position) is then
    target_vel = (bin_pos - target_pos) / flight_time - 0.5 * g * flight_time


    env = DLOScoop(
        sim_settings=SimulationSettings(timestep, True, Solver.DENSE_LINEAR),
        env_settings=DLOSettings.create(
            n_segments=4,
            length=0.3,
            outer_radius=0.015,
            inner_radius=0.013,
            density=1000,
            pose_estimate_linear_offsets=[],
            gripper1_offset=Transform(grippermc_to_marker, math.Rotations.unitary),
            gripper2_offset=Transform(-grippermc_to_marker, math.Rotations.unitary),
            loose_end=False,
        ),
        target_pos=target_pos,
        target_vel=target_vel,
        bin_pos=bin_pos,
    )
    env.camera_pos = jnp.array([0.5, -3.0, 0.0])
    env.camera_rot = math.quat_from_axis_angle(jnp.array([0.0, 0.0, 1.0]), 0.0)

    nu = 0.333
    E = 3e8
    cable_param = CableParameters(
        youngs_modulus=E,
        shear_modulus=E / (2 * (1 + nu)),
        damping=env.default_param.sparse_param.cable_param.damping,
    )

    env_param = env.default_param.tree_replace(
        src={"sparse_param.cable_param": cable_param}
    )

    initial_state = env.get_neutral_state(env_param)
    target_pos = env.target_pos
    target_vel = env.target_vel
    horizon = 60

    def softplus(beta, x):
        return 1 / beta * jnp.log(1 + jnp.exp(beta * x))

    def relu(x):
        return jnp.max(jnp.array([0.0, x]))

    def coulomb_limit_penalty(state):
        cylinder_lock_mul = state.multipliers[-6:]
        lambda_n = cylinder_lock_mul[2]
        lambda_t = cylinder_lock_mul[:2]
        lambda_t_norm = jnp.linalg.norm(lambda_t)
        coulomb_residual = 1e3 * relu(lambda_t_norm + mu_strict * lambda_n)
        positive_normal_residual = 1e3 * relu(lambda_n)
        return jnp.concatenate([coulomb_residual, positive_normal_residual], axis=None)

    def get_robot_power(state):
        link_multipliers = state.multipliers.reshape(-1, 6)[:3]

        pos_yaw_degrees = jnp.array([0, 1, 2, 5])
        hinge_degree = jnp.array([5])
        pos_yaw_mulitpliers = link_multipliers[0, pos_yaw_degrees]
        pitch_multipliers = link_multipliers[1, hinge_degree]
        roll_multipliers = link_multipliers[2, hinge_degree]
        G1 = env.lock_world_to_hidden1a.object_jacobian(state, env_param)
        f1 = G1.reshape(6, 6)[pos_yaw_degrees].T @ pos_yaw_mulitpliers / timestep
        G2 = env.lock_hidden1a_to_hidden2a.object_jacobian(state, env_param)
        G2_prime = G2.reshape(2, 6, 6)[:, hinge_degree].transpose(0, 2, 1)
        f2 = G2_prime @ pitch_multipliers / timestep
        G3 = env.lock_hidden2a_to_gripper1.object_jacobian(state, env_param)
        G3_prime = G3.reshape(2, 6, 6)[:, hinge_degree].transpose(0, 2, 1)
        # p = h*f =>
        f3 = G3_prime @ roll_multipliers / timestep

        P1 = jnp.dot(f1, state.gvel.data[0])
        P2_1 = jnp.dot(f2[0], state.gvel.data[0])
        P2_2 = jnp.dot(f2[1], state.gvel.data[1])

        P3_1 = jnp.dot(f3[0], state.gvel.data[1])
        P3_2 = jnp.dot(f3[1], state.gvel.data[2])
        return P1 + P2_1 + P2_2 + P3_1 + P3_2

    @jax.jit
    def residuals(control_signal):
        u = control_signal.data

        all_residuals = []

        def body_fn(i, carry):
            state, residuals = carry

            state = env.step_state(state, u[i], env_param)

            residual = coulomb_limit_penalty(state)

            residuals = residuals.at[i].set(residual)

            return state, residuals

        # Adjust shape depending on residual shape
        residuals0 = jnp.zeros((horizon, 2))

        state, residuals = jax.lax.fori_loop(
            0,
            horizon,
            body_fn,
            (initial_state, residuals0),
        )
        all_residuals.append(residuals)

        # 'state' is of type DLOState, and contains information on this
        # gvel is the generalized velocity, it has shape [:, 6]
        # The velocity for the object is the last entry (hence -1)
        # and use only the linear velocity (first 3 components)
        all_residuals.append(state.gvel.data[-1, :3] - target_vel)
        # Likewise we use only the position (and not the rotation)
        all_residuals.append(state.conf.pos[-1, :3] - target_pos)

        # For stability, it might be an idea to introduce a regularization
        # Here we add a residual term to dampen the control signals
        regularization = u * 0.01
        all_residuals.append(regularization)
        return jnp.concatenate(all_residuals, axis=None)

    residual = residuals
    jac_r = tangent_jacfwd(residual)

    def gauss_newton(x0, n_iter, damping):
        x = x0
        for i in range(n_iter):
            rx = residual(x)
            J = jac_r(x)

            # Gauss-Newton step: solve (J^T J) delta = -J^T r
            JTJ = J.T @ J + jnp.eye(J.shape[1]) * damping
            JTr = J.T @ rx
            delta = -jnp.linalg.solve(JTJ, JTr)

            x = x.retract(delta)
            print(f"Iter: {i}\t |rx|: {jnp.linalg.norm(rx)}")
        return x, J

    def levenberg_marquardt(joint_param, n_iter=50, damping=1e-3):
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

            # Candidate update
            x_new = x.retract(delta)
            r_new = residual(x_new)
            cost_new = 0.5 * jnp.dot(r_new, r_new)

            # Gain ratio (simplified)
            actual_reduction = cost - cost_new
            predicted_reduction = -jnp.dot(JTr, delta) - 0.5 * delta @ (JTJ @ delta)

            if predicted_reduction > 0:
                rho = actual_reduction / predicted_reduction
            else:
                rho = -jnp.inf

            # Decision
            if rho > 0:  # step is good
                x = x_new
                lam *= max(1 / 3, 1 - (2 * rho - 1) ** 3)  # decrease damping
                lam = jnp.maximum(lam, 1e-18)
                accepted = True
            else:  # step is bad
                lam *= 10.0  # increase damping
                accepted = False

            print(
                f"i: {i} | cost: {jnp.linalg.norm(r)} | λ: {lam:.3e} |"
                f"{'accepted' if accepted else 'rejected'}"
            )
            if lam > 1e50:
                break

        return x, J

    # Optimize
    from ajx.tree_util import ParameterNode
    from flax import struct

    @struct.dataclass
    class AJXArray(ParameterNode):
        data: jax.Array

    control_signal = AJXArray(jnp.ones([horizon, 12]))

    solution, Js = gauss_newton(control_signal, n_iter=10, damping=1e-8)
    solution, Js = levenberg_marquardt(solution, n_iter=15, damping=1e-4)

    import imageio
    from IPython.display import Video
    import numpy as np

    writer = imageio.get_writer("animation.mp4", fps=60)

    # Create an interesting control signal
    u = solution.data
    u = jnp.concatenate([u, jnp.zeros_like(u[-1])[None]])

    env_step = jax.jit(env.step)
    state = initial_state
    frames = []

    scene = EnvironmentScene(
        env,
        env_param,
        initial_state,
        show_text=False,
        show_fps=False,
        debug_render=False,
    )
    app = Application(scene, 60, "default", headless=True)

    extra_horizon = 60
    # Simulation loop
    for i in range(horizon + extra_horizon):
        if i == horizon:
            env_param = env_param.tree_replace(
                {
                    "constraint_param.compliance.lock_gripper2_to_cylinder": jnp.array(
                        [1e8, 1e8, 1e8, 1e8, 1e8, 1e8]
                    )
                },
            )
        # Step the environment and store the observation
        state, observations = env_step(state, u[i], env_param)
        cylinder_lock_mul = state.multipliers[-6:]
        lambda_n = -cylinder_lock_mul[2]
        lambda_t = cylinder_lock_mul[:2]
        lambda_t_norm = jnp.linalg.norm(lambda_t)
        below_coulomb_limit = lambda_t_norm <= mu * lambda_n
        if not below_coulomb_limit and i > 0:
            env_param = env_param.tree_replace(
                {
                    "constraint_param.compliance.lock_gripper2_to_cylinder": jnp.array(
                        [1e8, 1e8, 1e8, 1e8, 1e8, 1e8]
                    )
                },
            )
        scene.state = state
        scene.update_geometry()
        app.graphicsEngine.renderFrame()
        frame = app.get_headless_frame()
        writer.append_data(np.array(frame))
    writer.close()

    Video("animation.mp4")

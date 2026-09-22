import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import time

from ajx.constraints import ConstraintResidual
from ajx.example_environments.dlo_attached import DLOAttached, DLOAttachedSettings
from ajx.simulation import SimulationSettings, Solver


def setup_dlo_environment(
    timestep,
    pgs_iterations,
    solver,
    youngs_modulus,
    poission_ratio,
    mass_density,
    body_length=0.05,
    number_of_bodies=20,
):

    env = DLOAttached(
        sim_settings=SimulationSettings(timestep, True, solver, pgs_iterations),
        env_settings=DLOAttachedSettings(
            n_bodies=number_of_bodies,
            body_length=body_length,
            mass_density=mass_density,
            constraint_residual=ConstraintResidual.BEND_TWIST.value,
            hinge_motor_attachment=False,
            body_side_length=0.1 * body_length,
        ),
    )

    shear_modulus = youngs_modulus / (2 * (1 + poission_ratio))
    linear_stiffness, bend_stiffness, torsion_stiffness = env.get_stiffness_from_material_parameters(
        youngs_modulus, shear_modulus
    )

    yz_linear_stiffness = linear_stiffness
    x_linear_stiffness = linear_stiffness
    bend_linear_stiffness = bend_stiffness
    torsion_linear_stiffness = torsion_stiffness

    env_param = env.default_param.tree_replace(
        src={
            "sparse_param.coupled_constraint_param": {
                "linear_stiffness.data": jnp.array(
                    [
                        x_linear_stiffness,
                        yz_linear_stiffness,
                        yz_linear_stiffness,
                        bend_linear_stiffness,
                        bend_linear_stiffness,
                        torsion_linear_stiffness,
                    ]
                ),
                "is_velocity": jnp.array([0, 0, 0, 0, 0, 0], dtype=bool),
            }
        }
    )

    return env, env_param


def simulate_dlo(settings_dict, tmax=None):
    """
    INPUTS:
        settings_dict: A python dict with settings.
        tmax (Optional): Stop simulation if t > tmax
    OUTPUTS:
        dlo_x: x-position data for DLO. array of size (horizon+1) x N_BODIES
        dlo_z: z_position data for DLO. array of size (horizon+1) x N_BODIES
        dz: z_position data for end point of DLO. array of size (horizon+1) x 1
    """

    # To extract simulation settings/arguments
    horizon = settings_dict["horizon"]
    timestep = settings_dict["timestep"]
    pgs_iterations = settings_dict["pgs_iterations"]
    solver = settings_dict["solver"]
    youngs_modulus = settings_dict["youngs_modulus"]
    poisson_ratio = settings_dict["poisson_ratio"]
    mass_density = settings_dict["mass_density"]
    number_of_bodies = settings_dict["number_of_bodies"]

    # To setup and initialize a dlo environment
    env, env_param = setup_dlo_environment(
        timestep, pgs_iterations, solver, youngs_modulus, poisson_ratio, mass_density, number_of_bodies=number_of_bodies
    )

    state = env.state_from_angles(env_param)
    env_step = jax.jit(env.step_residual)
    u = np.zeros(
        [
            horizon,
        ]
    )

    # To prepare storage of rigid body data
    position = [[], [], []]
    for j in range(3):
        position[j].append(np.array(state.conf.pos)[:, j])
    loose_end_id = env_param.rigid_body_param.names.index(f"body{number_of_bodies-1}")

    # Simulation loop
    runtimes = []
    residuals = []
    for j in range(horizon):

        if tmax is not None and (j + 1) * timestep > tmax:
            break

        # Step the environment and store the observation
        start_time = time.perf_counter()
        state, res = env_step(state, u, env_param)
        jax.block_until_ready(state)
        jax.block_until_ready(res)
        step_runtime = time.perf_counter() - start_time

        for j in range(3):
            position[j].append(np.array(state.conf.pos)[:, j])
        runtimes.append(step_runtime)
        residuals.append(res)

    for j in range(3):
        position[j] = np.array(position[j])
    bend = position[2][:, loose_end_id]
    runtimes = np.array(runtimes)

    return tuple(position), bend, runtimes, residuals


def generic_2d_plot(ax, y, label="", title="", axis_labels=["", ""]):
    ax.plot(np.arange(y.size), y, linewidth=2, label=label)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_title(title)
    ax.legend()


def compute_runtime_stats(runtimes, confidence_level=0.95):
    """
    INPUT: runtimes
    """

    mean = np.mean(runtimes)
    sem = sp.stats.sem(runtimes)

    ci_low, ci_high = sp.stats.t.interval(
        confidence=confidence_level,
        df=runtimes.size - 1,
        loc=mean,
        scale=sem,
    )

    return mean, (ci_low, ci_high)


def print_runtime_stats(runtimes, output_stats=False):
    mean, ci = compute_runtime_stats(runtimes[1:])
    print(f"Compiled step: {runtimes[0]} sec")
    print(f"Runtime per step: {mean} sec, 95%-confidence interval ({ci[0]},{ci[1]})")

    if output_stats:
        return mean, ci


if __name__ == "__main__":

    settings_dict = {
        "horizon": 3000,
        "timestep": 0.005,
        "pgs_iterations": 1600,
        "solver": Solver.DENSE_LINEAR,
        "youngs_modulus": 1e9,
        "poisson_ratio": 0.3,
        "mass_density": 1000.0,
        "number_of_bodies": 20,
    }

    position, bend, runtimes, residuals = simulate_dlo(settings_dict)
    print_runtime_stats(runtimes)

    fig, ax = plt.subplots(2, 1)
    generic_2d_plot(
        ax[0], bend, label="Direct solver", title="Cantilever Beam, downward bend", axis_labels=["timestep", "[m]"]
    )

    settings_dict["solver"] = Solver.SPARSE_PGS
    iterations = [200, 400, 800, 1600]

    residual_max_norms = []
    for it in iterations:
        settings_dict["pgs_iterations"] = it
        position, bend, runtimes, residuals = simulate_dlo(settings_dict)
        print_runtime_stats(runtimes)
        generic_2d_plot(
            ax[0],
            bend,
            label=f"PGS, {it} iterations",
            title="Cantilever Beam, downward bend",
            axis_labels=["timestep", "[m]"],
        )

        residual_max_norms.append(np.array([np.linalg.norm(res, ord=np.inf) for res in residuals]))
        generic_2d_plot(
            ax[1],
            residual_max_norms[-1],
            label=f"PGS, {it} iterations",
            title="Solver residual",
            axis_labels=["timestep", "residual max norm"],
        )

    # To plot the largest absolute error in any time step versus iteration count.
    plt.figure()
    plt.semilogy(iterations, np.array([np.linalg.norm(res, ord=np.inf) for res in residual_max_norms]), marker="o")
    plt.xlabel("PGS-iterations")
    plt.ylabel("residual max norm")
    plt.grid(which="major", linestyle="-", linewidth=0.8)
    plt.grid(which="minor", linestyle=":", linewidth=0.65)
    plt.show()

    settings_dict = {
        "horizon": 100,
        "timestep": 0.005,
        "pgs_iterations": 400,
        "solver": Solver.DENSE_LINEAR,
        "youngs_modulus": 1e9,
        "poisson_ratio": 0.3,
        "mass_density": 1000.0,
        "number_of_bodies": 20,
    }

    mean_runtime = []
    confidence_interval = []
    number_of_bodies = [50, 100, 200, 400, 800]
    for solver in [Solver.DENSE_LINEAR, Solver.SPARSE_PGS]:
        rt = []
        cint = []
        settings_dict["solver"] = solver
        for nb in number_of_bodies:
            settings_dict["number_of_bodies"] = nb
            _, _, runtimes, _ = simulate_dlo(settings_dict)
            mean_rt, ci = print_runtime_stats(runtimes, output_stats=True)
            rt.append(mean_rt)
            cint.append(ci)
        mean_runtime.append(rt)
        confidence_interval.append(np.array(cint).T)

    error_magnitude = [0.5 * (ci[1, :] - ci[0, :]) for ci in confidence_interval]

    plt.figure()
    plt.errorbar(
        number_of_bodies,
        mean_runtime[0],
        yerr=error_magnitude[0],
        label=f"Direct solver, dt={settings_dict['timestep']}",
        capsize=5,
    )
    plt.errorbar(
        number_of_bodies,
        mean_runtime[1],
        yerr=error_magnitude[1],
        label=f"Sparse PGS solver, dt={settings_dict['timestep']}, iter={settings_dict['pgs_iterations']}",
        capsize=5,
    )
    plt.xlabel("number of bodies")
    plt.ylabel("Mean runtime per step [s]")
    plt.legend()
    plt.grid(which="major", linestyle="-", linewidth=0.8)
    plt.grid(which="minor", linestyle=":", linewidth=0.65)
    plt.show()

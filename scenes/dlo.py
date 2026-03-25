import jax

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)

from ajx.example_graphics.environment_scene import EnvironmentScene
from ajx.example_graphics.application import Application
from ajx.constraints import ConstraintType
from ajx.example_environments.dlo import DLO, DLOSettings
from ajx.simulation import SimulationSettings, Solver
import jax.numpy as jnp

if __name__ == "__main__":
    timestep = 0.016667

    environment = DLO(
        sim_settings=SimulationSettings(timestep, True, Solver.DENSE_LINEAR),
        env_settings=DLOSettings(
            n_bodies=100,
            body_length=0.02,
            constraint_type=ConstraintType.SE3.value,
            loose_end=False,
        ),
    )
    yz_linear_stiffness = 1e4
    x_linear_stiffness = 1e4
    bend_linear_stiffness = 1e4
    torsion_linear_stiffness = 1e4

    yz_quadratic_stiffness = 0.0
    x_quadratic_stiffness = 0.0
    bend_quadratic_stiffness = 0.0
    torsion_quadratic_stiffness = 0.0

    env_param = environment.default_param.tree_replace(
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
                "quadratic_stiffness.data": jnp.array(
                    [
                        x_quadratic_stiffness,
                        yz_quadratic_stiffness,
                        yz_quadratic_stiffness,
                        bend_quadratic_stiffness,
                        bend_quadratic_stiffness,
                        torsion_quadratic_stiffness,
                    ]
                ),
                "is_velocity": jnp.array([0, 0, 0, 0, 0, 0], dtype=bool),
            }
        }
    )
    env_param = env_param.tree_replace(
        {
            f"constraint_param.is_velocity.grapple2_lock": {5: True},
        }
    )

    initial_state = environment.state_from_angles(env_param)
    environment.lock_joints[1].get_free_degrees(initial_state, env_param)

    scene = EnvironmentScene(environment, env_param, initial_state)
    app = Application(scene, 60, "default")
    app.run()

import jax

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)

from ajx.example_graphics.environment_scene import EnvironmentScene
from ajx.example_graphics.application import Application
from ajx.constraints import ConstraintType
from ajx.example_environments.dlo import DLO, DLOSettings, CableParameters
from ajx.simulation import SimulationSettings, Solver
import jax.numpy as jnp
import ajx.math as math

if __name__ == "__main__":
    timestep = 0.016667

    environment = DLO(
        sim_settings=SimulationSettings(timestep, True, Solver.DENSE_LINEAR),
        env_settings=DLOSettings.create(
            n_segments=50,
            length=0.6,
            constraint_type=ConstraintType.SE3.value,
            observation_type="pose",
            pose_estimates_at=[0.10, 0.20, 0.30, 0.40, 0.50],
            loose_end=False,
        ),
    )

    cable_param = CableParameters(
        stiffness=jnp.array([1e3, 1e0, 1e0]),
        damping=environment.default_param.sparse_param.cable_param.damping,
        is_velocity=environment.default_param.sparse_param.cable_param.is_velocity,
    )

    env_param = environment.default_param.tree_replace(
        src={"sparse_param.cable_param": cable_param}
    )

    initial_state = environment.get_neutral_state(env_param)

    scene = EnvironmentScene(environment, env_param, initial_state)
    app = Application(scene, 60, "default")
    app.run()

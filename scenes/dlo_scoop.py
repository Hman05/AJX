import jax

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)

from ajx.example_graphics.environment_scene import EnvironmentScene
from ajx.example_graphics.application import Application
from ajx.constraints import ConstraintType
from ajx.example_environments.dlo_scoop import DLOScoop, DLOSettings, CableParameters
from ajx.simulation import SimulationSettings, Solver
import jax.numpy as jnp
import ajx.math as math
from ajx import Transform

if __name__ == "__main__":
    timestep = 0.016667
    grippermc_to_marker = jnp.array([0.0478024, 0, 0])

    marker_offsets = []
    n_marker_segments = len(marker_offsets)
    n_segments = 8
    print(f"n_segments: {n_segments}")
    environment = DLOScoop(
        sim_settings=SimulationSettings(timestep, True, Solver.DENSE_LINEAR),
        env_settings=DLOSettings.create(
            n_segments=n_segments,
            length=0.6,
            outer_radius=0.016,
            inner_radius=0.014,
            density=1000,
            pose_estimate_linear_offsets=marker_offsets,
            loose_end=False,
        ),
        target_pos=jnp.array([0.3, 0.0, 0.3]),
        target_vel=jnp.array([2.0, 0.0, 2.0, 0.0, 0.0, 0.0]),
        bin_pos=jnp.array([1.5, 0.0, -0.8]),
    )

    nu = 0.333
    E = 6e6
    cable_param = CableParameters(
        youngs_modulus=E,
        shear_modulus=E / (2 * (1 + nu)),
        damping=environment.default_param.sparse_param.cable_param.damping,
    )
    stiffness = cable_param.get_stiffness(
        0.016, 0.014, environment.env_settings.segment_halflength * 2
    )

    env_param = environment.default_param.tree_replace(
        src={"sparse_param.cable_param": cable_param}
    )

    initial_state = environment.get_neutral_state(env_param)

    scene = EnvironmentScene(environment, env_param, initial_state, debug_render=False)
    app = Application(scene, 60, "default")
    app.run()

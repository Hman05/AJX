import jax

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)

from ajx.example_graphics.environment_scene import EnvironmentScene
from ajx.example_graphics.application import Application
from ajx.constraints import ConstraintType
from ajx.example_environments.dlo import DLO, DLOSettings, CableParameters
from ajx.example_environments.locked_dlo import LockedDLO
from ajx.simulation import SimulationSettings, Solver
import jax.numpy as jnp
import ajx.math as math
from ajx import Transform

if __name__ == "__main__":
    timestep = 0.016667
    grippermc_to_marker = jnp.array([0.0478024, 0, 0])

    marker_offsets = [0.10, 0.20, 0.30, 0.40, 0.50]
    n_marker_segments = len(marker_offsets)
    n_segments_between_markers = 6
    n_segments = (
        n_segments_between_markers * (n_marker_segments + 1) + n_marker_segments
    )
    print(f"n_segments: {n_segments}")
    environment = DLO(
        sim_settings=SimulationSettings(timestep, True, Solver.DENSE_LINEAR),
        env_settings=DLOSettings.create(
            n_segments=n_segments,
            length=0.6,
            radius=0.016,
            density=1000,
            pose_estimate_linear_offsets=marker_offsets,
            gripper1_offset=Transform(grippermc_to_marker, math.Rotations.unitary),
            gripper2_offset=Transform(-grippermc_to_marker, math.Rotations.unitary),
            loose_end=False,
        ),
    )

    nu = 0.333
    E = 1e7
    cable_param = CableParameters(
        youngs_modulus=E,
        shear_modulus=E / (2 * (1 + nu)),
        damping=environment.default_param.sparse_param.cable_param.damping,
    )
    stiffness = cable_param.get_stiffness(
        0.016, environment.env_settings.segment_halflength * 2
    )

    env_param = environment.default_param.tree_replace(
        src={"sparse_param.cable_param": cable_param}
    )

    initial_state = environment.get_neutral_state(env_param)

    scene = EnvironmentScene(environment, env_param, initial_state, debug_render=False)
    app = Application(scene, 60, "default")
    app.run()

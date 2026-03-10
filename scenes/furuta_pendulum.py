from ajx.example_graphics.environment_scene import EnvironmentScene
from ajx.example_graphics.application import Application


from ajx.example_environments.furuta import Furuta

from ajx import *

if __name__ == "__main__":
    timestep = 0.016667

    environment = Furuta(
        sim_settings=SimulationSettings(timestep, True, Solver.DENSE_LINEAR)
    )
    override_param = {
        "sparse_param": {
            "electric_motor": {
                "inertia": 1e-4,
            }
        },
        "rigid_body_param": {
            "mass": {
                # "arm1": 0.2,
                "arm2": 0.4,
            },
        },
        # "rigid_body_param.mass.arm1": 0.8,
    }

    env_param = environment.default_param.tree_replace(src=override_param)
    env_param.get_value_at_path("rigid_body_param.mass.arm2")
    theta1 = 1.0
    theta2 = 4.0

    initial_state = environment.state_from_angles(theta1, theta2, env_param)

    scene = EnvironmentScene(environment, env_param, initial_state)
    app = Application(scene, 60, "default")
    app.run()

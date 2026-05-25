from ajx.example_graphics.environment_scene import EnvironmentScene
from ajx.example_graphics.application import Application


from ajx.example_environments.double_pendulum import (
    DoublePendulum,
    DoublePendulumSettings,
)

from ajx import *

if __name__ == "__main__":
    timestep = 0.016667

    environment = DoublePendulum(
        sim_settings=SimulationSettings(timestep, True, Solver.DENSE_LINEAR),
        env_settings=DoublePendulumSettings(None, 1.0, 1.0),
    )

    env_param = environment.default_param
    theta1 = jnp.pi + 1.6
    theta2 = 0.0 + 2.4

    initial_state = environment.state_from_angles(theta1, theta2, env_param)

    scene = EnvironmentScene(environment, env_param, initial_state)
    app = Application(scene, 60, "default")
    app.run()

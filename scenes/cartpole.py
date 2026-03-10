from ajx.example_graphics.environment_scene import EnvironmentScene
from ajx.example_graphics.application import Application

from ajx.example_environments.cartpole import CartPole

from ajx import *

if __name__ == "__main__":
    timestep = 0.016667

    env = CartPole(
        sim_settings=SimulationSettings(timestep, False, Solver.DENSE_LINEAR)
    )
    env_param = env.default_param.tree_replace(src={})

    initial_state = env.state_from_angles(5.0, 3.0, env_param)

    scene = EnvironmentScene(env, env_param, initial_state)
    app = Application(scene, 60, "default")
    app.run()

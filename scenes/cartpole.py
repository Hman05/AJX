from scenes.base import GraphicalEnvironmentBase

from environments.cartpole import CartPole
from functools import partial

from ajx import *

if __name__ == "__main__":
    timestep = 0.016667

    env = CartPole(timestep)
    env_param = env.default_param.insert(src={})

    initial_state = env.state_from_angles(5.0, 3.0, env_param)

    controller = GraphicalEnvironmentBase(env, env_param, initial_state)
    controller.run()

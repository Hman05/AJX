from scenes.base import GraphicalEnvironmentBase

from environments.furuta import Furuta

from ajx import *


if __name__ == "__main__":
    timestep = 0.016667

    environment = Furuta(
        timestep=timestep,
        reference_timestep=timestep,
        use_gyroscopic=True,
    )
    env_param = environment.default_param.insert(src={})
    theta1 = 1.0
    theta2 = 4.0

    initial_state = environment.state_from_angles(theta1, theta2, env_param)
    controller = GraphicalEnvironmentBase(environment, env_param, initial_state)
    controller.run()

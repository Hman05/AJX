from scenes.base import GraphicalEnvironmentBase

from environments.free_body import FreeBody

from ajx.simulation import SimulationSettings, Solver
import jax.numpy as jnp

if __name__ == "__main__":
    timestep = 0.016667

    environment = FreeBody(
        sim_settings=SimulationSettings(timestep, True, Solver.DENSE_LINEAR)
    )
    env_param = environment.default_param.insert(src={})
    angvel = jnp.array([0.0, 0.5, 0.5])

    initial_state = environment.state_from_angular_velocity(angvel)

    controller = GraphicalEnvironmentBase(environment, env_param, initial_state)
    controller.run()

from abc import ABC, abstractmethod
import jax.numpy as jnp
import ajx.math as math
from jax import vmap, jit
from ajx import *


class Environment(ABC):
    @abstractmethod
    def __init__(self, param={}):
        pass

    def post_init(self):
        observable_names = []
        residual_names = []
        for sensor in self.sim.sensor_list:
            obs_names = sensor.observable_names
            res_names = sensor.residual_names

            observable_names.extend([f"{sensor.name}.{name}" for name in obs_names])
            residual_names.extend([f"{sensor.name}.{name}" for name in res_names])
        self.observable_names = observable_names
        self.residual_names = residual_names

    def inverse_dynamics(self, state, target_state, u, param):
        return self.sim.inverse_dynamics(state, target_state, u, param)

    def observe_state(self, state, u, param):
        qdot_next, mul, code = self.force(state, u, param)
        return self.observe(state, qdot_next, param)

    def observation_residual(self, target, prediction):
        residual_list = []
        i = 0
        for sensor in self.sensors:
            i_next = i + len(sensor.observable_names)
            residual = sensor.residual(target[i:i_next], prediction[i:i_next])
            residual_list.append(residual)
            i = i_next
        return jnp.concatenate(residual_list)

    def step(self, state, action, param):
        (qdot_next, multipliers), code = self.sim.pre_step(state, action, param)
        observation = self.sim.observe(state, qdot_next, param)
        new_state = self.sim.post_step(state, qdot_next)
        return new_state, observation

    def get_state_names(self):
        pos_coords = [".x", ".y", ".z"]
        rot_coords = [".qs", ".qx", ".qy", ".qz"]
        gvel_coords = [".x_dot", ".y_dot", ".z_dot", ".rx_dot", ".ry_dot", ".rz_dot"]
        pos_dof_names = [
            item
            for body_name in self.rb_param.names
            for item in [body_name + c for c in pos_coords]
        ]
        rot_dof_names = [
            item
            for body_name in self.rb_param.names
            for item in [body_name + c for c in rot_coords]
        ]
        gvel_dof_names = [
            item
            for body_name in self.rb_param.names
            for item in [body_name + c for c in gvel_coords]
        ]
        return [*pos_dof_names, *rot_dof_names, *gvel_dof_names]

    def get_state_residual_names(self):
        pos_coords = [".x", ".y", ".z"]
        rot_coords = [".rx", ".ry", ".rz"]
        gvel_coords = [".x_dot", ".y_dot", ".z_dot", ".rx_dot", ".ry_dot", ".rz_dot"]
        pos_dof_names = [
            item
            for body_name in self.rb_param.names
            for item in [body_name + c for c in pos_coords]
        ]
        rot_dof_names = [
            item
            for body_name in self.rb_param.names
            for item in [body_name + c for c in rot_coords]
        ]
        gvel_dof_names = [
            item
            for body_name in self.rb_param.names
            for item in [body_name + c for c in gvel_coords]
        ]
        return [*pos_dof_names, *rot_dof_names, *gvel_dof_names]

    def observation_strings(self, observation):
        return [
            f"{name}: {obs}" for name, obs in zip(self.observable_names, observation)
        ]

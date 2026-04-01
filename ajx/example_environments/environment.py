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

    def observation_residual(self, target, prediction):
        residual_list = []
        i = 0
        for sensor in self.sim.sensor_list:
            i_next = i + len(sensor.observable_names)
            residual = sensor.residual(target[i:i_next], prediction[i:i_next])
            residual_list.append(residual)
            i = i_next
        return jnp.concatenate(residual_list)

    def observe_state(self, state, u, param):
        state, ((qdot_next, mul), code) = self.sim.pre_step(state, u, param)
        return self.sim.observe(state, qdot_next, param)

    def observe_state_nostep(self, state, u, param):
        return self.sim.observe(state, None, param)

    def step(self, state, action, param):
        state, ((qdot_next, multipliers), code) = self.sim.pre_step(
            state, action, param
        )
        observation = self.sim.observe(state, qdot_next, param)
        new_state = self.sim.post_step(state, qdot_next)
        return new_state, observation

    def step_state(self, state, action, param):
        state, ((qdot_next, multipliers), code) = self.sim.pre_step(
            state, action, param
        )
        new_state = self.sim.post_step(state, qdot_next)
        return new_state

    def get_multiplier_names(self):
        return [
            item
            for constraint in self.sim.constraint_list
            for item in [
                f"{constraint.name}.{mn}" for mn in constraint.get_multiplier_names()
            ]
        ]

    def get_state_names(self):
        pos_coords = [".x", ".y", ".z"]
        rot_coords = [".qs", ".qx", ".qy", ".qz"]
        gvel_coords = [".x_dot", ".y_dot", ".z_dot", ".rx_dot", ".ry_dot", ".rz_dot"]
        pos_dof_names = [
            item
            for rb in self.sim.rigid_body_list
            for item in [rb.name + c for c in pos_coords]
        ]
        rot_dof_names = [
            item
            for rb in self.sim.rigid_body_list
            for item in [rb.name + c for c in rot_coords]
        ]
        gvel_dof_names = [
            item
            for rb in self.sim.rigid_body_list
            for item in [rb.name + c for c in gvel_coords]
        ]
        return [*pos_dof_names, *rot_dof_names, *gvel_dof_names]

    def get_state_residual_names(self):
        pos_coords = [".x", ".y", ".z"]
        rot_coords = [".rx", ".ry", ".rz"]
        gvel_coords = [".x_dot", ".y_dot", ".z_dot", ".rx_dot", ".ry_dot", ".rz_dot"]
        pos_dof_names = [
            item
            for rb in self.sim.rigid_body_list
            for item in [rb.name + c for c in pos_coords]
        ]
        rot_dof_names = [
            item
            for rb in self.sim.rigid_body_list
            for item in [rb.name + c for c in rot_coords]
        ]
        gvel_dof_names = [
            item
            for rb in self.sim.rigid_body_list
            for item in [rb.name + c for c in gvel_coords]
        ]
        return [*pos_dof_names, *rot_dof_names, *gvel_dof_names]

    def observation_strings(self, observation):
        return [
            f"{name}: {obs}" for name, obs in zip(self.observable_names, observation)
        ]

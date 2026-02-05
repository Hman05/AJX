from abc import ABC, abstractmethod
import jax.numpy as jnp
import ajx.math as math
from jax import vmap, jit
from ajx import *


def vmap_batch_branch(fun: Callable, in_axes=0):
    batch_branch_in_axes = tuple(0 if a else None for a in in_axes)
    return vmap(vmap(fun, batch_branch_in_axes), batch_branch_in_axes)


class Environment(ABC):
    @abstractmethod
    def __init__(self, param={}):
        pass

    @abstractmethod
    def build_sim(self):
        pass

    @abstractmethod
    def observe(self, state, param):
        pass

    @abstractmethod
    def step(self, state, qdot_next):
        pass

    @abstractmethod
    def observation_to_configuration(self, observation, params):
        pass

    @abstractmethod
    def observation_residual(self, observation):
        pass

    def preprocess_observations(observations):
        return observations

    def get_dof_removed(self):
        total_dof_removed = 0
        for constraint in self.constraints:
            total_dof_removed += constraint.dof_removed
        return total_dof_removed

    def body_id(self, str):
        return self.rb_names.index(str)

    def post_init(self):
        observable_names = []
        residual_names = []
        for sensor in self.sensors:
            obs_names = sensor.observable_names
            res_names = sensor.residual_names

            observable_names.extend([f"{sensor.name}.{name}" for name in obs_names])
            residual_names.extend([f"{sensor.name}.{name}" for name in res_names])
        self.observable_names = observable_names
        self.residual_names = residual_names

        # ghost_names = []
        # for i, constraint in enumerate(self.constraints):
        #     names = get_ghost_names(constraint, self.param, i)
        #     ghost_names.extend(names)
        # self.ghost_names = ghost_names

        self.batched_force = jit(vmap(self.force, (0, 0, None)))
        self.batched_branched_force = jit(
            vmap_batch_branch(self.force, (True, True, False))
        )
        self.batched_observe = jit(vmap(self.observe, (0, 0, None)))
        self.batched_branched_observe = jit(
            vmap_batch_branch(self.observe, (True, True, False))
        )
        self.batched_step = jit(vmap(self.step))
        self.batched_branched_step = jit(vmap(vmap(self.step)))

        self.batched_branched_inverse_dynamics = jit(
            vmap_batch_branch(self.inverse_dynamics, (True, True, True, False))
        )

    def inverse_dynamics(self, state, target_state, u, param):
        return self.sim.inverse_dynamics(state, target_state, u, param)

    def force(self, state, u, param):
        return self.sim.force(state, u, param)

    def observe(self, state, qdot_next, param):
        observation_list = [jnp.zeros([0])]
        for sensor in self.sensors:
            observation = sensor.observe(state, qdot_next, param)
            observation_list.append(observation)

        return jnp.concatenate(observation_list)

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

    def step(self, state, qdot_next):
        return self.sim.step(state, qdot_next)

    def configurations_to_state(
        self, configuration0: Configuration, configuration1: Configuration
    ):
        vel = (configuration1.pos - configuration0.pos) / self.timestep
        ang = (
            vmap(math.quat_residual)(configuration1.rot, configuration0.rot)
            / self.timestep
        )
        gvel = GeneralizedVelocity(jnp.concatenate([vel, ang], axis=1))
        return State(configuration1, gvel)

    def xarray_to_trajectory(self, xtraj):
        pos_list = []
        rot_list = []
        gvel_list = []
        for name in self.rb_param.names:
            rb_x = xtraj.sel(dof=f"{name}.x").values
            rb_y = xtraj.sel(dof=f"{name}.y").values
            rb_z = xtraj.sel(dof=f"{name}.z").values
            rb_qs = xtraj.sel(dof=f"{name}.qs").values
            rb_qx = xtraj.sel(dof=f"{name}.qx").values
            rb_qy = xtraj.sel(dof=f"{name}.qy").values
            rb_qz = xtraj.sel(dof=f"{name}.qz").values

            rb_x_dot = xtraj.sel(dof=f"{name}.x_dot").values
            rb_y_dot = xtraj.sel(dof=f"{name}.y_dot").values
            rb_z_dot = xtraj.sel(dof=f"{name}.z_dot").values
            rb_rx_dot = xtraj.sel(dof=f"{name}.rx_dot").values
            rb_ry_dot = xtraj.sel(dof=f"{name}.ry_dot").values
            rb_rz_dot = xtraj.sel(dof=f"{name}.rz_dot").values

            rb_pos = jnp.stack([rb_x, rb_y, rb_z], 1)
            rb_rot = jnp.stack([rb_qs, rb_qx, rb_qy, rb_qz], 1)
            rb_gvel = jnp.stack(
                [rb_x_dot, rb_y_dot, rb_z_dot, rb_rx_dot, rb_ry_dot, rb_rz_dot],
                1,
            )
            pos_list.append(rb_pos)
            rot_list.append(rb_rot)
            gvel_list.append(rb_gvel)
        pos = jnp.stack(pos_list, axis=1)
        rot = jnp.stack(rot_list, axis=1)
        gvel = jnp.stack(gvel_list, axis=1)

        return State(Configuration(pos, rot), GeneralizedVelocity(gvel))

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

    def get_display_text(self, observation):
        return []

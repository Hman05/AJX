import jax.numpy as jnp
from ajx.environment import Environment
import ajx.simulation as simulation
from ajx import *
from util.deepinsert import deepinsert

from typing import Dict
import scenes.graphics.geometry as geometry

from ajx.param import SimulationParameters


class Pendulum(Environment):
    def __init__(
        self,
        sim_settings: SimulationSettings,
        has_quadratic_damping: bool,
    ):
        self.has_quadratic_damping = has_quadratic_damping

        self._build_sim(sim_settings)
        self.control_names = []

        super().post_init()

    def get_hyperparam(self):
        return {
            "timestep": self.timestep,
            "has_quadratic_damping": self.has_quadratic_damping,
        }

    def latex_param_names(pnames):
        pname_mapping = {
            "hinge.b": "$b\\:(\\si{\\kilogram\\per\\second})$",
            "damping.b": "$b\\:(\\si{\\kilogram\\per\\second})$",
            "damping.c": "$c\\:(\\si{\\kilogram\\per\\meter})$",
            "pendulum.inertia.0": "$J\\:(\\si{\\kilogram\\meter^2})$",
        }
        return {name: pname_mapping.get(name, name) for name in pnames}

    @property
    def hinge_offset(self):
        return self.param["hinge"].frame_b.position[1]

    def _build_sim(self, sim_settings):
        self.pendulum_box = geometry.Box(
            "pendulum_box", 0.08, 1.28, 0.08, (0.0, 0.0, 0.0), color=(0.9, 0.2, 0.2)
        )

        self.pendulum = RigidBody("pendulum", ("pendulum_box",))
        self.pendulum_param = RigidBodyParameters.create(
            mass=0.238,
            inertia_diag=jnp.array([0.02, 0.02, 0.02]),
            name="pendulum",
        )

        self.rb_param, self.rigid_bodies = RigidBodyParameters.stack_with_rigid_bodies(
            [(self.pendulum_param, self.pendulum)]
        )

        self.hinge = HingeJoint(
            name="hinge",
            body_a=None,
            body_b="pendulum",
        )

        frame_rotation = math.quat_from_axis_angle(jnp.array([0.0, 0.0, 1.0]), 0.0)

        self.hinge_param = ConstraintParameters.create(
            frame_a=Frame(jnp.array([0.0, 0.0, 0.0]), frame_rotation),
            frame_b=Frame(jnp.array([0.0, 0.091, 0.0]), frame_rotation),
            compliance=1e-5,
            damping=2 * sim_settings.timestep,
            b=0.0004,
            name="hinge",
        )

        self.constraint_param, self.constraints = (
            ConstraintParameters.stack_with_constraints(
                [(self.hinge_param, self.hinge)]
            )
        )

        self.damping_param = QuadraticDampingParameters(b=0.04, c=0.01)

        self.quadratic_damping = QuadraticDampingComponent("damping", self.hinge)

        self.pre_step_modifiers = ()
        if self.has_quadratic_damping:
            self.pre_step_modifiers.append(self.quadratic_damping)

        self.rotary_decoder = RotaryEncoderHingeMounted("rotary_encoder", self.hinge)
        self.sensors = (self.rotary_decoder,)

        self.sim = simulation.Simulation(
            sim_settings,
            self.rigid_bodies,
            self.constraints,
            self.sensors,
            self.pre_step_modifiers,
        )

        self.default_param = SimulationParameters(
            jnp.array([0.0, -9.82, 0.0]),
            self.rb_param,
            self.constraint_param,
            sparse_param={
                "damping": self.damping_param,
            },
        )

        self.geometry_list = (self.pendulum_box,)
        self.extra_geometry = (
            geometry.Square(
                "ground", 1.5, 1.5, (0.0, -3.0, 0.0), color=(0.2, 0.5, 0.2)
            ),
            geometry.Box(
                "stand", 0.2, 1.5, 0.2, (0.28, -1.5, 0.0), color=(0.2, 0.2, 0.2)
            ),
        )

    def stribeck_func(tangential_velocity, param):
        mu_kinematic = jnp.abs(param["stribeck"].mu_kinematic)
        mu_static_inc = jnp.abs(param["stribeck"].mu_static_inc)
        decay_rate = jnp.abs(param["stribeck"].decay_rate)
        abs_v = jnp.abs(tangential_velocity)
        return jnp.array(
            mu_kinematic + mu_static_inc * jnp.exp(-(abs_v**2) * decay_rate)
        )

    def observation_to_configuration(self, observation, param):
        world_transform = Configuration(
            jnp.array([0.0, 0.0, 0.0]), jnp.array([1.0, 0.0, 0.0, 0.0])
        )

        theta = observation[0]
        pendulum_transform = self.hinge.place_other(param, world_transform, theta)

        return Configuration.stack([pendulum_transform])

    def state_from_angle(self, theta, param):
        initial_obs = jnp.stack([theta], axis=-1)
        zero_velocity = jnp.zeros([6])

        initial_conf = self.observation_to_configuration(initial_obs, param)
        initial_gvel = GeneralizedVelocity(jnp.zeros([1, 6]))

        return State(initial_conf, initial_gvel)

    def preprocess_observations(observations):
        import numpy as np
        from itertools import product

        def dict_product(inp):
            return (dict(zip(inp.keys(), values)) for values in product(*inp.values()))

        loop_dims = {}
        if "batch" in observations.dims:
            loop_dims["batch"] = observations.batch.values
        if "branch" in observations.dims:
            loop_dims["branch"] = observations.branch.values
        for indices in dict_product(loop_dims):
            for dof in observations.dof.values:
                done = False
                a = observations.sel(**indices, dof=dof).values
                while not done:
                    delta = a[:-1] - a[1:]
                    maxind = np.argmax(np.abs(delta))
                    if np.abs(delta[maxind]) > np.pi / 2:
                        a[maxind + 1 :] += np.pi * np.sign(delta[maxind])
                    else:
                        done = True
        return observations

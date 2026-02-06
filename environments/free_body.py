import jax.numpy as jnp
from jax import vmap
from ajx import *
from ajx.simulation import SimulationSettings
from ajx.definitions import RigidBodyParameters, RigidBody
from ajx.environment import Environment

from util.deepinsert import deepinsert
import scenes.graphics.geometry as geometry

from typing import Dict
from ajx.param import SimulationParameters


class FreeBody(Environment):
    def __init__(self, sim_settings: SimulationSettings):

        self._build_sim(sim_settings)

        super().post_init()

    def get_hyperparam(self):
        return {
            "timestep": self.timestep,
            "use_gyroscopic": self.use_gyroscopic,
        }

    def _build_sim(self, sim_settings):
        inertia = jnp.array([1.0, 2.0, 4.0])
        extents = geometry.Box.extents_from_interia(inertia, 1.0)
        self.box = geometry.Box(
            "box",
            extents[0],
            extents[1],
            extents[2],
            (0.0, 0.0, 0.0),
            color=(0.9, 0.2, 0.2),
        )

        self.body = RigidBody("body", ("box",))
        body_param = RigidBodyParameters.create(
            mass=1.0, inertia_diag=inertia, name="body"
        )
        rb_param, rigid_bodies = RigidBodyParameters.stack_with_rigid_bodies(
            [(body_param, self.body)]
        )
        constraint_param, constraints = ConstraintParameters.stack_with_constraints([])

        pre_step_modifiers = ()

        rotation_encoder = AbsoluteRotationEncoder("rotation_encoder", "body")
        sensors = (rotation_encoder,)

        self.sim = Simulation(
            sim_settings,
            rigid_bodies,
            constraints,
            sensors,
            pre_step_modifiers,
        )
        self.default_param = SimulationParameters(
            jnp.array([0.0, 0.0, 0.0]),
            rb_param,
            constraint_param,
            sparse_param={},
        )

        self.geometry_list = (self.box,)
        self.extra_geometry = (
            geometry.Square(
                "ground", 1.5, 1.5, (0.0, -3.0, 0.0), color=(0.2, 0.5, 0.2)
            ),
        )

    def observation_to_configuration(self, observation, params):
        pos = jnp.zeros([1, 3])
        rot = observation[None]
        return Configuration(pos, rot)

    def state_from_angular_velocity(self, angvel):
        initial_conf = Configuration(
            jnp.array([[0.0, 0.0, 0.0]]), jnp.array([[1.0, 0.0, 0.0, 0.0]])
        )
        initial_linear_velocity = jnp.array([0.0, 0.0, 0.0])
        initial_gvel = jnp.concatenate([initial_linear_velocity, angvel])
        initial_gvel = GeneralizedVelocity(initial_gvel[None])
        return State(initial_conf, initial_gvel)

    def control_func(self, observation, last_observation, key_map):
        return jnp.array([0.0])

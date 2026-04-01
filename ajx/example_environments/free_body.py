import jax.numpy as jnp
from ajx import *
from ajx.simulation import SimulationSettings
from ajx.definitions import RigidBodyParameters, RigidBody
from ajx.example_environments.environment import Environment

import ajx.example_graphics.geometry as geometry

FreeBodySparseParam = create_parameter_node("FreeBodySparseParam", ())


class FreeBody(Environment):
    def __init__(self, sim_settings: SimulationSettings):

        self.timestep = sim_settings.timestep
        self.state_tangent_dim = 12 * 2
        self.sim_settings = sim_settings
        self.env_settings = None
        self._build_sim(sim_settings)
        self.dynamic_residual_names = self.get_state_residual_names()
        self.control_names = []
        self.initial_control_state = None

        super().post_init()

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
        rb_param = body_param
        rigid_bodies = (self.body,)
        constraint_param = ConstraintParameters.create_empty()
        constraints = ()

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
            sparse_param=FreeBodySparseParam(),
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

    def control_func(self, observation, last_observation, key_map, control_state):
        return jnp.array([0.0]), control_state

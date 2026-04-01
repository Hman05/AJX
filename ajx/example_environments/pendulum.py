import jax.numpy as jnp
from ajx.example_environments.environment import Environment
import ajx.simulation as simulation
from ajx import *

import ajx.example_graphics.geometry as geometry

PendulumSparseParamClass = create_parameter_node(
    "PendulumSparseParam",
    ("damping", "offset_param"),
)


class Pendulum(Environment):
    def __init__(
        self,
        sim_settings: SimulationSettings,
        has_quadratic_damping: bool,
    ):
        self.has_quadratic_damping = has_quadratic_damping
        self.timestep = sim_settings.timestep
        self.state_tangent_dim = 12
        self.sim_settings = sim_settings
        self.env_settings = has_quadratic_damping

        self._build_sim(sim_settings)
        self.dynamic_residual_names = self.get_state_residual_names()
        self.control_names = []
        self.initial_control_state = None

        super().post_init()

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

        self.rb_param = self.pendulum_param
        self.rigid_bodies = (self.pendulum,)

        self.hinge = OneBodyConstraint(
            name="hinge",
            body="pendulum",
            constraint_type=ConstraintType.HINGE.value,
        )

        frame_rotation = math.quat_from_axis_angle(jnp.array([0.0, 0.0, 1.0]), 0.0)

        self.hinge_param = ConstraintParameters.create(
            free_degree=5,
            frame_a=Frame(jnp.array([0.0, 0.0, 0.0]), frame_rotation),
            frame_b=Frame(jnp.array([0.0, 0.091, 0.0]), frame_rotation),
            compliance=1e-5,
            damping=2 * sim_settings.timestep,
            b=0.0004,
            name="hinge",
        )

        self.constraint_param = self.hinge_param
        self.constraints = (self.hinge,)

        self.damping_param = QuadraticDampingParameters(b=0.04, c=0.01)

        self.quadratic_damping = QuadraticDampingComponent("damping", self.hinge)

        self.pre_step_modifiers = ()
        if self.has_quadratic_damping:
            self.pre_step_modifiers.append(self.quadratic_damping)

        self.rotary_decoder = LinearEncoder("rotary_encoder", self.hinge)
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
            PendulumSparseParamClass(
                damping=self.damping_param,
                offset_param=OffsetParameters(
                    ("rotary_encoder"),
                    (0.0,),
                    (1.0,),
                ),
            ),
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

    def observation_to_configuration(self, observation, param):
        world_transform = Transform(
            jnp.array([0.0, 0.0, 0.0]), jnp.array([1.0, 0.0, 0.0, 0.0])
        )

        theta = observation[0]
        pendulum_transform = self.hinge.place_other(param, world_transform, theta)

        return Configuration.concatenate([pendulum_transform.to_configuration()])

    def state_from_angle(self, theta, param):
        initial_obs = jnp.stack([theta], axis=-1)
        initial_conf = self.observation_to_configuration(initial_obs, param)
        initial_gvel = GeneralizedVelocity(jnp.zeros([1, 6]))

        return State(initial_conf, initial_gvel)

    def unflatten(self, flat_state):
        sizes = jnp.array([1 * 3, 1 * 4, 1 * 6])

        offsets = jnp.cumulative_sum(sizes, include_initial=True)

        conf_pos = flat_state[offsets[0] : offsets[1]].reshape(1, 3)
        conf_rot = flat_state[offsets[1] : offsets[2]].reshape(1, 4)
        gvel = flat_state[offsets[2] : offsets[3]].reshape(1, 6)
        return State(
            Configuration(conf_pos, conf_rot),
            GeneralizedVelocity(gvel),
        )

    def control_func(self, observation, last_observation, key_map, control_state):
        return jnp.array([0.0]), control_state

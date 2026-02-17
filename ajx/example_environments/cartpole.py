import jax.numpy as jnp
from ajx import *
from ajx.example_environments.environment import Environment

import scenes.graphics.geometry as geometry


class CartPole(Environment):
    def __init__(self, sim_settings: SimulationSettings):

        self._build_sim(sim_settings)
        self.control_names = ["motor"]

        super().post_init()

    def _build_sim(self, sim_settings):
        thin = 0.1
        cart_z = 0.4
        cart_center = thin + cart_z / 2
        half_length = 1.24
        cart_half_height = 0.4
        self.cart_box = geometry.Box(
            "cart_box",
            0.5,
            cart_half_height,
            cart_z / 2,
            translation=(0.0, 0.0, thin + cart_z / 2),
            color=[0.8, 0.6, 0.6],
        )
        self.pendulum_box = geometry.Box(
            "pendulum_box",
            thin,
            half_length,
            thin,
            translation=(0.0, 0.0, 0.0),
            color=[0.0, 0.5, 0.5],
        )

        cart = RigidBody("cart", ("cart_box",))
        cart_param = RigidBodyParameters.create(
            mass=0.127, inertia_diag=jnp.array([0.02, 0.02, 0.02]), name="cart"
        )
        self.pendulum = RigidBody("pendulum", ("pendulum_box",))
        pendulum_param = RigidBodyParameters.create(
            mass=0.5, inertia_diag=jnp.array([0.02, 0.02, 0.02]), name="pendulum"
        )

        # Prismatic
        self.prismatic = OneBodyConstraint(
            name="prismatic",
            body="cart",
            constraint_type=ConstraintType.PRISMATIC.value,
        )
        motor_param = GainMotorParameters(0.04, 10.0)
        motor = GainMotor2("motor", self.prismatic, sim_settings.timestep, 0)
        prismatic_direction = math.quat_from_axis_angle(
            jnp.array([0.0, 0.0, 1.0]), jnp.pi / 2
        )
        prismatic_param = ConstraintParameters.create(
            frame_a=Frame(jnp.array([0.0, 0.0, 0.0]), prismatic_direction),
            frame_b=Frame(jnp.array([0.0, 0.0, 0.0]), prismatic_direction),
            compliance=1e-8,
            damping=2 * sim_settings.timestep,
            b=0.04,
            name="prismatic",
        )

        track_angle = 0.0
        hinge_cart_rotation = math.quat_from_axis_angle(
            jnp.array([0.0, 1.0, 0.0]), jnp.pi / 2
        )
        horizontal_rotation = math.quat_from_axis_angle(
            jnp.array([0.0, 1.0, 0.0]), track_angle
        )
        hinge_world_rotation = math.quat_mul(hinge_cart_rotation, horizontal_rotation)
        self.hinge = TwoBodyConstraint(
            name="hinge",
            body_a="cart",
            body_b="pendulum",
            constraint_type=ConstraintType.HINGE.value,
        )
        hinge_param = ConstraintParameters.create(
            frame_a=Frame(jnp.array([0.0, 0.0, 0.0]), hinge_world_rotation),
            frame_b=Frame(jnp.array([0.0, 1.24, 0.0]), hinge_cart_rotation),
            compliance=1e-8,
            damping=2 * sim_settings.timestep,
            b=0.04,
            name="hinge",
        )
        rb_param = RigidBodyParameters.concatenate([cart_param, pendulum_param])
        self.rigid_bodies = (cart, self.pendulum)
        constraint_param = ConstraintParameters.concatenate(
            [prismatic_param, hinge_param]
        )
        self.constraints = (self.prismatic, self.hinge)

        self.pre_step_modifiers = (motor,)

        self.distance_sensor = PrismaticEncoder("distance_sensor", self.prismatic)
        self.rotary_encoder = RotaryEncoderHingeMounted("rotary_encoder", self.hinge)

        self.sensors = (
            self.distance_sensor,
            self.rotary_encoder,
        )

        self.sim = Simulation(
            sim_settings,
            self.rigid_bodies,
            self.constraints,
            self.sensors,
            self.pre_step_modifiers,
        )

        SparseParamClass = create_parameter_node("CartPoleSparseParam", ("motor",))
        self.default_param = SimulationParameters(
            jnp.array([0.0, -9.82, 0.0]),
            rb_param,
            constraint_param,
            sparse_param=SparseParamClass(motor=motor_param),
        )

        self.geometry_list = (self.cart_box, self.pendulum_box)

        self.extra_geometry = (
            geometry.Box(
                "rail",
                30,
                0.2,
                0.05,
                translation=(0.0, -cart_half_height, cart_center),
                color=[0.4, 0.1, 0.1],
            ),
        )

    def observation_to_configuration(self, observation, param):
        world_transform = Transform(
            jnp.array([0.0, 0.0, 0.0]), jnp.array([1.0, 0.0, 0.0, 0.0])
        )

        x = observation[0]
        theta = observation[1]
        cart_transform = self.prismatic.place_other(param, world_transform, x)
        pendulum_transform = self.hinge.place_other(param, cart_transform, theta)
        return Configuration.concatenate(
            [cart_transform.to_configuration(), pendulum_transform.to_configuration()]
        )

    def state_from_angles(self, x, theta, param):
        initial_observations = jnp.stack([x, theta], axis=-1)

        initial_conf = self.observation_to_configuration(initial_observations, param)
        initial_gvel = GeneralizedVelocity(jnp.zeros([2, 6]))
        return State(initial_conf, initial_gvel)

    def control_func(self, observation, last_observation, keymap):
        if not keymap:
            return jnp.array([0.0])
        motor = 0.0
        if keymap["h"]:
            motor = 10.0
        elif keymap["l"]:
            motor = -10.0
        return jnp.array([motor])

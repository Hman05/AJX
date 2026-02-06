import jax.numpy as jnp
from ajx import *
from ajx.environment import Environment

import scipy
import scenes.graphics.geometry as geometry

from ajx.param import SimulationParameters


class CartPole(Environment):
    def __init__(self, sim_settings: SimulationSettings):

        self._build_sim(sim_settings)
        self.control_names = ["motor"]

        super().post_init()

    def get_hyperparam(self):
        return {
            "timestep": self.timestep,
        }

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
        self.prismatic = PrismaticJoint(
            name="prismatic",
            body_a=None,
            body_b="cart",
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
        self.hinge = HingeJoint(name="hinge", body_a="cart", body_b="pendulum")
        hinge_param = ConstraintParameters.create(
            frame_a=Frame(jnp.array([0.0, 0.0, 0.0]), hinge_world_rotation),
            frame_b=Frame(jnp.array([0.0, 1.24, 0.0]), hinge_cart_rotation),
            compliance=1e-8,
            damping=2 * sim_settings.timestep,
            b=0.04,
            name="hinge",
        )

        rb_param, self.rigid_bodies = RigidBodyParameters.stack_with_rigid_bodies(
            [
                (cart_param, cart),
                (pendulum_param, self.pendulum),
            ]
        )

        constraint_param, self.constraints = (
            ConstraintParameters.stack_with_constraints(
                [
                    (prismatic_param, self.prismatic),
                    (hinge_param, self.hinge),
                ]
            )
        )

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

        self.default_param = SimulationParameters(
            jnp.array([0.0, -9.82, 0.0]),
            rb_param,
            constraint_param,
            sparse_param={
                "motor": motor_param,
            },
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
        world_transform = Configuration(
            jnp.array([0.0, 0.0, 0.0]), jnp.array([1.0, 0.0, 0.0, 0.0])
        )

        x = observation[0]
        theta = observation[1]
        cart_transform = self.prismatic.place_other(param, world_transform, x)
        pendulum_transform = self.hinge.place_other(param, cart_transform, theta)
        return Configuration.stack([cart_transform, pendulum_transform])

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


def state_equation_matrices_cartpole(param):
    m1 = param["m1"]
    m2 = param["m2"]
    g = param["g"]
    l = param["l"]
    J = param["J2"]

    det = (m1 + m2) * J + m1 * m2 * l**2
    A32 = m2**2 * l**2 * g / det
    A42 = (m1 + m2) * m2 * l * g / det
    B3 = (J + m2 * l**2) / det
    B4 = m2 * l / det

    # A32 = m2*g/m1
    # A42 = (m1+m2)*g/(m2*l)
    # B3 = 1/m2
    # B4 = 1/(m2*l)

    A = jnp.zeros((4, 4))
    B = jnp.zeros((4, 1))
    A = A.at[0, 2].set(1)
    A = A.at[1, 3].set(1)
    A = A.at[2, 1].set(A32)
    A = A.at[3, 1].set(A42)
    B = B.at[2, 0].set(B3)
    B = B.at[3, 0].set(B4)
    return A, B


def compute_control_vector_cartpole(param):
    # Get matrices that describe the LQR problem
    A, B = state_equation_matrices_cartpole(param)
    Q = jnp.diag(jnp.array([1, 10, 1, 10]))
    r = 0.1

    # Solve Riccati equation and compute control vector.
    # The control vector should approximately be [0, 59.54303, -3.16229, 14.04044])
    P = scipy.linalg.solve_continuous_are(A, B, Q, jnp.array([[r]]))
    control_vector = (1 / r) * (B.T @ P)
    return control_vector

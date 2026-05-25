import jax.numpy as jnp
import ajx.math as math
import os
from ajx import *
from ajx.example_environments.environment import Environment

from typing import Optional
import ajx.example_graphics.geometry as geometry

DoublePendulumSparseParam = create_parameter_node(
    "DoublePendulumSparseParam", ("electric_motor", "offset_param")
)


class DoublePendulum(Environment):
    def __init__(
        self,
        sim_settings: SimulationSettings,
        reference_timestep: Optional[float] = None,
    ):
        # Double Pendulum

        self.n_control = 1
        self.timestep = sim_settings.timestep
        self.env_settings = reference_timestep

        self.reference_timestep = reference_timestep
        if not reference_timestep:
            self.reference_timestep = sim_settings.timestep

        self.camera_pos = jnp.array([0.0, 5.0, 0.0])
        self.camera_rot = math.quat_from_axis_angle(jnp.array([0.0, 0.0, 0.0]), 0.0)

        self.control_names = ["voltage"]
        self.state_tangent_dim = 2 * 12
        self.sim_settings = sim_settings
        self._build_sim(sim_settings)
        self.dynamic_residual_names = self.get_state_residual_names()
        self.initial_control_state = 0

        super().post_init()

    def _build_sim(self, sim_settings):
        # l_1 = 1.13665 m
        # l_2 = 1.28134 m
        l_1to1 = 1.13665
        l_1to2 = 1.06335
        l_2to1 = 1.28134

        script_dir = os.path.dirname(__file__)
        arm1_model = os.path.join(script_dir, "assets/double_pendulum_arm1.bam")
        arm2_model = os.path.join(script_dir, "assets/double_pendulum_arm2.bam")
        stand_model = os.path.join(script_dir, "assets/double_pendulum_stand.bam")

        arm1_model = geometry.Model(
            "arm1_model", arm1_model, rotation=math.Rotations.y_to_z
        )
        arm2_model = geometry.Model(
            "arm2_model", arm2_model, rotation=math.Rotations.y_to_z
        )

        arm1 = RigidBody("arm1", [("arm1_model", Transform.identity())])
        m1 = 1.0
        Jz = 0.5 * m1 * 0.1**2
        Jxy = 1 / 12 * m1 * (3 * 0.1**2 + l_1to1 + l_1to2)
        arm1_param = RigidBodyParameters.create(
            mass=m1,
            inertia_diag=jnp.array([Jxy, Jxy, Jz]),
            name="arm1",
        )

        m2 = 1.0
        Jz = 0.5 * m2 * 0.1**2
        Jxy = 1 / 12 * m2 * (3 * 0.1**2 + l_2to1 + 1.0)
        arm2 = RigidBody("arm2", [("arm2_model", Transform.identity())])
        arm2_param = RigidBodyParameters.create(
            mass=m2, inertia_diag=jnp.array([Jxy, Jxy, Jz]), name="arm2"
        )

        # enable_motor = not self.no_motor
        self.hinge1 = OneBodyConstraint(
            name="hinge1",
            # body_a=None,
            body="arm1",
            constraint_residual=ConstraintResidual.AXIAL_WORLD_SPHERICAL.value,
        )

        electric_motor_param = GainMotorParameters(0.0004, 75.0)  # 0.00265, 0.0039
        electric_motor = GainMotor(
            "electric_motor", self.hinge1, sim_settings.timestep, 0, 5
        )
        # self.electric_motor = TargetSpeedMotor("electric_motor", "hinge1_motor", 0)

        hinge1_param = ConstraintParameters.create(
            free_degree=5,
            frame_a=Frame(jnp.array([0.0, 0.0, 0.0]), math.Rotations.identity),
            frame_b=Frame(jnp.array([0.0, 0.0, -l_1to1]), math.Rotations.identity),
            compliance=1e-8,
            damping=2 * self.reference_timestep,
            b=4e-6,
            name="hinge1",
        )

        self.hinge2 = TwoBodyConstraint(
            name="hinge2",
            body_a="arm1",
            body_b="arm2",
            constraint_residual=ConstraintResidual.AXIAL_WORLD_SPHERICAL.value,
        )
        hinge2_param = ConstraintParameters.create(
            free_degree=5,
            frame_a=Frame(jnp.array([0.0, 0.0, l_1to2]), math.Rotations.identity),
            frame_b=Frame(jnp.array([0, 0.0, -l_2to1]), math.Rotations.identity),
            compliance=1e-8,
            damping=2 * self.reference_timestep,
            b=0.0003,
            name="hinge2",
        )

        rb_param = RigidBodyParameters.concatenate([arm1_param, arm2_param])
        rigid_bodies = (arm1, arm2)

        constraint_param = ConstraintParameters.concatenate(
            [hinge1_param, hinge2_param]
        )
        constraints = (self.hinge1, self.hinge2)

        pre_step_modifiers = (electric_motor,)

        rotary_decoder1 = RotaryEncoder("rotary_encoder1", self.hinge1)
        rotary_decoder2 = RotaryEncoder("rotary_encoder2", self.hinge2)

        sensors = (rotary_decoder1, rotary_decoder2)

        self.sim = Simulation(
            sim_settings,
            rigid_bodies,
            constraints,
            sensors,
            pre_step_modifiers,
        )
        # Specification of Maxon 218009
        # https://www.maxongroup.com/maxon/view/product/motor/dcmotor/re/re40/218009
        self.default_param = SimulationParameters(
            jnp.array([0.0, 0.0, -9.82]),
            rb_param,
            constraint_param,
            DoublePendulumSparseParam(
                electric_motor=electric_motor_param,
                offset_param=OffsetParameters(
                    ("rotary_encoder1", "rotary_encoder2"),
                    (0.0, 0.0),
                    (1.0, 1.0),
                ),
            ),
        )

        stand = geometry.Model(
            "stand",
            stand_model,
        )

        self.geometry_list = (arm1_model, arm2_model, stand)

        self.extra_geometry = [
            ("stand", Transform.identity()),
        ]

    def observation_to_configuration(self, observation, param):
        world_transform = Transform(
            jnp.array([0.0, 0.0, 0.0]), jnp.array([1.0, 0.0, 0.0, 0.0])
        )

        theta1 = observation[0]
        theta2 = observation[1]
        arm1_transform = self.hinge1.place_other(param, world_transform, theta1)
        arm2_transform = self.hinge2.place_other(5, param, arm1_transform, theta2)
        return Configuration.concatenate(
            [arm1_transform.to_configuration(), arm2_transform.to_configuration()]
        )

    def state_from_angles(self, theta1, theta2, param):
        initial_observations = jnp.stack([theta1, theta2], axis=-1)

        initial_conf = self.observation_to_configuration(initial_observations, param)
        initial_gvel = GeneralizedVelocity(jnp.zeros([2, 6]))

        # When using the PGS-solver with warm starting, multiplier size needs to be correctly specified for jax.jit compilation to work
        multipliers_size = self.get_multiplier_size()
        multipliers = jnp.zeros([multipliers_size])

        return State(initial_conf, initial_gvel, multipliers=multipliers)

    def control_func(self, observation, last_observation, keymap, control_state):
        if not keymap:
            return jnp.array([0.0])
        motor = 0.0
        if (keymap["l"] and keymap["h"]) or (
            keymap["arrow_left"] and keymap["arrow_right"]
        ):
            motor = 0.0
        elif keymap["h"] or keymap["arrow_left"]:
            motor = -8.0
        elif keymap["l"] or keymap["arrow_right"]:
            motor = 8.0
        return jnp.array([motor]), control_state

    def control_help_strings(self):
        return [
            "arrow right/l: move clockwise",
            "arrow left/h: move counterclockwise",
        ]

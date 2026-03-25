import jax.numpy as jnp
import ajx.math as math
import os
from ajx import *
from ajx.example_environments.environment import Environment

from typing import Optional
import ajx.example_graphics.geometry as geometry

FurutaSparseParam = create_parameter_node(
    "FurutaSparseParam", ("electric_motor", "gravity", "offset_param")
)


class Furuta(Environment):
    def __init__(
        self,
        sim_settings: SimulationSettings,
        reference_timestep: Optional[float] = None,
    ):
        self.n_control = 1
        self.timestep = sim_settings.timestep

        self.reference_timestep = reference_timestep
        if not reference_timestep:
            self.reference_timestep = sim_settings.timestep

        self.camera_pos = jnp.array([0.0, 5.0, 0.0])
        self.camera_rot = math.quat_from_axis_angle(
            jnp.array([1.0, 0.0, 0.0]), 0.5 * jnp.pi
        )

        self.control_names = ["voltage"]
        self.state_tangent_dim = 2 * 12
        self.settings = sim_settings
        self._build_sim(sim_settings)
        self.dynamic_residual_names = self.get_state_residual_names()
        self.initial_control_state = 0

        super().post_init()

    def _build_sim(self, sim_settings):
        com_displacement1 = 0.11
        com_displacement2 = 0.091
        length1 = 0.248
        length2 = 0.395

        script_dir = os.path.dirname(__file__)
        arm1_model = os.path.join(script_dir, "assets/arm1.bam")
        arm2_model = os.path.join(script_dir, "assets/arm2.bam")
        stand_model = os.path.join(script_dir, "assets/base.bam")

        arm1_box = geometry.Model(
            "arm1_box", arm1_model, translation=(-com_displacement1, 0.0, 0.0)
        )
        arm2_box = geometry.Model(
            "arm2_box", arm2_model, translation=(0.0, com_displacement2, 0.0)
        )

        arm1 = RigidBody("arm1", ("arm1_box",))
        arm1_param = RigidBodyParameters.create(
            mass=0.428,
            inertia_diag=jnp.array([1e-6, 0.012, 0.012]),
            name="arm1",
        )
        arm2 = RigidBody("arm2", ("arm2_box",))
        arm2_param = RigidBodyParameters.create(
            mass=0.238, inertia_diag=jnp.array([0.0016, 1e-6, 0.0016]), name="arm2"
        )

        # enable_motor = not self.no_motor
        self.hinge1 = OneBodyConstraint(
            name="hinge1",
            # body_a=None,
            body="arm1",
            constraint_type=ConstraintType.HINGE.value,
        )
        rotation1 = math.quat_from_axis_angle(jnp.array([0.0, 0.0, 1.0]), 0.5 * jnp.pi)
        rotation2 = math.quat_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), -0.0 * jnp.pi)
        rotation3 = rotation2

        electric_motor_param = GainMotorParameters(0.0004, 7.5)  # 0.00265, 0.0039
        electric_motor = GainMotor2(
            "electric_motor", self.hinge1, sim_settings.timestep, 0, 5
        )
        # self.electric_motor = TargetSpeedMotor("electric_motor", "hinge1_motor", 0)

        hinge1_param = ConstraintParameters.create(
            free_degree=5,
            frame_a=Frame(jnp.array([0.0, 0.0, 0.0]), rotation1),
            frame_b=Frame(jnp.array([-com_displacement1, 0.0, 0.0]), rotation1),
            compliance=1e-5,
            damping=2 * self.reference_timestep,
            b=4e-6,
            name="hinge1",
        )

        com_to_rod_end1 = length1 - com_displacement1
        com_to_rod_end2 = length2 - com_displacement2

        self.hinge2 = TwoBodyConstraint(
            name="hinge2",
            body_a="arm1",
            body_b="arm2",
            constraint_type=ConstraintType.HINGE.value,
        )
        hinge2_param = ConstraintParameters.create(
            free_degree=5,
            frame_a=Frame(jnp.array([com_to_rod_end1, 0.0, 0.0]), rotation3),
            frame_b=Frame(jnp.array([0, com_displacement2, 0.0]), rotation2),
            compliance=1e-5,
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
            jnp.array([0.0, -9.82, 0.0]),
            rb_param,
            constraint_param,
            FurutaSparseParam(
                electric_motor=electric_motor_param,
                gravity=TiltGravityParam(
                    jnp.array([0.0, -9.82, 0.0]),
                    jnp.array([1.0, 0.0, 0.0, 0.0]),
                ),
                offset_param=OffsetParameters(
                    ("rotary_encoder1", "rotary_encoder2"),
                    (0.0, 0.0),
                    (1.0, 1.0),
                ),
            ),
        )

        self.geometry_list = (arm1_box, arm2_box)

        self.axes_frame_a = geometry.Model(
            "axes_frame_a",
            "axes.bam",
            translation=-self.camera_pos,
            rotation=self.camera_rot,
        )

        self.extra_geometry = [
            self.axes_frame_a,
            geometry.Square(
                "ground",
                1.0,
                1.0,
                (0.0, -0.8183 - 0.0304 - 0.0071 - 0.01, 0.0),
                color=(0.3, 0.3, 0.4),
            ),
            geometry.Model(
                "stand",
                stand_model,
                translation=(0.0, -0.8183 - 0.0071 - 0.01, -0.06925),
            ),
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
        return State(initial_conf, initial_gvel)

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

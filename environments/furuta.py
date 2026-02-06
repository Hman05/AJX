import jax.numpy as jnp
import ajx.math as math
from ajx import *
from ajx.environment import Environment

from typing import Optional
import scenes.graphics.geometry as geometry

from ajx.param import SimulationParameters


class Furuta(Environment):
    def __init__(
        self,
        sim_settings: SimulationSettings,
        reference_timestep: Optional[float] = None,
    ):
        self.n_control = 1

        self.reference_timestep = reference_timestep
        if not reference_timestep:
            self.reference_timestep = sim_settings.timestep

        self.control_names = ["voltage"]
        self._build_sim(sim_settings)

        super().post_init()

    def get_hyperparam(self):
        return {
            "timestep": self.timestep,
            "reference_timestep": self.reference_timestep,
            "use_gyroscopic": self.use_gyroscopic,
        }

    def latex_param_names(pnames):
        pname_mapping = {
            "hinge1.b": "$b_1\\,(\\si{\\kilogram\\per\\second})$",
            "hinge2.b": "$b_2\\,(\\si{\\kilogram\\per\\second})$",
            "arm1.inertia.0": "$J_{\\tA xx}\\,(\\si{\\kilogram\\meter^2})$",
            "arm1.inertia.1": "$J_{\\tA zz}\\,(\\si{\\kilogram\\meter^2})$",
            "arm1.inertia.2": "$J_{\\tA yy}\\,(\\si{\\kilogram\\meter^2})$",
            "arm2.inertia.0": "$J_{\\tB yy}\\,(\\si{\\kilogram\\meter^2})$",
            "arm2.inertia.1": "$J_{\\tB xx}\\,(\\si{\\kilogram\\meter^2})$",
            "arm2.inertia.2": "$J_{\\tB zz}\\,(\\si{\\kilogram\\meter^2})$",
            "hinge1.dry_friction.mu": "$\\mu_1$",
            "hinge2.dry_friction.mu": "$\\mu_2$",
            "electric_motor.gain": "$\\kappa$",
        }
        return {name: pname_mapping.get(name, name) for name in pnames}

    def _build_sim(self, sim_settings):
        com_displacement1 = 0.11
        com_displacement2 = 0.091
        length1 = 0.248
        length2 = 0.395

        arm1_box = geometry.Model(
            "arm1_box", "arm1.bam", translation=(-com_displacement1, 0.0, 0.0)
        )
        arm2_box = geometry.Model(
            "arm2_box", "arm2.bam", translation=(0.0, com_displacement2, 0.0)
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
        self.hinge1 = HingeJoint(
            name="hinge1",
            body_a=None,
            body_b="arm1",
        )
        rotation1 = math.quat_from_axis_angle(jnp.array([0.0, 0.0, 1.0]), 0.5 * jnp.pi)
        rotation2 = math.quat_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), -0.0 * jnp.pi)
        rotation3 = rotation2

        electric_motor_param = GainMotorParameters(0.0004, 7.5)  # 0.00265, 0.0039
        electric_motor = GainMotor2(
            "electric_motor", self.hinge1, sim_settings.timestep, 0
        )
        # self.electric_motor = TargetSpeedMotor("electric_motor", "hinge1_motor", 0)

        hinge1_param = ConstraintParameters.create(
            frame_a=Frame(jnp.array([0.0, 0.0, 0.0]), rotation1),
            frame_b=Frame(jnp.array([-com_displacement1, 0.0, 0.0]), rotation1),
            compliance=1e-5,
            damping=2 * self.reference_timestep,
            b=4e-6,
            name="hinge1",
        )

        com_to_rod_end1 = length1 - com_displacement1
        com_to_rod_end2 = length2 - com_displacement2

        self.hinge2 = HingeJoint(
            name="hinge2",
            body_a="arm1",
            body_b="arm2",
        )
        hinge2_param = ConstraintParameters.create(
            frame_a=Frame(jnp.array([com_to_rod_end1, 0.0, 0.0]), rotation3),
            frame_b=Frame(jnp.array([0, com_displacement2, 0.0]), rotation2),
            compliance=1e-5,
            damping=2 * self.reference_timestep,
            b=0.0003,
            name="hinge2",
        )

        rb_param, rigid_bodies = RigidBodyParameters.stack_with_rigid_bodies(
            [
                (arm1_param, arm1),
                (arm2_param, arm2),
            ]
        )

        constraint_param, constraints = ConstraintParameters.stack_with_constraints(
            [
                (hinge1_param, self.hinge1),
                (hinge2_param, self.hinge2),
            ]
        )

        pre_step_modifiers = (electric_motor,)

        rotary_decoder1 = RotaryEncoderHingeMounted("rotary_encoder1", self.hinge1)
        rotary_decoder2 = RotaryEncoderHingeMounted("rotary_encoder2", self.hinge2)

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
            sparse_param={
                "electric_motor": electric_motor_param,
                "gravity": TiltGravityParam(
                    jnp.array([0.0, -9.82, 0.0]),
                    jnp.array([1.0, 0.0, 0.0, 0.0]),
                ),
            },
        )

        self.geometry_list = (arm1_box, arm2_box)

        self.extra_geometry = [
            geometry.Square(
                "ground",
                1.0,
                1.0,
                (0.0, -0.8183 - 0.0304 - 0.0071 - 0.01, 0.0),
                color=(0.3, 0.3, 0.4),
            ),
            geometry.Model(
                "stand",
                "base.bam",
                translation=(0.0, -0.8183 - 0.0071 - 0.01, -0.06925),
            ),
        ]

    def observation_to_configuration(self, observation, param):
        world_transform = Configuration(
            jnp.array([0.0, 0.0, 0.0]), jnp.array([1.0, 0.0, 0.0, 0.0])
        )

        theta1 = observation[0]
        theta2 = observation[1]
        arm1_transform = self.hinge1.place_other(param, world_transform, theta1)
        arm2_transform = self.hinge2.place_other(param, arm1_transform, theta2)
        return Configuration.stack([arm1_transform, arm2_transform])

    def preprocess_observations(observations):
        # TODO: Should be done per sensor. Not for each environment
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

    def state_from_angles(self, theta1, theta2, param):
        initial_observations = jnp.stack([theta1, theta2], axis=-1)

        initial_conf = self.observation_to_configuration(initial_observations, param)
        initial_gvel = GeneralizedVelocity(jnp.zeros([2, 6]))
        return State(initial_conf, initial_gvel)

    def control_func(self, observation, last_observation, key_map):
        return jnp.array([0.0])
        # Sensor dynamics and signal processing
        theta1 = observation[0]
        theta2 = observation[1]
        prev_theta1 = last_observation[0]
        prev_theta2 = last_observation[1]

        theta1_dot = (theta1 - prev_theta1) / self.timestep
        theta2_dot = -(theta2 - prev_theta2) / self.timestep

        # Control
        alpha = 0.75
        gml = 9.82 * self.param["arm2"].mass * self.com_displacement2  # 0.2286711
        joint_inertia2 = 0.0016 + 0.111 * 0.210**2

        joint_inertia2 = (
            self.param["arm2"].inertia[0]
            + self.param["arm2"].mass * self.com_displacement2**2
        )

        weighted_energy = alpha * 0.5 * joint_inertia2 * theta2_dot**2 - gml * (
            1 - jnp.cos(theta2)
        )
        gain = 40
        kp = 0.1
        u_energy = (
            gain * weighted_energy * theta2_dot * jnp.cos(theta2)
        ) - kp * theta1_dot
        if jnp.abs(theta1) < 0.4 and jnp.abs(theta2_dot) < 8:
            u_energy = 10
        u = jnp.clip(jnp.array([u_energy]), -5, 5)
        return -u

        state = jnp.stack([0, theta2, theta1_dot, theta2_dot])
        u_lqr = (control_vector @ state)[0]

        # Change controller depending on angle
        u = u_lqr * (jnp.abs(state[1]) < jnp.pi / 4) + u_energy * (
            jnp.abs(state[1]) >= jnp.pi / 4
        )
        u = jnp.clip(jnp.array([u]), -10, 10)
        return u

    def control_func(self, observation, last_observation, key_map):
        motor = 0.0
        if key_map["l"] and key_map["h"]:
            motor = 0.0
        elif key_map["h"]:
            motor = -4.0  # -0.5
        elif key_map["l"]:
            motor = 4.0  # 0.5
        return jnp.array([motor])

import jax.numpy as jnp
import ajx.math as math
import os
from ajx import *
from ajx.example_environments.environment import Environment

from typing import Optional
import ajx.example_graphics.geometry as geometry
import numpy as np


@dataclass
class DLOSettings:
    n_segments: int
    segment_halflength: float
    constraint_type: ConstraintType
    observation_type: str
    pose_estimates_at: List = ()
    loose_end: bool = False
    diameter: float = 0.032

    def create(
        n_segments,
        length,
        constraint_type,
        observation_type,
        pose_estimates_at=(),
        loose_end=False,
    ):
        segment_length = length / n_segments
        return DLOSettings(
            n_segments,
            0.5 * segment_length,
            constraint_type,
            observation_type,
            pose_estimates_at,
            loose_end,
        )


@struct.dataclass
class DLOState(ParameterNode):
    conf: Configuration
    gvel: GeneralizedVelocity
    lock_targets: jax.Array


@struct.dataclass
class CableParameters(ParameterNode):
    # Stretch, bend, torsion
    stiffness: jax.Array
    damping: jax.Array
    is_velocity: jax.Array


DLOSparseParam = create_parameter_node("DLOSparseParam", ("cable_param",))


@dataclass
class CoupleAsCable(PreStepModifier):
    name: str
    target_slice: Tuple
    body_ids: jax.Array
    constraint_ids: jax.Array
    constraint_type: ConstraintType
    segment_length: jax.Array

    def update_params(self, state: DLOState, u: jax.Array, param: SimulationParameters):
        cable_param: CableParameters = param.sparse_param.cable_param
        slice_begin = self.target_slice[0]
        slice_end = self.target_slice[1]
        constraint_param = param.constraint_param

        stiffness = jnp.array(
            [
                cable_param.stiffness[0] / self.segment_length,
                1e8,
                1e8,
                cable_param.stiffness[2] / self.segment_length,
                cable_param.stiffness[1] / self.segment_length,
                cable_param.stiffness[1] / self.segment_length,
            ]
        )
        constraint_param = constraint_param.replace(
            compliance=constraint_param.compliance.at[slice_begin:slice_end].set(
                1 / stiffness
            )
        )
        constraint_param = constraint_param.replace(
            damping=constraint_param.damping.at[slice_begin:slice_end].set(
                cable_param.damping
            )
        )
        constraint_param = constraint_param.replace(
            is_velocity=constraint_param.is_velocity.at[slice_begin:slice_end].set(
                cable_param.is_velocity
            )
        )
        new_param = param.replace(constraint_param=constraint_param)
        return state, new_param


@struct.dataclass
class LockAtZeroSpeedMotor(PreStepModifier):
    name: str
    constraint: Constraint
    u_idx: int
    lock_idx: int
    target_dof: int

    def update_params(self, state: DLOState, u: jax.Array, param: SimulationParameters):
        lock = u[self.u_idx] == 0.0
        not_lock = jnp.logical_not(lock)
        current_offset = self.constraint.func2(state, param)[self.target_dof]
        target = state.lock_targets[self.lock_idx] * lock + u[self.u_idx] * not_lock
        new_lock_target = (
            state.lock_targets[self.lock_idx] * lock + current_offset * not_lock
        )
        state = state.replace(
            lock_targets=state.lock_targets.at[self.lock_idx].set(new_lock_target)
        )
        param_w_is_velocity = param.tree_replace(
            {
                f"constraint_param.is_velocity.{self.constraint.name}": {
                    self.target_dof: not_lock
                },
            }
        )
        return state, (
            param_w_is_velocity.tree_replace(
                {
                    f"constraint_param.target.{self.constraint.name}": {
                        self.target_dof: target
                    }
                }
            )
        )


class DLO(Environment):
    def __init__(
        self,
        sim_settings: SimulationSettings,
        env_settings: DLOSettings,
    ):
        self.n_control = 1
        self.timestep = sim_settings.timestep
        self.env_settings = env_settings

        self.reference_timestep = sim_settings.timestep

        self.control_names = ["voltage"]
        self.state_tangent_dim = self.env_settings.n_segments * 12
        self.settings = sim_settings
        self._build_sim(sim_settings)
        self.dynamic_residual_names = self.get_state_residual_names()

        self.camera_pos = jnp.array(
            [
                self.env_settings.segment_halflength * self.env_settings.n_segments
                + 0.5 * 0.0795,
                1.8,
                0.0,
            ]
        )
        self.camera_rot = math.quat_from_axis_angle(jnp.array([0.0, 0.0, 1.0]), jnp.pi)
        self.initial_control_state = (False, False)

        super().post_init()

    def _build_sim(self, sim_settings):
        boxes = []
        arms = []
        arms_param = []
        self.lock_joints = []
        lock_joint_param = []
        gradient_start = jnp.array([1.0, 0.0, 0.0])
        gradient_end = jnp.array([0.0, 1.0, 1.0])
        n = self.env_settings.n_segments
        gradient = gradient_start - jnp.outer(
            jnp.arange(n), (gradient_start - gradient_end) / n
        )
        density = 10.0
        length = 2 * self.env_settings.segment_halflength * self.env_settings.n_segments
        grapple_box_length = 0.0795

        script_dir = os.path.dirname(__file__)
        open_cylinder_path = os.path.join(script_dir, "assets/capsule.bam")
        grip_tool_path = os.path.join(script_dir, "assets/grip_tool.bam")

        reference_box = geometry.Box(
            f"grip_tool1_box",
            grapple_box_length,
            0.6,
            0.3,
            translation=(0.0, 0.0, 0.0),
            color=(0.1, 0.1, 0.1),
        )
        grip_tool1_model = geometry.Model(
            f"grip_tool1_model",
            grip_tool_path,
            scale=(0.001, 0.001, 0.001),
            rotation=math.Rotations.x_to_y,
        )
        grip_tool2_model = geometry.Model(
            f"grip_tool2_model",
            grip_tool_path,
            scale=(0.001, 0.001, 0.001),
            rotation=math.Rotations.y_to_x,
        )
        grip_tool1 = RigidBody(f"grip_tool1", (f"grip_tool1_model",))
        grip_tool1_param = RigidBodyParameters.create(
            mass=density * 0.4 * 0.4 * grapple_box_length,
            inertia_diag=reference_box.get_diag_inertia(density),
            name="grip_tool1",
        )

        grip_tool2 = RigidBody(f"grip_tool2", (f"grip_tool2_model",))
        grip_tool2_param = RigidBodyParameters.create(
            mass=density * 0.4 * 0.4 * grapple_box_length,
            inertia_diag=reference_box.get_diag_inertia(density),
            name="grip_tool2",
        )
        for i in range(self.env_settings.n_segments):
            pose_at_body_ids = [
                int(self.env_settings.n_segments * disp / length)
                for disp in self.env_settings.pose_estimates_at
            ]

            color = tuple([*gradient[i]])
            if i in pose_at_body_ids:
                color = (0.0, 0.0, 0.1)
            box_old = geometry.Box(
                f"box",
                self.env_settings.segment_halflength,
                0.1,
                0.1,
                translation=(0.0, 0.0, 0.0),
                color=color,
            )
            box = geometry.Model(
                f"box{i}",
                open_cylinder_path,
                rotation=math.Rotations.y_to_x,
                scale=(
                    0.5 * self.env_settings.diameter,
                    self.env_settings.segment_halflength,
                    0.5 * self.env_settings.diameter,
                ),
                color=color,
            )
            boxes.append(box)
            mass = density * 0.1 * 0.1 * self.env_settings.segment_halflength
            inertia = box_old.get_diag_inertia(density)

            arms.append(RigidBody(f"body{i}", (f"box{i}",)))
            arms_param.append(
                RigidBodyParameters.create(
                    mass=mass,
                    inertia_diag=inertia,
                    name=f"body{i}",
                )
            )
        rotation1 = math.quat_from_axis_angle(
            jnp.array([-1.0, 0.0, 0.0]), -0.0 * jnp.pi
        )
        rotation2 = math.quat_from_axis_angle(
            jnp.array([-1.0, 0.0, 0.0]), -0.0 * jnp.pi
        )

        self.first_lock = OneBodyConstraint(
            name=f"grip_tool1_lock",
            body="grip_tool1",
            constraint_type=self.env_settings.constraint_type,
        )
        bl = self.env_settings.segment_halflength
        first_lock_param = ConstraintParameters.create_locked_ext(
            frame_a=Frame(jnp.array([0.0, 0.0, 0.0]), rotation1),
            frame_b=Frame(jnp.array([0.0, 0.0, 0.0]), rotation2),
            compliance_lin=1e-5,
            compliance_rot=1e-5,
            viscous_compliance_lin=1e-3,
            viscous_compliance_rot=1e-2,
            damping=2 * self.reference_timestep,
            offset=0.0,
            name="grip_tool1_lock",
        )
        # First lock joint
        self.lock_joints.append(
            TwoBodyConstraint(
                name=f"lock_g1_to_dlo",
                body_a=f"grip_tool1",
                body_b=f"body0",
                constraint_type=self.env_settings.constraint_type,
            )
        )
        # [0.531634 m, -0.008073 m, -79.5134] -> 0.0795
        lock_joint_param.append(
            ConstraintParameters.create_locked(
                frame_a=Frame(jnp.array([0.0795, 0.0, 0.0]), rotation1),
                frame_b=Frame(jnp.array([-bl, 0.0, 0.0]), rotation2),
                compliance=1e-5,
                viscous_compliance=1e-5,
                damping=2 * self.reference_timestep,
                offset=0.0,
                name=f"lock_g1_to_dlo",
            )
        )
        for i in range(0, self.env_settings.n_segments - 1):
            self.lock_joints.append(
                TwoBodyConstraint(
                    name=f"lock{i}",
                    body_a=f"body{i}",
                    body_b=f"body{i+1}",
                    constraint_type=self.env_settings.constraint_type,
                )
            )
            lock_joint_param.append(
                ConstraintParameters.create_locked(
                    frame_a=Frame(jnp.array([bl, 0.0, 0.0]), rotation1),
                    frame_b=Frame(jnp.array([-bl, 0.0, 0.0]), rotation2),
                    compliance=1e-5,
                    viscous_compliance=1e-5,
                    damping=2 * self.reference_timestep,
                    offset=0.0,
                    name=f"lock{i}",
                )
            )
        self.lock_joints.append(
            TwoBodyConstraint(
                name="lock_dlo_to_g2",
                body_a=f"body{self.env_settings.n_segments - 1}",
                body_b="grip_tool2",
                constraint_type=self.env_settings.constraint_type,
            )
        )
        lock_joint_param.append(
            ConstraintParameters.create_locked(
                frame_a=Frame(jnp.array([bl, 0.0, 0.0]), rotation1),
                frame_b=Frame(jnp.array([-grapple_box_length, 0.0, 0.0]), rotation2),
                compliance=1e-5,
                viscous_compliance=1e-5,
                damping=2 * self.reference_timestep,
                offset=0.0,
                name="lock_dlo_to_g2",
            )
        )
        self.last_lock = OneBodyConstraint(
            name=f"grip_tool2_lock",
            body=f"grip_tool2",
            constraint_type=self.env_settings.constraint_type,
        )
        last_lock_param = ConstraintParameters.create_locked_ext(
            frame_a=Frame(
                jnp.array(
                    [
                        bl * 2 * self.env_settings.n_segments + 2 * grapple_box_length,
                        0.0,
                        0.0,
                    ]
                ),
                rotation1,
            ),
            frame_b=Frame(jnp.array([0.0, 0.0, 0.0]), rotation2),
            compliance_lin=1e-5,
            compliance_rot=1e-5,
            viscous_compliance_lin=1e-3,
            viscous_compliance_rot=1e-2,
            damping=2 * self.reference_timestep,
            offset=0.0,
            name="grip_tool2_lock",
        )

        rb_param = RigidBodyParameters.concatenate(
            [grip_tool1_param, *arms_param, grip_tool2_param]
        )
        rigid_bodies = tuple([grip_tool1, *arms, grip_tool2])

        constraint_param = ConstraintParameters.concatenate(
            [first_lock_param, *lock_joint_param, last_lock_param]
        )
        constraints = tuple([self.first_lock, *self.lock_joints, self.last_lock])
        if self.env_settings.loose_end:
            constraints = tuple([self.first_lock, *self.lock_joints])

        target_speed_motor1 = LockAtZeroSpeedMotor("motor1", self.first_lock, 0, 0, 0)
        target_speed_motor2 = LockAtZeroSpeedMotor("motor2", self.first_lock, 1, 1, 1)
        target_speed_motor3 = LockAtZeroSpeedMotor("motor3", self.first_lock, 2, 2, 2)
        target_speed_motor4 = LockAtZeroSpeedMotor("motor4", self.first_lock, 3, 3, 3)
        target_speed_motor5 = LockAtZeroSpeedMotor("motor5", self.first_lock, 4, 4, 4)
        target_speed_motor6 = LockAtZeroSpeedMotor("motor6", self.first_lock, 5, 5, 5)

        target_speed_motor7 = LockAtZeroSpeedMotor("motor7", self.last_lock, 6, 6, 0)
        target_speed_motor8 = LockAtZeroSpeedMotor("motor8", self.last_lock, 7, 7, 1)
        target_speed_motor9 = LockAtZeroSpeedMotor("motor9", self.last_lock, 8, 8, 2)
        target_speed_motor10 = LockAtZeroSpeedMotor("motor10", self.last_lock, 9, 9, 3)
        target_speed_motor11 = LockAtZeroSpeedMotor(
            "motor11", self.last_lock, 10, 10, 4
        )
        target_speed_motor12 = LockAtZeroSpeedMotor(
            "motor12", self.last_lock, 11, 11, 5
        )

        # n_constraints = one per body + one
        n_segment_locks = self.env_settings.n_segments + 1
        couple_constraints = CoupleAsCable(
            "couple_constraints",
            target_slice=(1, n_segment_locks + 1),
            body_ids=jnp.stack(
                [
                    jnp.arange(0, n_segment_locks),
                    jnp.arange(1, n_segment_locks + 1),
                ],
                axis=-1,
            ),
            constraint_ids=jnp.arange(0, n_segment_locks)[:, None],
            constraint_type=self.env_settings.constraint_type,
            segment_length=self.env_settings.segment_halflength * 2,
        )

        pre_step_modifiers = (
            target_speed_motor1,
            target_speed_motor2,
            target_speed_motor3,
            target_speed_motor4,
            target_speed_motor5,
            target_speed_motor6,
            target_speed_motor7,
            target_speed_motor8,
            target_speed_motor9,
            target_speed_motor10,
            target_speed_motor11,
            target_speed_motor12,
            couple_constraints,
        )

        offsets = [
            jnp.array([0, 0.1, 0.1]),
            jnp.array([0, 0.1, -0.1]),
            jnp.array([0, -0.1, 0.1]),
            jnp.array([0, -0.1, -0.1]),
        ]

        # point_set = [(i, offset) for offset in offsets for i in range(n)]
        temp_limit = 1
        point_set = [
            (i + 1, offset) for i in range(max(n, temp_limit)) for offset in offsets
        ]
        # point_set5 = [(i, jnp.array([-bl, 0.1, 0.1])) for i in range(n)]
        # point_set6 = [(i, jnp.array([-bl, 0.1, -0.1])) for i in range(n)]
        # point_set7 = [(i, jnp.array([-bl, -0.1, 0.1])) for i in range(n)]
        # point_set8 = [(i, jnp.array([-bl, -0.1, -0.1])) for i in range(n)]
        camera_transform = Transform(
            jnp.array([bl * self.env_settings.n_segments, 0.0, 1.0]),
            math.quat_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), jnp.pi),
        )
        self.camera = PointTrackingCamera("camera", [*point_set], camera_transform)

        sensors = (self.camera,)

        self.sim = Simulation(
            sim_settings,
            rigid_bodies,
            constraints,
            sensors,
            pre_step_modifiers,
        )

        coupled_constraint_param = CableParameters(
            stiffness=jnp.array([1e5, 1e5, 1e5]),
            damping=jnp.ones(6) * 2 * self.sim.settings.timestep * 4,
            is_velocity=jnp.zeros(6, dtype=bool),
        )

        self.default_param = SimulationParameters(
            jnp.array([0.0, 0.0, -9.82]),
            rb_param,
            constraint_param,
            DLOSparseParam(coupled_constraint_param),
        )

        self.geometry_list = tuple([grip_tool1_model, *boxes, grip_tool2_model])

        self.extra_geometry = [
            geometry.Square(
                "ground",
                400.0,
                400.0,
                translation=(bl * self.env_settings.n_segments, 0.0, -100.0),
                rotation=math.quat_from_axis_angle(
                    jnp.array([1.0, 0.0, 0.0]), jnp.pi / 2
                ),
                color=(0.3, 0.3, 0.4),
            ),
        ]

    def observation_to_configuration(self, observation, param):
        world_transform = Transform(
            jnp.array([0.0, 0.0, 0.0]), jnp.array([1.0, 0.0, 0.0, 0.0])
        )

        body_transforms = []
        body_transforms.append(self.first_lock.place_other(param, world_transform, 0))
        for i in range(self.env_settings.n_segments):
            new_transform = self.lock_joints[i].place_other(
                0, param, body_transforms[-1], 0
            )
            body_transforms.append(new_transform)
        body_transforms.append(self.last_lock.place_other(param, world_transform, 0))
        return Configuration.concatenate(
            [body_transform.to_configuration() for body_transform in body_transforms]
        )

    def get_neutral_state(self, param):
        initial_conf = self.observation_to_configuration(None, param)
        n_bodies = self.env_settings.n_segments
        initial_gvel = GeneralizedVelocity(jnp.zeros([n_bodies + 2, 6]))
        targets = jnp.zeros([12])
        return DLOState(initial_conf, initial_gvel, targets)

    def control_help_strings(self):
        return [
            "h/l: left/right",
            "j/k: up/down",
            "u/i: in/out",
            "m/,: twist clockwise/counterclockwise",
            "y/n: tilt up/down",
            "6/7: tilt left/right",
            "8: hold to shift control target",
        ]

    def control_func(self, observation, last_observation, key_map, control_state):
        motor1 = 0.0
        motor2 = 0.0
        motor3 = 0.0
        motor4 = 0.0
        motor5 = 0.0
        motor6 = 0.0
        if (key_map["l"] and key_map["h"]) or (
            key_map["arrow_left"] and key_map["arrow_right"]
        ):
            motor1 = 0.0
        elif key_map["h"] or key_map["arrow_left"]:
            motor1 = 0.3  # -0.5
        elif key_map["l"] or key_map["arrow_right"]:
            motor1 = -0.3  # 0.5

        if (key_map["j"] and key_map["k"]) or (
            key_map["arrow_down"] and key_map["arrow_up"]
        ):
            motor3 = 0.0
        elif key_map["j"] or key_map["arrow_down"]:
            motor3 = -0.3
        elif key_map["k"] or key_map["arrow_up"]:
            motor3 = 0.3

        if key_map["u"] and key_map["i"]:
            motor2 = 0.0
        elif key_map["u"]:
            motor2 = -0.3
        elif key_map["i"]:
            motor2 = 0.3

        elif key_map["m"]:
            motor4 = -1.0
        elif key_map[","]:
            motor4 = 1.0

        elif key_map["y"]:
            motor5 = -1.0
        elif key_map["n"]:
            motor5 = 1.0

        elif key_map["6"]:
            motor6 = -1.0
        elif key_map["7"]:
            motor6 = 1.0
        motor1_to_6 = jnp.array([motor1, motor2, motor3, motor4, motor5, motor6])
        motor7_to_12 = jnp.zeros([6])

        control_first = control_state[0]
        switch_is_down = control_state[1]
        if key_map["8"] and not switch_is_down:
            switch_is_down = True
            control_first = not control_first
        if not key_map["8"] and switch_is_down:
            switch_is_down = False
        if control_first:
            motor_1_to_12 = jnp.concatenate([motor1_to_6, motor7_to_12])
        else:
            motor_1_to_12 = jnp.concatenate([motor7_to_12, motor1_to_6])
        control_state = (control_first, switch_is_down)
        return motor_1_to_12, control_state


# ui (right-left)
# ui (right-left)
# hjkl (left,right,up,down)

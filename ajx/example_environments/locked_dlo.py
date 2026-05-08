import jax.numpy as jnp
import ajx.math as math
import os
from ajx import *
from ajx.example_environments.environment import Environment

from typing import Optional
import ajx.example_graphics.geometry as geometry
import numpy as np
from ajx import Transform

from ajx.example_environments.dlo import (
    CableParameters,
    DLOSparseParam,
    DLOSettings,
    CoupleAsCable,
    DLOState,
    DLO,
)


@struct.dataclass
class LockAtZeroSpeedMotorSE3(PreStepModifier):
    name: str
    constraint: Constraint
    u_idx: int
    lock_idx: int

    def update_params(self, state: DLOState, u, param: SimulationParameters):
        lock = u[self.u_idx : self.u_idx + 6] == 0.0
        pos_lock = jnp.all(lock)
        rot_lock = jnp.all(lock)
        lock = jnp.concatenate(
            [pos_lock, pos_lock, pos_lock, rot_lock, rot_lock, rot_lock], axis=None
        )

        updated_frame = self.constraint.place_frame_a(state, param)

        target = u[self.u_idx : self.u_idx + 6] * jnp.logical_not(lock)

        new_frame_pos = state.lock_targets[self.lock_idx][
            :3
        ] * pos_lock + updated_frame.pos * jnp.logical_not(pos_lock)
        new_frame_rot = state.lock_targets[self.lock_idx][
            3:
        ] * rot_lock + updated_frame.rot * jnp.logical_not(rot_lock)
        new_frame = jnp.concatenate([new_frame_pos, new_frame_rot])
        state = state.replace(
            lock_targets=state.lock_targets.at[self.lock_idx].set(new_frame)
        )
        constraint_id = param.constraint_param.names.index(self.constraint.name)
        new_frames = param.constraint_param.frame_a.replace(
            position=param.constraint_param.frame_a.position.at[constraint_id].set(
                new_frame[:3]
            ),
            rotation=param.constraint_param.frame_a.rotation.at[constraint_id].set(
                new_frame[3:7]
            ),
        )
        return state, (
            param.tree_replace(
                {
                    f"constraint_param.is_velocity.{self.constraint.name}": jnp.logical_not(
                        lock
                    ),
                    f"constraint_param.target.{self.constraint.name}": target,
                    f"constraint_param.frame_a": new_frames,
                }
            )
        )


class LockedDLO(DLO):
    """
    Deformable Linear Object (DLO) with fixed boundary conditions.

    This is a constrained variant of the standard DLO model in which both end grippers
    are immobilized in the world frame. Their positions and orientations are fixed and
    cannot be actuated.

    Compared to the base DLO environment, this removes boundary actuation entirely,
    resulting in a system that evolves only through internal deformation dynamics of
    the segmented rod model.
    """

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

        self.camera_pos = jnp.array([0.0, 2.0, 0.0])
        self.camera_rot = jnp.array([1.0, 0.0, 0.0, 0.0])
        self.initial_control_state = (False, False)

        super().post_init()

    def _build_sim(self, sim_settings):
        arms = []
        arms_param = []
        self.lock_joints = []
        lock_joint_param = []
        bl = self.env_settings.segment_halflength
        grapple_box_length = 0.0795

        reference_box = geometry.Box(
            f"grip_tool1_box",
            grapple_box_length,
            0.3,
            0.15,
        )

        tool1_model_local_transform = Transform(
            jnp.array([0.0, 0.0, 0.0]), math.Rotations.x_to_y
        )
        tool2_model_local_transform = Transform(
            jnp.array([0.0, 0.0, 0.0]), math.Rotations.y_to_x
        )
        marker1_local_transform = self.env_settings.pose_estimate_offsets[0]
        marker2_local_transform = self.env_settings.pose_estimate_offsets[-1]
        tool1_to_dlo_frame = Transform(
            jnp.array([grapple_box_length, 0.0, 0.0]), math.Rotations.unitary
        )
        tool2_to_dlo_frame = Transform(
            jnp.array([-grapple_box_length, 0.0, 0.0]), math.Rotations.unitary
        )

        grip_tool1 = RigidBody(
            f"grip_tool1",
            [("grip_tool_model", tool1_model_local_transform)],
            [
                ("grip_tool_debug_model", tool1_model_local_transform),
                ("marker_model", marker1_local_transform),
                ("axes_model", tool1_to_dlo_frame),
            ],
        )
        density = self.env_settings.density
        grip_tool1_param = RigidBodyParameters.create(
            mass=density * 0.2 * 0.2 * grapple_box_length,
            inertia_diag=reference_box.get_diag_inertia(density),
            name="grip_tool1",
        )

        grip_tool2 = RigidBody(
            f"grip_tool2",
            [("grip_tool_model", tool2_model_local_transform)],
            [
                ("grip_tool_debug_model", tool2_model_local_transform),
                ("marker_model", marker2_local_transform),
                ("axes_model", tool2_to_dlo_frame),
            ],
        )
        grip_tool2_param = RigidBodyParameters.create(
            mass=density * 0.2 * 0.2 * grapple_box_length,
            inertia_diag=reference_box.get_diag_inertia(density),
            name="grip_tool2",
        )
        for i in range(self.env_settings.n_segments):
            frame_a_transform = Transform(
                jnp.array([bl, 0.0, 0.0]), math.Rotations.unitary
            )
            frame_b_transform = Transform(
                jnp.array([-bl, 0.0, 0.0]), math.Rotations.unitary
            )
            segment_geometry = [("segment_model", Transform.unitary())]
            debug_geometry = [
                ("axes_model", frame_a_transform),
                ("axes_model", frame_b_transform),
                ("segment_wireframe_model", Transform.unitary()),
            ]
            if f"body{i}" in self.env_settings.pose_estimate_bodies:
                segment_geometry = [
                    ("segment_model", Transform.unitary()),
                    ("marker_model", Transform.unitary()),
                ]
                debug_geometry = [
                    ("axes_model", frame_a_transform),
                    ("axes_model", frame_b_transform),
                    ("marker_model", Transform.unitary()),
                    ("segment_wireframe_model", Transform.unitary()),
                ]

            area = jnp.pi * (self.env_settings.radius) ** 2
            # Cylinder mass
            mass_cyl = (
                self.env_settings.density
                * area
                * self.env_settings.segment_halflength
                * 2
            )
            mass_sphere = (
                self.env_settings.density
                * 4
                * jnp.pi
                * (self.env_settings.radius) ** 3
                / 3
            )
            mass = mass_cyl + mass_sphere
            inertia_cyl_x = 0.5 * mass * self.env_settings.radius**2
            inertia_cyl_yz = (
                1
                / 12
                * mass
                * (
                    3 * self.env_settings.radius**2
                    + (self.env_settings.segment_halflength * 2) ** 2
                )
            )
            inertia = jnp.array([inertia_cyl_x, inertia_cyl_yz, inertia_cyl_yz])

            arms.append(RigidBody(f"body{i}", segment_geometry, debug_geometry))
            arms_param.append(
                RigidBodyParameters.create(
                    mass=mass,
                    inertia_diag=inertia,
                    name=f"body{i}",
                )
            )
        # Strange name is used for compatability with regular DLO
        self.first_lock = OneBodyConstraint(
            name=f"lock_hidden2a_to_gripper1",
            body="grip_tool1",
            constraint_type=self.env_settings.constraint_type,
        )
        first_lock_param = ConstraintParameters.create_locked_ext(
            frame_a=Frame(jnp.array([0.0, 0.0, 0.0]), math.Rotations.unitary),
            frame_b=Frame(jnp.array([0.0, 0.0, 0.0]), math.Rotations.unitary),
            compliance_lin=1e-12,
            compliance_rot=1e-12,
            viscous_compliance_lin=1e-3,
            viscous_compliance_rot=1e-2,
            damping=2 * self.reference_timestep,
            offset=0.0,
            name="lock_hidden2a_to_gripper1",
        )
        # First lock joint
        self.lock_joints.append(
            TwoBodyConstraint(
                name=f"lock_gripper1_to_dlo",
                body_a=f"grip_tool1",
                body_b=f"body0",
                constraint_type=self.env_settings.constraint_type,
            )
        )
        # [0.531634 m, -0.008073 m, -79.5134] -> 0.0795
        lock_joint_param.append(
            ConstraintParameters.create_locked(
                frame_a=Frame(tool1_to_dlo_frame.pos, tool1_to_dlo_frame.rot),
                frame_b=Frame(jnp.array([-bl, 0.0, 0.0]), math.Rotations.unitary),
                compliance=1e-8,
                viscous_compliance=1e-5,
                damping=2 * self.reference_timestep,
                offset=0.0,
                name=f"lock_gripper1_to_dlo",
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
                    frame_a=Frame(jnp.array([bl, 0.0, 0.0]), math.Rotations.unitary),
                    frame_b=Frame(jnp.array([-bl, 0.0, 0.0]), math.Rotations.unitary),
                    compliance=1e-5,
                    viscous_compliance=1e-5,
                    damping=2 * self.reference_timestep,
                    offset=0.0,
                    name=f"lock{i}",
                )
            )
        self.lock_joints.append(
            TwoBodyConstraint(
                name="lock_dlo_to_gripper2",
                body_a=f"body{self.env_settings.n_segments - 1}",
                body_b="grip_tool2",
                constraint_type=self.env_settings.constraint_type,
            )
        )
        lock_joint_param.append(
            ConstraintParameters.create_locked(
                frame_a=Frame(jnp.array([bl, 0.0, 0.0]), math.Rotations.unitary),
                frame_b=Frame(tool2_to_dlo_frame.pos, tool2_to_dlo_frame.rot),
                compliance=1e-8,
                viscous_compliance=1e-5,
                damping=2 * self.reference_timestep,
                offset=0.0,
                name="lock_dlo_to_gripper2",
            )
        )
        self.last_lock = OneBodyConstraint(
            name=f"lock_gripper2_to_hidden2b",
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
                math.Rotations.unitary,
            ),
            frame_b=Frame(jnp.array([0.0, 0.0, 0.0]), math.Rotations.unitary),
            compliance_lin=1e-12,
            compliance_rot=1e-12,
            viscous_compliance_lin=1e-3,
            viscous_compliance_rot=1e-2,
            damping=2 * self.reference_timestep,
            offset=0.0,
            name="lock_gripper2_to_hidden2b",
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

        target_speed_motor1 = LockAtZeroSpeedMotorSE3("motor1", self.first_lock, 0, 0)
        target_speed_motor2 = LockAtZeroSpeedMotorSE3("motor2", self.last_lock, 6, 1)

        self.cable = CoupleAsCable(
            "couple_constraints",
            constraint_offset=1,
            body_offset=1,
            n_segments=self.env_settings.n_segments,
            segment_length=self.env_settings.segment_halflength * 2,
            radius=self.env_settings.radius,
        )

        pre_step_modifiers = (
            target_speed_motor1,
            target_speed_motor2,
            self.cable,
        )

        sensor_list = []
        for i, body in enumerate(self.env_settings.pose_estimate_bodies):
            pose_encoder = PoseEncoder(
                f"pose_encoder{i:03d}",
                body,
                self.env_settings.pose_estimate_offsets[i],
            )
            sensor_list.append(pose_encoder)

        sensors = tuple(sensor_list)

        self.sim = Simulation(
            sim_settings,
            rigid_bodies,
            constraints,
            sensors,
            pre_step_modifiers,
        )

        coupled_constraint_param = CableParameters(
            youngs_modulus=1e8,
            shear_modulus=1e8 / (2 * (1 + 0.333)),
            damping=2 * self.sim.settings.timestep,
        )

        self.default_param = SimulationParameters(
            jnp.array([0.0, 0.0, -9.82]),
            rb_param,
            constraint_param,
            DLOSparseParam(coupled_constraint_param),
        )

        self.ground = geometry.Square(
            "ground",
            400.0,
            400.0,
            translation=(bl * self.env_settings.n_segments, 0.0, -100.0),
            rotation=math.quat_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), jnp.pi / 2),
            color=(0.3, 0.3, 0.4),
        )
        self.geometry_list = self._create_geometry()

        self.extra_geometry = [("ground", Transform.unitary())]

    def create_neutral_configuration(self, observation, param):
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
        initial_conf = self.create_neutral_configuration(None, param)
        n_bodies = self.env_settings.n_segments
        initial_gvel = GeneralizedVelocity(jnp.zeros([n_bodies + 2, 6]))
        targets = jnp.stack(
            [
                param.constraint_param.frame_a[0].flatten(),
                param.constraint_param.frame_a[-1].flatten(),
            ]
        )
        return DLOState(initial_conf, initial_gvel, targets)

    def control_help_strings(self):
        return []

    def control_func(self, observation, last_observation, key_map, control_state):
        return jnp.zeros([12]), control_state

    def _place_hidden_links(self, interp_pos, interp_rot, param, state0):
        gripper1_transform = Transform(interp_pos[0], interp_rot[0])
        gripper2_transform = Transform(interp_pos[-1], interp_rot[-1])

        new_conf = Configuration(
            pos=interp_pos,
            rot=interp_rot,
        )

        targets = jnp.stack(
            [
                jnp.concatenate([gripper1_transform.pos, gripper1_transform.rot]),
                jnp.concatenate([gripper2_transform.pos, gripper2_transform.rot]),
            ]
        )

        new_state = state0.replace(conf=new_conf, lock_targets=targets)
        return new_state

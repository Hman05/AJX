import jax.numpy as jnp
import ajx.math as math
import os
from ajx import *
from ajx.example_environments.environment import Environment

from typing import Optional
import ajx.example_graphics.geometry as geometry
import numpy as np
from ajx import Transform


@dataclass
class DLOSettings:
    n_segments: int
    segment_halflength: float
    radius: float
    constraint_type: ConstraintType
    density: float
    pose_estimate_bodies: List[str] = ()
    pose_estimate_constraints_a: List[str] = ()
    pose_estimate_constraints_b: List[str] = ()
    pose_estimate_offsets: List[Transform] = ()
    loose_end: bool = False

    def create(
        n_segments: int,
        length: float,
        radius: float,
        density: float,
        pose_estimate_linear_offsets: List[float],
        gripper1_offset: Transform,
        gripper2_offset: Transform,
        loose_end: bool = False,
    ):
        segment_length = length / n_segments

        pose_estimate_bodies = []
        pose_estimate_constraints_a = []
        pose_estimate_constraints_b = []
        pose_estimate_offsets = []
        pose_estimate_bodies.append("grip_tool1")
        pose_estimate_offsets.append(gripper1_offset)
        pose_estimate_constraints_a.append("lock_gripper1_to_dlo")
        # pose_estimate_constraints_b.append("lock_hidden2a_to_gripper1")

        unit_transform = Transform(
            jnp.array([0.0, 0.0, 0.0]), jnp.array([1.0, 0.0, 0.0, 0.0])
        )
        for displacement in pose_estimate_linear_offsets:
            i = int(n_segments * displacement / length)
            pose_estimate_bodies.append(f"body{i}")
            pose_estimate_offsets.append(unit_transform)

            if i == n_segments - 1:
                pose_estimate_constraints_a.append("lock_dlo_to_gripper2")
            else:
                pose_estimate_constraints_a.append(f"lock{i}")
            if i == 0:
                pose_estimate_constraints_b.append(f"lock_gripper1_to_dlo")
            else:
                pose_estimate_constraints_b.append(f"lock{i-1}")

        pose_estimate_bodies.append("grip_tool2")
        pose_estimate_offsets.append(gripper2_offset)
        # pose_estimate_constraints_a.append("lock_gripper2_to_hidden2b")
        pose_estimate_constraints_b.append("lock_dlo_to_gripper2")
        return DLOSettings(
            n_segments,
            0.5 * segment_length,
            radius,
            ConstraintType.SE3.value,
            density,
            pose_estimate_bodies,
            pose_estimate_constraints_a,
            pose_estimate_constraints_b,
            pose_estimate_offsets,
            loose_end,
        )


@struct.dataclass
class DLOState(ParameterNode):
    conf: Configuration
    gvel: GeneralizedVelocity
    lock_targets: jax.Array
    multipliers: jax.Array = struct.field(default_factory=lambda: jnp.zeros([0]))

    tangent_restrictions: Tuple[str, ...] = struct.field(
        pytree_node=False, default=tuple(["conf", "gvel", "lock_targets"])
    )


@struct.dataclass
class CableParameters(ParameterNode):
    youngs_modulus: float
    shear_modulus: float
    damping: float

    def get_stiffness(self, radius, segment_length):
        area = radius * 2 * jnp.pi
        area_moment = jnp.pi * radius**4 / 4
        polar_moment = jnp.pi * radius**4 / 2

        E = self.youngs_modulus
        G = self.shear_modulus

        stretch_stiffness = E * area / segment_length
        bend_stiffness = E * area_moment / segment_length
        twist_stiffness = G * polar_moment / segment_length
        return stretch_stiffness, bend_stiffness, twist_stiffness


DLOSparseParam = create_parameter_node("DLOSparseParam", ("cable_param",))


@dataclass
class CoupleAsCable(PreStepModifier):
    name: str
    constraint_offset: float
    body_offset: float
    n_segments: float
    segment_length: jax.Array
    radius: jax.Array

    def update_params(self, state: DLOState, u: jax.Array, param: SimulationParameters):
        cable_param: CableParameters = param.sparse_param.cable_param
        n_constraints = self.n_segments + 1
        slice_begin = self.constraint_offset
        slice_end = self.constraint_offset + n_constraints
        constraint_param = param.constraint_param

        area = self.radius * 2 * jnp.pi
        area_moment = jnp.pi * self.radius**4 / 4
        polar_moment = jnp.pi * self.radius**4 / 2

        E = cable_param.youngs_modulus
        G = cable_param.shear_modulus

        stretch_stiffness = E * area / self.segment_length
        bend_stiffness = E * area_moment / self.segment_length
        twist_stiffness = G * polar_moment / self.segment_length
        shear_stiffness = 1e6  # G * area / self.segment_length

        stiffness = jnp.array(
            [
                stretch_stiffness,
                shear_stiffness,
                shear_stiffness,
                twist_stiffness,
                bend_stiffness,
                bend_stiffness,
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
        new_param = param.replace(constraint_param=constraint_param)
        return state, new_param


@struct.dataclass
class LockAtZeroSpeedMotor(PreStepModifier):
    name: str
    constraint: Constraint
    lock_degrees: List[int]
    u_idx: int
    lock_idx: int

    def update_params(self, state: DLOState, u: jax.Array, param: SimulationParameters):
        num_lock_deg = len(self.lock_degrees)
        lock = u[self.u_idx : self.u_idx + num_lock_deg] == 0.0
        not_lock = jnp.logical_not(lock)

        current_offset = self.constraint.object_func(state, param)[self.lock_degrees]
        target = (
            state.lock_targets[self.lock_idx : self.lock_idx + num_lock_deg] * lock
            + u[self.u_idx : self.u_idx + num_lock_deg] * not_lock
        )
        new_lock_target = (
            state.lock_targets[self.lock_idx : self.lock_idx + num_lock_deg] * lock
            + current_offset * not_lock
        )
        state = state.replace(
            lock_targets=state.lock_targets.at[
                self.lock_idx : self.lock_idx + num_lock_deg
            ].set(new_lock_target)
        )
        constraint_id = param.constraint_param.names.index(self.constraint.name)
        return state, (
            param.tree_replace(
                {
                    f"constraint_param.is_velocity": param.constraint_param.is_velocity.at[
                        constraint_id, self.lock_degrees
                    ].set(
                        not_lock
                    ),
                    f"constraint_param.target": param.constraint_param.target.at[
                        constraint_id, self.lock_degrees
                    ].set(target),
                }
            )
        )


class DLO(Environment):
    """
    Deformable Linear Object (DLO) environment controlled by two grippers.

    This environment models a deformable linear object (e.g., an elastic beam, rod, or cable)
    using a rigid-body segment approximation. The DLO is discretized into multiple segments,
    enabling approximate simulation of continuous deformation dynamics.

    The environment provides pose encoders that return the translation and rotation of each
    segment along the DLO. These observations can be used for calibration of real-world DLOs instrumented with corresponding markers.

    The environment is controlled by a 12-dimensional action vector representing two grippers.
    Each gripper contributes a 6-DOF command consisting of 3D position (x, y, z) and
    3D orientation (yaw, pitch, roll).
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

    def _create_geometry(self):
        """Create and configure geometry models used in the environment"""

        script_dir = os.path.dirname(__file__)
        capsule_path = os.path.join(script_dir, "assets/capsule.bam")
        hex_wireframe_path = os.path.join(
            script_dir, "assets/hex_cylinder_wireframe.bam"
        )
        grip_tool_path = os.path.join(script_dir, "assets/grip_tool.bam")
        grip_tool_debug_path = os.path.join(
            script_dir, "assets/grip_tool_wireframe.bam"
        )
        axes_path = os.path.join(script_dir, "assets/axes.glb")
        marker_debug_path = os.path.join(script_dir, "assets/cube_wireframe.glb")

        grip_tool_model = geometry.Model(
            f"grip_tool_model",
            grip_tool_path,
            scale=(0.001, 0.001, 0.001),
        )
        grip_tool_debug_model = geometry.Model(
            f"grip_tool_debug_model",
            grip_tool_debug_path,
            scale=(0.001, 0.001, 0.001),
        )
        marker_model = geometry.Model(
            f"marker_model",
            marker_debug_path,
            scale=(0.02, 0.02, 0.02),
        )
        frame_model = geometry.Model(
            f"axes_model",
            axes_path,
            scale=(0.03, 0.03, 0.03),
            rotation=math.Rotations.z_to_y,
        )
        segment_model = geometry.Model(
            f"segment_model",
            capsule_path,
            rotation=math.Rotations.y_to_x,
            scale=(
                self.env_settings.radius,
                self.env_settings.segment_halflength,
                self.env_settings.radius,
            ),
            color=(0.1, 0.1, 0.5),
        )
        segment_wireframe_model = geometry.Model(
            f"segment_wireframe_model",
            hex_wireframe_path,
            rotation=math.Rotations.y_to_x,
            scale=(
                self.env_settings.radius,
                self.env_settings.segment_halflength,
                self.env_settings.radius,
            ),
            color=(0.0, 0.0, 0.0),
        )
        bl = self.env_settings.segment_halflength
        ground = geometry.Square(
            "ground",
            400.0,
            400.0,
            translation=(bl * self.env_settings.n_segments, 0.0, -100.0),
            rotation=math.quat_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), jnp.pi / 2),
            color=(0.3, 0.3, 0.4),
        )
        return tuple(
            [
                grip_tool_model,
                grip_tool_debug_model,
                marker_model,
                frame_model,
                segment_model,
                segment_wireframe_model,
                frame_model,
                ground,
            ]
        )

    def _build_sim(self, sim_settings):
        arms = []
        arms_param = []
        self.lock_joints = []
        lock_joint_param = []
        bl = self.env_settings.segment_halflength
        self.grapple_box_length = 0.0795
        grapple_box_length = self.grapple_box_length

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

        density = self.env_settings.density

        # Hidden links to grip tool 1
        hidden_link1a = RigidBody(f"hidden_link1a", [], [])
        hidden_link1a_param = RigidBodyParameters.create(
            mass=1.0,
            inertia_diag=jnp.array([1.0, 1.0, 1.0]),
            name="hidden_link1a",
        )
        hidden_link2a = RigidBody(f"hidden_link2a", [], [])
        hidden_link2a_param = RigidBodyParameters.create(
            mass=1.0,
            inertia_diag=jnp.array([1.0, 1.0, 1.0]),
            name="hidden_link2a",
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
        grip_tool1_param = RigidBodyParameters.create(
            mass=density * 0.2 * 0.2 * grapple_box_length,
            inertia_diag=reference_box.get_diag_inertia(density),
            name="grip_tool1",
        )

        # Hidden links to grip tool 2
        hidden_link1b = RigidBody(f"hidden_link1b", [], [])
        hidden_link1b_param = RigidBodyParameters.create(
            mass=1.0,
            inertia_diag=jnp.array([1.0, 1.0, 1.0]),
            name="hidden_link1b",
        )
        hidden_link2b = RigidBody(f"hidden_link2b", [], [])
        hidden_link2b_param = RigidBodyParameters.create(
            mass=1.0,
            inertia_diag=jnp.array([1.0, 1.0, 1.0]),
            name="hidden_link2b",
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
            offset_a = bl
            offset_b = -bl
            frame_a_transform = Transform(
                jnp.array([offset_a, 0.0, 0.0]), math.Rotations.unitary
            )
            frame_b_transform = Transform(
                jnp.array([offset_b, 0.0, 0.0]), math.Rotations.unitary
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
            mass = mass_cyl
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

        ### Constraints

        # Locks [world -> hidden_link1a -> hidden_link2a -> gripper1]
        self.lock_world_to_hidden1a = OneBodyConstraint(
            name=f"lock_world_to_hidden1a",
            body="hidden_link1a",
            constraint_type=ConstraintType.HINGE.value,
        )
        lock_world_to_hidden1_param = ConstraintParameters.create_locked_ext(
            frame_a=Frame(jnp.array([0.0, 0.0, 0.0]), math.Rotations.x_to_z),
            frame_b=Frame(jnp.array([0.0, 0.0, 0.0]), math.Rotations.x_to_z),
            compliance_lin=1e-8,
            compliance_rot=1e-8,
            viscous_compliance_lin=1e-3,
            viscous_compliance_rot=1e-2,
            damping=2 * self.reference_timestep,
            offset=0.0,
            name="lock_world_to_hidden1a",
        )
        self.lock_hidden1a_to_hidden2a = TwoBodyConstraint(
            name=f"lock_hidden1a_to_hidden2a",
            body_a=f"hidden_link1a",
            body_b=f"hidden_link2a",
            constraint_type=ConstraintType.HINGE.value,
        )
        lock_hidden1a_to_hidden2a_param = ConstraintParameters.create_locked(
            frame_a=Frame(jnp.array([0.0, 0.0, 0.0]), math.Rotations.x_to_y),
            frame_b=Frame(jnp.array([0.0, 0.0, 0.0]), math.Rotations.x_to_y),
            compliance=1e-8,
            viscous_compliance=1e-5,
            damping=2 * self.reference_timestep,
            offset=0.0,
            name=f"lock_hidden1a_to_hidden2a",
        )
        self.lock_hidden2a_to_gripper1 = TwoBodyConstraint(
            name=f"lock_hidden2a_to_gripper1",
            body_a=f"hidden_link2a",
            body_b=f"grip_tool1",
            constraint_type=ConstraintType.HINGE.value,
        )
        lock_hidden2a_to_gripper1_param = ConstraintParameters.create_locked(
            frame_a=Frame(jnp.array([0.0, 0.0, 0.0]), math.Rotations.unitary),
            frame_b=Frame(jnp.array([0.0, 0.0, 0.0]), math.Rotations.unitary),
            compliance=1e-8,
            viscous_compliance=1e-5,
            damping=2 * self.reference_timestep,
            offset=0.0,
            name=f"lock_hidden2a_to_gripper1",
        )

        # Lock joint between gripper and first DLO segment
        self.lock_joints.append(
            TwoBodyConstraint(
                name=f"lock_gripper1_to_dlo",
                body_a=f"grip_tool1",
                body_b=f"body0",
                constraint_type=self.env_settings.constraint_type,
            )
        )
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
            offset_a = bl
            offset_b = -bl
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
                    frame_a=Frame(
                        jnp.array([offset_a, 0.0, 0.0]), math.Rotations.unitary
                    ),
                    frame_b=Frame(
                        jnp.array([offset_b, 0.0, 0.0]), math.Rotations.unitary
                    ),
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

        # Locks [gripper2 -> hidden_link2b -> hidden_link1b -> world]
        self.lock_gripper2_to_hidden_link2b = TwoBodyConstraint(
            name=f"lock_gripper2_to_hidden2b",
            body_a=f"grip_tool2",
            body_b=f"hidden_link2b",
            constraint_type=ConstraintType.HINGE.value,
        )
        lock_gripper2_to_hidden_link2b_param = ConstraintParameters.create_locked(
            frame_a=Frame(jnp.array([0.0, 0.0, 0.0]), math.Rotations.unitary),
            frame_b=Frame(jnp.array([0.0, 0.0, 0.0]), math.Rotations.unitary),
            compliance=1e-8,
            viscous_compliance=1e-5,
            damping=2 * self.reference_timestep,
            offset=0.0,
            name=f"lock_gripper2_to_hidden2b",
        )
        self.lock_hidden2b_to_hidden1b = TwoBodyConstraint(
            name=f"lock_hidden2b_to_hidden1b",
            body_a=f"hidden_link2b",
            body_b=f"hidden_link1b",
            constraint_type=ConstraintType.HINGE.value,
        )
        lock_hidden2b_to_hidden1b_param = ConstraintParameters.create_locked(
            frame_a=Frame(jnp.array([0.0, 0.0, 0.0]), math.Rotations.x_to_y),
            frame_b=Frame(jnp.array([0.0, 0.0, 0.0]), math.Rotations.x_to_y),
            compliance=1e-8,
            viscous_compliance=1e-5,
            damping=2 * self.reference_timestep,
            offset=0.0,
            name=f"lock_hidden2b_to_hidden1b",
        )

        self.lock_hidden1b_to_world = OneBodyConstraint(
            name=f"lock_hidden1b_to_world",
            body=f"hidden_link1b",
            constraint_type=ConstraintType.HINGE.value,
        )
        # Locked to the world with a special offset (total length)
        lock_hidden1b_to_world_param = ConstraintParameters.create_locked_ext(
            frame_a=Frame(
                jnp.array(
                    [
                        0.0,  # bl * 2 * self.env_settings.n_segments + 2 * grapple_box_length,
                        0.0,
                        0.0,
                    ]
                ),
                math.Rotations.x_to_z,
            ),
            frame_b=Frame(jnp.array([0.0, 0.0, 0.0]), math.Rotations.x_to_z),
            compliance_lin=1e-8,
            compliance_rot=1e-8,
            viscous_compliance_lin=1e-3,
            viscous_compliance_rot=1e-2,
            damping=2 * self.reference_timestep,
            offset=0.0,
            name="lock_hidden1b_to_world",
        )

        rb_param = RigidBodyParameters.concatenate(
            [
                hidden_link1a_param,
                hidden_link2a_param,
                grip_tool1_param,
                *arms_param,
                grip_tool2_param,
                hidden_link2b_param,
                hidden_link1b_param,
            ]
        )
        rigid_bodies = tuple(
            [
                hidden_link1a,
                hidden_link2a,
                grip_tool1,
                *arms,
                grip_tool2,
                hidden_link2b,
                hidden_link1b,
            ]
        )

        constraint_param = ConstraintParameters.concatenate(
            [
                lock_world_to_hidden1_param,
                lock_hidden1a_to_hidden2a_param,
                lock_hidden2a_to_gripper1_param,
                *lock_joint_param,
                lock_gripper2_to_hidden_link2b_param,
                lock_hidden2b_to_hidden1b_param,
                lock_hidden1b_to_world_param,
            ]
        )
        constraints = tuple(
            [
                self.lock_world_to_hidden1a,
                self.lock_hidden1a_to_hidden2a,
                self.lock_hidden2a_to_gripper1,
                *self.lock_joints,
                self.lock_gripper2_to_hidden_link2b,
                self.lock_hidden2b_to_hidden1b,
                self.lock_hidden1b_to_world,
            ]
        )
        if self.env_settings.loose_end:
            constraints = tuple(
                [
                    self.lock_world_to_hidden1a,
                    self.lock_hidden1a_to_hidden2a,
                    self.lock_hidden2a_to_gripper1,
                    *self.lock_joints,
                ]
            )

        pos_yaw_degrees = jnp.array([0, 1, 2, 5])
        hinge_degree = jnp.array([5])
        target_speed_motor1 = LockAtZeroSpeedMotor(
            "motor1_pos_yaw", self.lock_world_to_hidden1a, pos_yaw_degrees, 0, 0
        )
        target_speed_motor2 = LockAtZeroSpeedMotor(
            "motor1_pitch", self.lock_hidden1a_to_hidden2a, hinge_degree, 4, 4
        )
        target_speed_motor3 = LockAtZeroSpeedMotor(
            "motor1_roll", self.lock_hidden2a_to_gripper1, hinge_degree, 5, 5
        )

        target_speed_motor4 = LockAtZeroSpeedMotor(
            "motor2_pos_yaw", self.lock_hidden1b_to_world, pos_yaw_degrees, 6, 6
        )
        target_speed_motor5 = LockAtZeroSpeedMotor(
            "motor2_pitch", self.lock_hidden2b_to_hidden1b, hinge_degree, 10, 10
        )
        target_speed_motor6 = LockAtZeroSpeedMotor(
            "motor2_roll", self.lock_gripper2_to_hidden_link2b, hinge_degree, 11, 11
        )

        self.cable = CoupleAsCable(
            "couple_constraints",
            constraint_offset=constraint_param.names.index("lock_gripper1_to_dlo"),
            body_offset=rb_param.names.index("body0"),
            n_segments=self.env_settings.n_segments,
            segment_length=self.env_settings.segment_halflength * 2,
            radius=self.env_settings.radius,
        )

        pre_step_modifiers = (
            target_speed_motor1,
            target_speed_motor2,
            target_speed_motor3,
            target_speed_motor4,
            target_speed_motor5,
            target_speed_motor6,
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

        self.geometry_list = self._create_geometry()

        self.extra_geometry = [
            ("ground", Transform.unitary()),
        ]

    def create_neutral_configuration(self, observation, param):
        # TODO: Function name is misleading
        bl = self.env_settings.segment_halflength
        world_transform = Transform(
            jnp.array(
                [
                    -bl * self.env_settings.n_segments - self.grapple_box_length,
                    0.0,
                    0.0,
                ]
            ),
            jnp.array([1.0, 0.0, 0.0, 0.0]),
        )

        body_transforms = []
        body_transforms.append(
            self.lock_world_to_hidden1a.place_other(param, world_transform, 0)
        )
        body_transforms.append(
            self.lock_hidden1a_to_hidden2a.place_other(0, param, world_transform, 0)
        )
        body_transforms.append(
            self.lock_hidden2a_to_gripper1.place_other(0, param, body_transforms[-1], 0)
        )
        for i in range(self.env_settings.n_segments):
            new_transform = self.lock_joints[i].place_other(
                0, param, body_transforms[-1], 0
            )
            body_transforms.append(new_transform)
        gripper2_transform = self.lock_joints[-1].place_other(
            0, param, body_transforms[-1], 0
        )
        hidden2b_transform = self.lock_gripper2_to_hidden_link2b.place_other(
            0, param, gripper2_transform, 0
        )
        hidden1b_transform = self.lock_hidden2b_to_hidden1b.place_other(
            0, param, hidden2b_transform, 0
        )

        body_transforms.append(gripper2_transform)
        body_transforms.append(hidden2b_transform)
        body_transforms.append(hidden1b_transform)

        return Configuration.concatenate(
            [body_transform.to_configuration() for body_transform in body_transforms]
        )

    def get_neutral_state(self, param):
        initial_conf = self.create_neutral_configuration(None, param)
        n_segments = self.env_settings.n_segments
        n_bodies = n_segments + 6
        initial_gvel = GeneralizedVelocity(jnp.zeros([n_bodies, 6]))

        # TODO
        targets = jnp.stack(
            [
                param.constraint_param.frame_a[0].flatten(),
                param.constraint_param.frame_a[-1].flatten(),
            ]
        )
        bl = self.env_settings.segment_halflength

        gripper1_pos = -jnp.array(
            [-bl * self.env_settings.n_segments - self.grapple_box_length, 0, 0]
        )
        gripper2_pos = -jnp.array(
            [bl * self.env_settings.n_segments + self.grapple_box_length, 0, 0]
        )
        targets = jnp.zeros([12]).at[6:9].set(gripper2_pos)
        targets = targets.at[0:3].set(gripper1_pos)

        # When using the PGS-solver with warm starting, multiplier size needs to be correctly specified for jax.jit compilation to work
        multipliers_size = self.get_multiplier_size()
        multipliers = jnp.zeros([multipliers_size])

        return DLOState(initial_conf, initial_gvel, targets, multipliers=multipliers)

    def convert_marker_transforms_to_body_transforms(
        self, marker_transforms: Transform
    ):
        """
        Convert marker transforms (world → marker) into body transforms (world → body)
        using known marker offsets (body → marker).

        Parameters
        ----------
        marker_transforms : Transform
            Batched marker transforms in the world frame. The `pos` and `rot` fields
            must have shapes (n, 3) and (n, 4), respectively, where `n` is the number
            of markers.

        Returns
        -------
        Transform
            Batched body transforms in the world frame (world → body), with the same
            batch size `n` as the input.
        """
        offset_transforms = Transform(
            pos=jnp.stack([po.pos for po in self.env_settings.pose_estimate_offsets]),
            rot=jnp.stack([po.rot for po in self.env_settings.pose_estimate_offsets]),
        )
        marker_to_body = vmap(Transform.inverse)(offset_transforms)
        body_transforms = vmap(Transform.multiply)(marker_transforms, marker_to_body)
        return body_transforms

    def get_state_by_body_interpolation(
        self, param: SimulationParameters, transforms: Transform
    ):
        # Find "relaxed-offset" at interpolation transforms
        state0 = self.get_neutral_state(param)
        n_interp_bodies = self.cable.n_segments + 2
        inertp_offset = self.cable.body_offset - 1

        interp_names = param.rigid_body_param.names[
            inertp_offset : inertp_offset + n_interp_bodies
        ]
        indices_p = [
            interp_names.index(name) for name in self.env_settings.pose_estimate_bodies
        ]
        off0 = np.array(
            state0.conf.pos[inertp_offset : inertp_offset + n_interp_bodies, 0]
        )
        off0_p = off0[jnp.array(indices_p)]

        # Begin with just xyz-interpolation
        from scipy.interpolate import CubicSpline, interp1d

        cs = CubicSpline(off0_p, transforms.pos)
        new_pos = cs(off0)

        # Rotation-interpolation
        qp = transforms.rot
        new_rot = math.quat_interp(off0, off0_p, qp)

        new_state = self._place_hidden_links(new_pos, new_rot, param, state0)

        return new_state

    def get_state_by_interface_interpolation(
        self, param: SimulationParameters, transforms: Transform
    ):
        # Find "relaxed-offset" at interpolation transforms
        state0 = self.get_neutral_state(param)
        # These are the offsets that we transform to and from...

        # First frame is not included in offsets a (DLO + last lock)
        # Note that the full interpolation ensamble includes both DLO and grippers
        n_interp_bodies = self.cable.n_segments + 2
        inertp_offset = self.cable.body_offset - 1
        offset_b = self.cable.constraint_offset - 1
        offset_a = self.cable.constraint_offset
        interp_names = param.rigid_body_param.names[
            inertp_offset : inertp_offset + n_interp_bodies
        ]
        indices_body = [
            interp_names.index(name) for name in self.env_settings.pose_estimate_bodies
        ]

        # Exclude the last frame (not part of interpolation)
        frame_offsets_a = param.constraint_param.frame_a.as_vectorized_transform()[
            offset_a : offset_a + n_interp_bodies
        ][:-1]

        # Last frame is not included in offsets b (DLO + first lock)
        frame_offsets_b = param.constraint_param.frame_b.as_vectorized_transform()[
            offset_b : offset_b + n_interp_bodies
        ][1:]

        pos = state0.conf.pos[inertp_offset : inertp_offset + n_interp_bodies]
        rot = state0.conf.rot[inertp_offset : inertp_offset + n_interp_bodies]

        # T[world->body] @ T[body->frame]
        constraint_transforms_a = vmap(Transform.multiply)(
            Transform(pos, rot)[:-1], frame_offsets_a
        )
        constraint_transforms_b = vmap(Transform.multiply)(
            Transform(pos, rot)[1:], frame_offsets_b
        )
        # One can verify that constraint_transforms_b is (approx) the same as constraint_transforms_a
        reference_offset = constraint_transforms_a.pos[:, 0]

        # The first and final indices corresponds to the grippers, they only have one relevant frame
        last_segment_w_marker_id = indices_body[-2]
        first_segment_w_marker_id = indices_body[1]
        last_interface_w_marker_id = last_segment_w_marker_id + 1
        first_interface_w_marker_id = first_segment_w_marker_id - 1

        # The points at which to reconstruct "frame a"s
        off0a = reference_offset[:last_interface_w_marker_id]
        off0b = reference_offset[first_interface_w_marker_id:]

        # The points at which the transforms are known
        off0a_p = reference_offset[jnp.array(indices_body[:-1])]
        off0b_p = reference_offset[jnp.array(indices_body[1:]) - 1]

        # Begin with just xyz-interpolation
        from scipy.interpolate import CubicSpline, interp1d

        # Frame A
        cs = interp1d(
            off0a_p, transforms[0].pos, kind="linear", axis=0
        )  # axis=0 if fp is (N, ...)
        # cs = CubicSpline(off0a_p, transforms[0].pos[:-1])
        new_pos_a = cs(off0a)
        new_rot_a = math.quat_interp(off0a, off0a_p, transforms[0].rot)
        new_transforms_a = Transform(new_pos_a, new_rot_a)

        # T[world->frame] = T[world->body] @ T[body->frame]
        # which implies
        # T[world->body] = T[world->frame] @T[frame->body]
        body_locations_a = vmap(Transform.get_relative)(
            new_transforms_a, frame_offsets_a[:last_interface_w_marker_id]
        )
        # Frame B
        cs = interp1d(
            off0b_p, transforms[1].pos, kind="linear", axis=0
        )  # axis=0 if fp is (N, ...)
        # cs = CubicSpline(off0b_p, transforms[1].pos[1:])
        new_pos_b = cs(off0b)
        new_rot_b = math.quat_interp(off0b, off0b_p, transforms[1].rot)
        new_transforms_b = Transform(new_pos_b, new_rot_b)

        # T[w->f] = T[w->b] @ T[b->f]
        # T[w->b] = T[w->f] @ inv(T[b->f])
        body_locations_b = vmap(Transform.get_relative)(
            new_transforms_b, frame_offsets_b[first_interface_w_marker_id:]
        )
        a_only_pos = body_locations_a.pos[:first_segment_w_marker_id]
        a_only_rot = body_locations_a.rot[:first_segment_w_marker_id]
        a_shared_pos = body_locations_a.pos[first_segment_w_marker_id:]
        a_shared_rot = body_locations_a.rot[first_segment_w_marker_id:]
        b_only_pos = body_locations_b.pos[
            last_segment_w_marker_id - first_segment_w_marker_id + 1 :
        ]
        b_only_rot = body_locations_b.rot[
            last_segment_w_marker_id - first_segment_w_marker_id + 1 :
        ]
        b_shared_pos = body_locations_b.pos[
            : last_segment_w_marker_id - first_segment_w_marker_id + 1
        ]
        b_shared_rot = body_locations_b.rot[
            : last_segment_w_marker_id - first_segment_w_marker_id + 1
        ]

        shared_pos = (a_shared_pos + b_shared_pos) / 2
        delta = vmap(math.quat_residual)(b_shared_rot, a_shared_rot)
        half_step = vmap(math.from_rotation_vector)(0.5 * delta)
        shared_rot = vmap(math.quat_mul)(half_step, a_shared_rot)
        interp_pos = jnp.concatenate([a_only_pos, shared_pos, b_only_pos])
        interp_rot = jnp.concatenate([a_only_rot, shared_rot, b_only_rot])

        new_state = self._place_hidden_links(interp_pos, interp_rot, param, state0)

        return new_state

    def _place_hidden_links(self, interp_pos, interp_rot, param, state):
        # The configuration of the interpolated segment (DLO are grippers) is now known
        # The final step is to place the robot arms (hidden links)
        gripper1_transform = Transform(interp_pos[0], interp_rot[0])
        gripper2_transform = Transform(interp_pos[-1], interp_rot[-1])

        # TODO: Fix
        lock_pos1 = -gripper1_transform.pos
        lock_euler1 = math.quat_to_euler(math.conjugate(gripper1_transform.rot))
        # lock_euler1 = lock_euler1.at[2].set(-lock_euler1[2])
        lock1_yaw, lock1_pitch, lock1_roll = lock_euler1
        lock_euler1 = lock_euler1.at[0].set(-lock1_yaw)
        lock_euler1 = lock_euler1.at[1].set(lock1_pitch)
        lock_euler1 = lock_euler1.at[2].set(-lock1_roll)

        lock_pos2 = -gripper2_transform.pos
        lock_euler2 = math.quat_to_euler(math.conjugate(gripper2_transform.rot))
        # lock_euler2 = lock_euler2.at[2].set(-lock_euler2[2])
        # lock_euler2 = lock_euler2.at[1].set(-lock_euler2[1])
        lock_euler2 = lock_euler2.at[0].set(-lock_euler2[0])

        hidden2a_transform = self.lock_hidden2a_to_gripper1.place_other(
            5, param, gripper1_transform, -lock_euler1[2]
        )
        hidden1a_transform = self.lock_hidden1a_to_hidden2a.place_other(
            5, param, hidden2a_transform, lock_euler1[1]
        )

        hidden2b_transform = self.lock_gripper2_to_hidden_link2b.place_other(
            5, param, gripper2_transform, -lock_euler2[2]
        )
        hidden1b_transform = self.lock_hidden2b_to_hidden1b.place_other(
            5, param, hidden2b_transform, lock_euler2[1]
        )

        full_pos = jnp.concatenate(
            [
                hidden1a_transform.pos[None],
                hidden2a_transform.pos[None],
                interp_pos,
                hidden2b_transform.pos[None],
                hidden1b_transform.pos[None],
            ]
        )
        full_rot = jnp.concatenate(
            [
                hidden1a_transform.rot[None],
                hidden2a_transform.rot[None],
                interp_rot,
                hidden2b_transform.rot[None],
                hidden1b_transform.rot[None],
            ]
        )

        new_conf = Configuration(
            pos=full_pos,
            rot=full_rot,
        )

        new_state = state.replace(conf=new_conf)

        targets = jnp.concatenate(
            [
                jnp.concatenate([lock_pos1, lock_euler1]),
                jnp.concatenate([lock_pos2, lock_euler2]),
            ]
        )

        return new_state.replace(lock_targets=targets)

    def convert_body_transforms_to_frame_transforms(
        self, param: SimulationParameters, body_transforms: Transform
    ):
        indices_a = [
            param.constraint_param.names.index(name)
            for name in self.env_settings.pose_estimate_constraints_a
        ]
        indices_b = [
            param.constraint_param.names.index(name)
            for name in self.env_settings.pose_estimate_constraints_b
        ]

        frame_offsets_a = param.constraint_param.frame_a.as_vectorized_transform()[
            jnp.array(indices_a)
        ]
        frame_offsets_b = param.constraint_param.frame_b.as_vectorized_transform()[
            jnp.array(indices_b)
        ]

        constraint_transforms_a = vmap(Transform.multiply)(
            body_transforms[:-1], frame_offsets_a
        )
        constraint_transforms_b = vmap(Transform.multiply)(
            body_transforms[1:], frame_offsets_b
        )

        return constraint_transforms_a, constraint_transforms_b

    def relax_shear_displacement(self, state, param):
        # Simulate the system for a few systems to avoid shear displacement
        test_param = param.tree_replace(
            src={
                "sparse_param.cable_param.youngs_modulus": 1e0,
                "sparse_param.cable_param.shear_modulus": 1e0,
                "rigid_body_param.mass": param.rigid_body_param.mass.at[:].set(1e4),
                "gravity": jnp.array([0.0, 0.0, 0.0]),
            }
        )
        horizon = 10
        # Simulation loop
        for i in range(horizon):
            # Step the environment and store the observation
            new_state, _ = jax.jit(self.step)(state, jnp.zeros([12]), test_param)
            new_conf = new_state.conf.replace(pos=state.conf.pos)
            state = new_state.replace(conf=new_conf)

        return state

    def control_help_strings(self):
        return [
            "h/l: left/right",
            "j/k: up/down",
            "u/i: in/out",
            "m/,: yaw (z)",
            "y/n: pitch (y')",
            "6/7: roll (x'')",
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
            motor4 = 1.0
        elif key_map[","]:
            motor4 = -1.0

        elif key_map["y"]:
            motor5 = 1.0
        elif key_map["n"]:
            motor5 = -1.0

        elif key_map["6"]:
            motor6 = 1.0
        elif key_map["7"]:
            motor6 = -1.0
        motor1_to_6 = jnp.array([-motor1, -motor2, -motor3, motor4, motor5, motor6])
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

    def get_state_with_floating_markers(
        self, param: SimulationParameters, transforms: Transform
    ):
        """Only for debug initialization"""
        # Find "relaxed-offset" at interpolation transforms
        state0 = self.get_neutral_state(param)
        indices_p = [
            param.rigid_body_param.names.index(name)
            for name in self.env_settings.pose_estimate_bodies
        ]
        new_pos = state0.conf.pos.at[indices_p, :].set(transforms.pos)
        new_rot = state0.conf.rot.at[indices_p, :].set(transforms.rot)
        new_conf = state0.conf.replace(pos=new_pos, rot=new_rot)
        return state0.replace(conf=new_conf)


# ui (right-left)
# ui (right-left)
# hjkl (left,right,up,down)

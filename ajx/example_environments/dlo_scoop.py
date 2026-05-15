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
    outer_radius: float
    inner_radius: float
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
        outer_radius: float,
        inner_radius: float,
        density: float,
        pose_estimate_linear_offsets: List[float],
        loose_end: bool = False,
    ):
        segment_length = length / n_segments

        gripper1_offset = Transform.unitary()
        gripper2_offset = Transform.unitary()

        pose_estimate_bodies = []
        pose_estimate_constraints_a = []
        pose_estimate_constraints_b = []
        pose_estimate_offsets = []
        pose_estimate_bodies.append("grip_tool1")
        pose_estimate_offsets.append(gripper1_offset)
        pose_estimate_constraints_a.append("lock_gripper1_to_dlo")

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
        pose_estimate_constraints_b.append("lock_dlo_to_gripper2")
        return DLOSettings(
            n_segments,
            0.5 * segment_length,
            outer_radius,
            inner_radius,
            ConstraintType.BEND_TWIST.value,
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

    def get_stiffness(self, inner_radius, outer_radius, segment_length):
        area = jnp.pi * (outer_radius**2 - inner_radius**2)
        area_moment = jnp.pi * (outer_radius**4 - inner_radius**4) / 4
        polar_moment = jnp.pi * (outer_radius**4 - inner_radius**4) / 2

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
    r_outer: jax.Array
    r_inner: jax.Array
    model_shear_deformation: bool = True

    def update_params(self, state: DLOState, u: jax.Array, param: SimulationParameters):
        cable_param: CableParameters = param.sparse_param.cable_param
        n_constraints = self.n_segments + 1
        slice_begin = self.constraint_offset
        slice_end = self.constraint_offset + n_constraints
        constraint_param = param.constraint_param

        area = jnp.pi * (self.r_outer**2 - self.r_inner**2)
        area_moment = jnp.pi * (self.r_outer**4 - self.r_inner**4) / 4
        polar_moment = jnp.pi * (self.r_outer**4 - self.r_inner**4) / 2

        E = cable_param.youngs_modulus
        G = cable_param.shear_modulus

        stretch_stiffness = E * area / self.segment_length
        bend_stiffness = E * area_moment / self.segment_length
        twist_stiffness = G * polar_moment / self.segment_length
        shear_stiffness = 1e9

        if self.model_shear_deformation:
            timoshenko_shear_coefficient = 0.857
            shear_stiffness = (
                G * area / self.segment_length * timoshenko_shear_coefficient
            )

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


class DLOScoop(Environment):
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
        target_pos: jax.Array,
        target_vel: jax.Array,
        bin_pos: jax.Array,
    ):
        self.n_control = 1
        self.timestep = sim_settings.timestep
        self.env_settings = env_settings

        self.reference_timestep = sim_settings.timestep

        self.control_names = ["voltage"]
        self.state_tangent_dim = self.env_settings.n_segments * 12
        self.settings = sim_settings
        self.target_pos = target_pos
        self.target_vel = target_vel
        self.bin_pos = bin_pos

        self._build_sim(sim_settings)
        self.dynamic_residual_names = self.get_state_residual_names()

        self.camera_pos = jnp.array([0.0, 2.0, 0.0])
        self.camera_rot = jnp.array([1.0, 0.0, 0.0, 0.0])

        self.initial_control_state = None

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
        shovel_path = os.path.join(script_dir, "assets/shovel.glb")
        cylinder_path = os.path.join(script_dir, "assets/cylinder.glb")
        arrow_path = os.path.join(script_dir, "assets/arrow.bam")
        bin_path = os.path.join(script_dir, "assets/ring.bam")

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
                self.env_settings.outer_radius,
                self.env_settings.segment_halflength,
                self.env_settings.outer_radius,
            ),
            color=(0.1, 0.1, 0.5),
        )
        segment_wireframe_model = geometry.Model(
            f"segment_wireframe_model",
            hex_wireframe_path,
            rotation=math.Rotations.y_to_x,
            scale=(
                self.env_settings.outer_radius,
                self.env_settings.segment_halflength,
                self.env_settings.outer_radius,
            ),
            color=(0.0, 0.0, 0.0),
        )
        shovel = geometry.Model(
            f"shovel",
            shovel_path,
            rotation=math.Rotations.z_to_y,
            color=(0.0, 0.0, 0.0),
        )
        cylinder = geometry.Model(
            f"cylinder",
            cylinder_path,
            rotation=math.Rotations.z_to_y,
            color=(1.0, 0.0, 0.0),
        )
        x_scale = jnp.linalg.norm(self.target_vel[:3]) * 0.1
        arrow = geometry.Model(
            f"arrow",
            arrow_path,
            rotation=math.Rotations.unitary,
            scale=(x_scale, 0.1, 0.1),
            color=(1.0, 0.0, 0.0),
        )
        bl = self.env_settings.segment_halflength
        bin = geometry.Model(
            f"bin",
            bin_path,
            rotation=math.Rotations.z_to_y,
            scale=(0.1, 0.1, 0.1),
            color=(1.0, 0.0, 0.0),
        )
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
                shovel,
                arrow,
                cylinder,
                bin,
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
            f"reference_box",
            0.03,
            0.03,
            0.06,
        )
        reference_box2 = geometry.Box(
            f"reference_box2",
            0.14 * 0.5,
            0.25 * 0.5,
            0.01 * 0.5,
        )
        tool1_model_local_transform = Transform(
            jnp.array([0.0, 0.0, 0.0]), math.Rotations.x_to_y
        )
        marker1_local_transform = self.env_settings.pose_estimate_offsets[0]
        tool1_to_dlo_frame = Transform(
            jnp.array([grapple_box_length, 0.0, 0.0]), math.Rotations.unitary
        )
        tool2_to_dlo_frame = Transform(
            jnp.array([-0.060425, 0.0, 0.0]), math.Rotations.unitary
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

        grip_tool2 = RigidBody(
            f"grip_tool2",
            [("shovel", Transform.unitary())],
            [("shovel", Transform.unitary())],
        )
        grip_tool2_param = RigidBodyParameters.create(
            mass=0.14 * 0.25 * 0.01 * density * 0.5**3,
            inertia_diag=reference_box2.get_diag_inertia(density),
            name="grip_tool2",
        )

        cylinder = RigidBody(
            f"cylinder",
            [
                ("cylinder", Transform.unitary()),
            ],
            [("cylinder", Transform.unitary())],
        )
        r = 0.03
        h = 0.04
        cyl_mass = density * jnp.pi * r**2 * h
        Jxy = 1 / 12 * cyl_mass * (3 * r**2 + h**2)
        Jz = 0.5 * cyl_mass * r**2
        cylinder_param = RigidBodyParameters.create(
            mass=cyl_mass,
            inertia_diag=jnp.array([Jxy, Jxy, Jz]),
            name="cylinder",
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
            r_o = self.env_settings.outer_radius
            r_i = self.env_settings.inner_radius
            length = 2 * self.env_settings.segment_halflength
            area = jnp.pi * (r_o**2 - r_i**2)
            # Cylinder mass
            mass_cyl = self.env_settings.density * area * length * 2
            mass = mass_cyl
            inertia_cyl_x = 0.5 * mass * (r_o**2 + r_i**2)

            inertia_cyl_yz = (1.0 / 12.0) * mass * (3 * (r_o**2 + r_i**2) + length**2)
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
            frame_b=Frame(jnp.array([0.0, 0.0, 0.0]), math.Rotations.unitary),
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
                frame_a=Frame(jnp.array([bl, 0.0, 0.0]), math.Rotations.z_to_x),
                frame_b=Frame(tool2_to_dlo_frame.pos, tool2_to_dlo_frame.rot),
                compliance=1e-8,
                viscous_compliance=1e-5,
                damping=2 * self.reference_timestep,
                offset=0.0,
                name="lock_dlo_to_gripper2",
            )
        )
        self.lock_gripper2_to_cylinder = TwoBodyConstraint(
            name="lock_gripper2_to_cylinder",
            body_a="grip_tool2",
            body_b="cylinder",
            constraint_type=ConstraintType.PRISMATIC.value,
        )
        lock_gripper2_to_cylinder_param = ConstraintParameters.create_locked(
            frame_a=Frame(jnp.array([0.03, 0.0, -0.004686]), math.Rotations.unitary),
            frame_b=Frame(jnp.array([0.0, 0.0, -0.04]), math.Rotations.unitary),
            compliance=1e-8,
            viscous_compliance=1e-5,
            damping=2 * self.reference_timestep,
            offset=0.0,
            name="lock_gripper2_to_cylinder",
        )
        rb_param = RigidBodyParameters.concatenate(
            [
                hidden_link1a_param,
                hidden_link2a_param,
                grip_tool1_param,
                *arms_param,
                grip_tool2_param,
                cylinder_param,
            ]
        )
        rigid_bodies = tuple(
            [
                hidden_link1a,
                hidden_link2a,
                grip_tool1,
                *arms,
                grip_tool2,
                cylinder,
            ]
        )

        constraint_param = ConstraintParameters.concatenate(
            [
                lock_world_to_hidden1_param,
                lock_hidden1a_to_hidden2a_param,
                lock_hidden2a_to_gripper1_param,
                *lock_joint_param,
                lock_gripper2_to_cylinder_param,
            ]
        )
        constraints = tuple(
            [
                self.lock_world_to_hidden1a,
                self.lock_hidden1a_to_hidden2a,
                self.lock_hidden2a_to_gripper1,
                *self.lock_joints,
                self.lock_gripper2_to_cylinder,
            ]
        )
        if self.env_settings.loose_end:
            constraints = tuple(
                [
                    self.lock_world_to_hidden1a,
                    self.lock_hidden1a_to_hidden2a,
                    self.lock_hidden2a_to_gripper1,
                    *self.lock_joints,
                    self.lock_gripper2_to_cylinder,
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

        self.cable = CoupleAsCable(
            "couple_constraints",
            constraint_offset=constraint_param.names.index("lock_gripper1_to_dlo"),
            body_offset=rb_param.names.index("body0"),
            n_segments=self.env_settings.n_segments,
            segment_length=self.env_settings.segment_halflength * 2,
            r_outer=self.env_settings.outer_radius,
            r_inner=self.env_settings.inner_radius,
        )

        pre_step_modifiers = (
            target_speed_motor1,
            target_speed_motor2,
            target_speed_motor3,
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
        self.geometry_list = tuple([*list(self.geometry_list), reference_box2])

        vel_rotation_vec = self.target_vel[:3] / jnp.linalg.norm(self.target_vel[:3])
        # Rotation interpretation: x-axis lands on vel_rotation_vec
        rot_mat = math.make_rotation_matrix(vel_rotation_vec)

        vel_quat = math.matrix_to_quaternion(rot_mat)

        self.extra_geometry = [
            ("ground", Transform.unitary()),
            ("marker_model", Transform.unitary().replace(pos=self.target_pos)),
            ("arrow", Transform.unitary().replace(pos=self.target_pos, rot=vel_quat)),
            ("bin", Transform.unitary().replace(pos=self.bin_pos)),
        ]

    def create_neutral_configuration(self, observation, param):
        # TODO: Function name is misleading
        bl = self.env_settings.segment_halflength
        world_transform = Transform(
            jnp.array(
                [
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
            math.Rotations.unitary,  # This should not be required...
        )

        body_transforms = []
        body_transforms.append(
            self.lock_world_to_hidden1a.place_other(param, world_transform, 0)
        )
        body_transforms.append(
            self.lock_hidden1a_to_hidden2a.place_other(0, param, body_transforms[-1], 0)
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
        cylinder_transform = self.lock_gripper2_to_cylinder.place_other(
            0, param, gripper2_transform, 0
        )
        body_transforms.append(gripper2_transform)
        body_transforms.append(cylinder_transform)

        return Configuration.concatenate(
            [body_transform.to_configuration() for body_transform in body_transforms]
        )

    def get_neutral_state(self, param):
        initial_conf = self.create_neutral_configuration(None, param)
        n_segments = self.env_settings.n_segments
        n_bodies = n_segments + 5
        initial_gvel = GeneralizedVelocity(jnp.zeros([n_bodies, 6]))

        targets = jnp.stack(
            [
                param.constraint_param.frame_a[0].flatten(),
                param.constraint_param.frame_a[-1].flatten(),
            ]
        )
        bl = self.env_settings.segment_halflength

        gripper1_pos = -jnp.array([0, 0, 0])
        targets = jnp.zeros([12])
        targets = targets.at[0:3].set(gripper1_pos)

        multipliers_size = self.get_multiplier_size()
        multipliers = jnp.zeros([multipliers_size])

        return DLOState(initial_conf, initial_gvel, targets, multipliers)

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

        control_state = None
        return jnp.concatenate([motor1_to_6]), control_state

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

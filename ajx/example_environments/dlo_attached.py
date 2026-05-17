import jax.numpy as jnp
import ajx.math as math
import os
from ajx import *
from ajx.example_environments.environment import Environment

from typing import Optional
import ajx.example_graphics.geometry as geometry
import numpy as np


@dataclass
class DLOAttachedSettings:
    n_bodies: int
    body_length: float
    mass_density: float
    constraint_residual: ConstraintResidual
    hinge_motor_attachment: bool
    body_side_length: Optional[float] = 0.1


@struct.dataclass
class DLOState(ParameterNode):
    conf: Configuration
    gvel: GeneralizedVelocity
    lock_targets: jax.Array
    multipliers: jax.Array = struct.field(default_factory=lambda: jnp.zeros([0]))


@struct.dataclass
class PositiveParam(ParameterNode):
    data: jax.Array

    def retract(self, delta):
        def clip_st(x, lo, hi):
            clipped = jnp.clip(x, lo, hi)
            return x + jax.lax.stop_gradient(clipped - x)

        updated = self.data + delta
        return PositiveParam(clip_st(updated, 1e-12, 1e12))

    # def retract(self, delta):
    #     new = self.data / (1 + delta * self.data)
    #     return PositiveParam(jnp.clip(new, 1e-12, 1e12))


@struct.dataclass
class CoupledConstraintParameters(ParameterNode):
    linear_stiffness: PositiveParam
    quadratic_stiffness: PositiveParam
    damping: jax.Array
    is_velocity: jax.Array


DLOSparseParam = create_parameter_node("DLOSparseParam", ("coupled_constraint_param",))


@dataclass
class NonlinearUpdate(PreStepModifier):
    name: str
    target: str
    lbda: Any

    def update_params(self, state: DLOState, u: jax.Array, param: SimulationParameters):
        new_param = param.tree_replace({self.target: self.lbda(state)})
        return state, new_param


@dataclass
class CoupleConstraints(PreStepModifier):
    name: str
    target_slice: Tuple
    body_ids: jax.Array
    constraint_ids: jax.Array
    constraint_residual: ConstraintResidual

    def update_params(self, state: DLOState, u: jax.Array, param: SimulationParameters):
        ccp: CoupledConstraintParameters = param.sparse_param.coupled_constraint_param
        slice_begin = self.target_slice[0]
        slice_end = self.target_slice[1]
        constraint_param = param.constraint_param
        offsets = vmap(TwoBodyConstraint.func, in_axes=(None, None, 0, 0, None))(
            param,
            state,
            self.body_ids,
            self.constraint_ids,
            self.constraint_residual,
        )
        compliance = jnp.clip(
            1
            / (
                ccp.linear_stiffness.data
                + jnp.abs(offsets) * ccp.quadratic_stiffness.data
            ),
            1e-6,
            1e8,
        )
        constraint_param = constraint_param.replace(
            compliance=constraint_param.compliance.at[slice_begin:slice_end].set(
                compliance
            )
        )
        constraint_param = constraint_param.replace(
            damping=constraint_param.damping.at[slice_begin:slice_end].set(ccp.damping)
        )
        constraint_param = constraint_param.replace(
            is_velocity=constraint_param.is_velocity.at[slice_begin:slice_end].set(
                ccp.is_velocity
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

    def update_params(self, state: DLOState, u, param: SimulationParameters):
        lock = u[self.u_idx] == 0.0
        not_lock = jnp.logical_not(lock)
        current_offset = self.constraint.object_func(state, param)[self.target_dof]
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


class DLOAttached(Environment):
    def __init__(
        self,
        sim_settings: SimulationSettings,
        env_settings: DLOAttachedSettings,
    ):
        self.n_control = 1
        self.timestep = sim_settings.timestep
        self.env_settings = env_settings

        self.reference_timestep = sim_settings.timestep

        self.control_names = ["voltage"]
        self.state_tangent_dim = self.env_settings.n_bodies * 12
        self.settings = sim_settings
        self._build_sim(sim_settings)
        self.dynamic_residual_names = self.get_state_residual_names()

        self.camera_pos = jnp.array(
            [self.env_settings.body_length * self.env_settings.n_bodies, 15.0, 0.0]
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
        n = self.env_settings.n_bodies
        gradient = gradient_start - jnp.outer(
            jnp.arange(n), (gradient_start - gradient_end) / n
        )
        density = self.env_settings.mass_density

        for i in range(self.env_settings.n_bodies):
            box = geometry.Box(
                f"box{i}",
                0.5 * self.env_settings.body_length,
                0.5 * self.env_settings.body_side_length,
                0.5 * self.env_settings.body_side_length,
                translation=(0.0, 0.0, 0.0),
                color=tuple([*gradient[i]]),
            )
            boxes.append(box)
            mass = (
                density
                * self.env_settings.body_side_length
                * self.env_settings.body_side_length
                * self.env_settings.body_length
            )
            inertia = box.get_diag_inertia(density)

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

        if self.env_settings.hinge_motor_attachment:
            self.attachment_constraint = OneBodyConstraint(
                name=f"attachment_hinge",
                body="body0",
                constraint_residual=ConstraintResidual.AXIAL_WORLD_SPHERICAL.value,
            )
            attachment_constraint_param = ConstraintParameters.create(
                free_degree=5,
                frame_a=Frame(jnp.array([0.0, 0.0, 0.0]), math.Rotations.x_to_y),
                frame_b=Frame(jnp.array([0.0, 0.0, 0.0]), math.Rotations.x_to_y),
                compliance=1e-8,
                damping=2 * self.reference_timestep,
                b=1e5,
                name="attachment_hinge",
            )
        else:
            self.attachment_constraint = OneBodyConstraint(
                name=f"attachment_lock",
                body="body0",
                constraint_residual=self.env_settings.constraint_residual,
            )
            attachment_constraint_param = ConstraintParameters.create_locked(
                frame_a=Frame(jnp.array([0.0, 0.0, 0.0]), rotation1),
                frame_b=Frame(jnp.array([0.0, 0.0, 0.0]), rotation2),
                compliance=1e-8,
                viscous_compliance=1e-5,
                damping=2 * self.reference_timestep,
                offset=0.0,
                name="attachment_lock",
            )

        # To make DLO lock joint constraints between all bodies
        bl = self.env_settings.body_length
        for i in range(0, self.env_settings.n_bodies - 1):
            self.lock_joints.append(
                TwoBodyConstraint(
                    name=f"lock{i}",
                    body_a=f"body{i}",
                    body_b=f"body{i+1}",
                    constraint_residual=self.env_settings.constraint_residual,
                )
            )
            lock_joint_param.append(
                ConstraintParameters.create_locked(
                    frame_a=Frame(jnp.array([0.5 * bl, 0.0, 0.0]), rotation1),
                    frame_b=Frame(jnp.array([-0.5 * bl, 0.0, 0.0]), rotation2),
                    compliance=1e-8,
                    viscous_compliance=1e-5,
                    damping=2 * self.reference_timestep,
                    offset=0.0,
                    name=f"lock{i}",
                )
            )

        rb_param = RigidBodyParameters.concatenate([*arms_param])
        rigid_bodies = tuple([*arms])

        constraint_param = ConstraintParameters.concatenate(
            [attachment_constraint_param, *lock_joint_param]
        )
        constraints = tuple([self.attachment_constraint, *self.lock_joints])

        # n_constraints = one per body + one
        n_segment_locks = self.env_settings.n_bodies - 1
        couple_constraints = CoupleConstraints(
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
            constraint_residual=self.env_settings.constraint_residual,
        )

        if self.env_settings.hinge_motor_attachment:
            hinge_motor = LockAtZeroSpeedMotor(
                "hinge_motor", self.attachment_constraint, 0, 0, 5
            )
            pre_step_modifiers = (couple_constraints, hinge_motor)
        else:
            pre_step_modifiers = (couple_constraints,)

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
            jnp.array([bl * self.env_settings.n_bodies, 0.0, 1.0]),
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

        coupled_constraint_param = CoupledConstraintParameters(
            linear_stiffness=PositiveParam(jnp.ones(6) * 1e5),
            # compliance=jnp.ones(6) * 1e-5,
            quadratic_stiffness=PositiveParam(jnp.ones(6) * 0.0),
            damping=jnp.ones(6) * 2 * self.sim.settings.timestep,
            is_velocity=jnp.zeros(6, dtype=bool),
        )

        self.default_param = SimulationParameters(
            jnp.array([0.0, 0.0, -9.82]),
            rb_param,
            constraint_param,
            DLOSparseParam(coupled_constraint_param),
        )

        self.geometry_list = tuple([*boxes])

        self.extra_geometry = [
            geometry.Square(
                "ground",
                400.0,
                400.0,
                translation=(bl * self.env_settings.n_bodies, 0.0, -100.0),
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
        body_transforms.append(
            self.attachment_constraint.place_other(param, world_transform, 0)
        )
        for i in range(len(self.lock_joints)):
            new_transform = self.lock_joints[i].place_other(
                0, param, body_transforms[-1], 0
            )
            body_transforms.append(new_transform)
        return Configuration.concatenate(
            [body_transform.to_configuration() for body_transform in body_transforms]
        )

    def state_from_angles(self, param):

        initial_conf = self.observation_to_configuration(None, param)
        n_bodies = self.env_settings.n_bodies
        initial_gvel = GeneralizedVelocity(jnp.zeros([n_bodies, 6]))
        targets = jnp.zeros([1])

        # When using the PGS-solver with warm starting, multiplier size needs to be correctly specified for jax.jit compilation to work
        multipliers_size = self.get_multiplier_size()
        multipliers = jnp.zeros([multipliers_size])

        return DLOState(initial_conf, initial_gvel, targets, multipliers=multipliers)

    def get_stiffness_from_material_parameters(self, youngs_modulus, shear_modulus):

        l = self.env_settings.body_length
        s = self.env_settings.body_side_length

        area = s**2  # Cross-sectional area
        area_moment = s**4 / 12  # Second moment of area
        polar_moment = (
            s**4 / 6
        )  # Second polar moment of area, assumes square cross section

        # Stiffness values per segment
        axial_stiffness = youngs_modulus * area / l
        bending_stiffness = youngs_modulus * area_moment / l
        torsional_stiffness = shear_modulus * polar_moment / l

        return axial_stiffness, bending_stiffness, torsional_stiffness

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
            motor1 = 3.0  # -0.5
        elif key_map["l"] or key_map["arrow_right"]:
            motor1 = -3.0  # 0.5

        if (key_map["j"] and key_map["k"]) or (
            key_map["arrow_down"] and key_map["arrow_up"]
        ):
            motor3 = 0.0
        elif key_map["j"] or key_map["arrow_down"]:
            motor3 = -3.0
        elif key_map["k"] or key_map["arrow_up"]:
            motor3 = 3.0

        if key_map["u"] and key_map["i"]:
            motor2 = 0.0
        elif key_map["u"]:
            motor2 = -3.0
        elif key_map["i"]:
            motor2 = 3.0

        elif key_map["m"]:
            motor4 = -3.0
        elif key_map[","]:
            motor4 = 3.0

        elif key_map["y"]:
            motor5 = -3.0
        elif key_map["n"]:
            motor5 = 3.0

        elif key_map["6"]:
            motor6 = -3.0
        elif key_map["7"]:
            motor6 = 3.0
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

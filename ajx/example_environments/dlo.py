import jax.numpy as jnp
import ajx.math as math
import os
from ajx import *
from ajx.example_environments.environment import Environment

from typing import Optional
import ajx.example_graphics.geometry as geometry


@dataclass
class DLOSettings:
    n_bodies: int
    body_length: float
    constraint_type: ConstraintType
    loose_end: bool


@struct.dataclass
class DLOState(ParameterNode):
    conf: Configuration
    gvel: GeneralizedVelocity
    lock_targets: jax.Array


@struct.dataclass
class CoupledConstraintParameters(ParameterNode):
    compliance: jax.Array
    damping: jax.Array
    is_velocity: jax.Array


DLOSparseParam = create_parameter_node("DLOSparseParam", ("coupled_constraint_param",))


@struct.dataclass
class CoupleConstraints(PreStepModifier):
    name: str
    target_slice: Tuple[int, int]

    def update_params(self, state: DLOState, u: jax.Array, param: SimulationParameters):
        ccp: CoupledConstraintParameters = param.sparse_param.coupled_constraint_param
        slice_begin = self.target_slice[0]
        slice_end = self.target_slice[1]
        constraint_param = param.constraint_param
        constraint_param = constraint_param.replace(
            compliance=constraint_param.compliance.at[slice_begin:slice_end].set(
                ccp.compliance
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
    target_dof: int = 5

    def update_params(self, state: DLOState, u, param: SimulationParameters):
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
        return state, (
            param.tree_replace(
                {
                    f"constraint_param.is_velocity.{self.constraint.name}": {
                        self.target_dof: not_lock
                    },
                }
            ).tree_replace(
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
        self.state_tangent_dim = self.env_settings.n_bodies * 12
        self.settings = sim_settings
        self._build_sim(sim_settings)
        self.dynamic_residual_names = self.get_state_residual_names()

        self.camera_pos = jnp.array(
            [self.env_settings.body_length * self.env_settings.n_bodies, 15.0, 0.0]
        )
        self.camera_rot = math.quat_from_axis_angle(jnp.array([0.0, 0.0, 1.0]), jnp.pi)

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
        for i in range(self.env_settings.n_bodies):
            box = geometry.Box(
                f"box{i}",
                self.env_settings.body_length,
                0.1,
                0.1,
                translation=(0.0, 0.0, 0.0),
                color=tuple([*gradient[i]]),
            )
            boxes.append(box)
            density = 10.0
            mass = density * 0.1 * 0.1 * self.env_settings.body_length
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
        # if self.env_settings.constraint_type == ConstraintType.PRISMATIC.value:
        #     # v-axis (uvw) should point along x-axis (rotate 90 deg about z-axis)
        #     rotation1 = math.quat_from_axis_angle(
        #         jnp.array([0.0, 0.0, 1.0]), -0.5 * jnp.pi
        #     )
        #     rotation2 = math.quat_from_axis_angle(
        #         jnp.array([0.0, 0.0, 1.0]), -0.5 * jnp.pi
        #     )

        self.first_lock = OneBodyConstraint(
            name=f"lock_first",
            body="body0",
            constraint_type=self.env_settings.constraint_type,
        )
        bl = self.env_settings.body_length
        first_lock_param = ConstraintParameters.create_locked(
            frame_a=Frame(jnp.array([0.0, 0.0, 0.0]), rotation1),
            frame_b=Frame(jnp.array([bl, 0.0, 0.0]), rotation2),
            compliance=1e-8,
            damping=2 * self.reference_timestep,
            offset=0.0,
            name="lock_first",
        )
        for i in range(self.env_settings.n_bodies - 1):
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
                    damping=0.5 * self.reference_timestep,
                    offset=0.0,
                    name=f"lock{i}",
                )
            )
        self.last_lock = OneBodyConstraint(
            name=f"lock_last",
            body=f"body{self.env_settings.n_bodies-1}",
            constraint_type=self.env_settings.constraint_type,
        )
        last_lock_param = ConstraintParameters.create_locked(
            frame_a=Frame(
                jnp.array([bl * 2 * self.env_settings.n_bodies, 0.0, 0.0]), rotation1
            ),
            frame_b=Frame(jnp.array([0.0, 0.0, 0.0]), rotation2),
            compliance=1e-8,
            damping=2 * self.reference_timestep,
            offset=0.0,
            name="lock_last",
        )

        rb_param = RigidBodyParameters.concatenate([*arms_param])
        rigid_bodies = tuple(arms)

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

        couple_constraints = CoupleConstraints(
            "couple_constraints", (1, self.env_settings.n_bodies)
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
        point_set = [(i, offset) for i in range(n) for offset in offsets]
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
            compliance=jnp.ones(6) * 1e-5,
            damping=jnp.ones(6) * 2 * self.sim.settings.timestep,
            is_velocity=jnp.ones(6, dtype=bool),
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
                1.0,
                1.0,
                (0.0, 0.0, -2.0),
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
        for i in range(self.env_settings.n_bodies - 1):
            new_transform = self.lock_joints[i].place_other(
                param, body_transforms[-1], 0
            )
            body_transforms.append(new_transform)
        return Configuration.concatenate(
            [body_transform.to_configuration() for body_transform in body_transforms]
        )

    def state_from_angles(self, param):

        initial_conf = self.observation_to_configuration(None, param)
        n_bodies = self.env_settings.n_bodies
        initial_gvel = GeneralizedVelocity(jnp.zeros([n_bodies, 6]))
        targets = jnp.zeros([12])
        return DLOState(initial_conf, initial_gvel, targets)

    def control_help_strings(self):
        return [
            "h/l: motor1",
            "j/k: motor2",
            "u/i: motor3",
            "m/,: motor4",
            "y/n: motor5",
            "6/7: motor6",
        ]

    def control_func(self, observation, last_ozbservation, key_map):
        motor1 = 0.0
        motor2 = 0.0
        motor3 = 0.0
        motor4 = 0.0
        motor5 = 0.0
        motor6 = 0.0
        if key_map["l"] and key_map["h"]:
            motor1 = 0.0
        elif key_map["h"]:
            motor1 = -3.0  # -0.5
        elif key_map["l"]:
            motor1 = 3.0  # 0.5

        if key_map["j"] and key_map["k"]:
            motor2 = 0.0
        elif key_map["j"]:
            motor2 = -3.0
        elif key_map["k"]:
            motor2 = 3.0

        if key_map["u"] and key_map["i"]:
            motor3 = 0.0
        elif key_map["u"]:
            motor3 = -1.0
        elif key_map["i"]:
            motor3 = 1.0

        elif key_map["m"]:
            motor4 = -1.0
        elif key_map[","]:
            motor4 = 1.0

        elif key_map["y"]:
            motor5 = -3.0
        elif key_map["n"]:
            motor5 = 3.0

        elif key_map["6"]:
            motor6 = -3.0
        elif key_map["7"]:
            motor6 = 3.0
        motor7 = 0.0
        return jnp.array([motor1, motor2, motor3, motor4, motor5, motor6, motor7])


# ui (right-left)
# ui (right-left)
# hjkl (left,right,up,down)

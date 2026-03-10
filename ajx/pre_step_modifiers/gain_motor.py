from flax import struct
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Literal
from ajx.pre_step_modifiers.base import PreStepModifier
from ajx.tree_util import ParameterNode


@struct.dataclass
class GainMotorParameters(ParameterNode):
    # Gamma nu =  G vk + gamma
    # [speed_constant] = [1/emf_constant] = (rad/s)/V
    # target_speed = gamma = -U(t)/emf_constant
    # regularization = Gamma = R/(torque_constant * emf_constant)
    # = R/toruq_constant * speed_constant

    # For ..
    # speed_constant = 116 rpm/V = 14.6 (rad/s) / V
    # torque_constant = 82.2 mNm/A =0.082 Nm/A
    # Resistance = 2.44 Ohm
    # regularization = 434 = 2.44 / 0.082 * 14.6
    # b = 230e-5 Nm / (rad/s)
    # DCMotorParameters(14.6, 230e-5)
    inertia: float
    gain: float


class GainMotor2(PreStepModifier):
    def __init__(
        self, name: str, constraint, timestep: float, idx: int, target_dof: int
    ):
        self.name = name
        self.constraint = constraint
        self.timestep = timestep
        self.idx = idx
        self.target_dof = target_dof

    def update_params(self, state, u, param):
        body_ids = tuple(
            param.rigid_body_param.names.index(body) for body in self.constraint.bodies
        )
        constraint_ids = (param.constraint_param.names.index(self.constraint.name),)
        jac = self.constraint.__class__.jacobian(
            param, state, body_ids, constraint_ids, self.constraint.constraint_type
        ).reshape(
            len(self.constraint.bodies), self.constraint.get_constrained_degrees(), -1
        )
        # jac = self.constraint.jacobian(param, state) # TODO: Why is this not working?
        omega = sum(
            [
                val[self.target_dof, None] @ state.gvel.data[body]
                for body, val in zip(body_ids, jac)
            ]
        )

        sparse_dict_param = param.sparse_param.__dict__

        speed = (
            omega[0]
            - self.timestep**2
            / sparse_dict_param[self.name].inertia
            * sparse_dict_param[self.name].gain
            * u[self.idx]
        )
        return state, param.tree_replace(
            {
                "constraint_param": {
                    "target": {self.constraint.name: {self.target_dof: speed}},
                    "compliance": {
                        self.constraint.name: {
                            self.target_dof: self.timestep
                            / sparse_dict_param[self.name].inertia
                        }
                    },
                }
            }
        )

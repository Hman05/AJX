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


class GainMotor(PreStepModifier):
    def __init__(self, name: str, target_constraint, target_constraint_name, timestep):
        self.name = name
        self.target_constraint = target_constraint
        self.target_constraint_name = target_constraint_name
        self.timestep = timestep

    def update_params(self, state, u, param):
        jac = self.target_constraint.tangential_projection(param, state)
        omega = sum([val @ state.gvel.data[body] for body, val in jac.items()])

        speed = (
            omega[0]
            - self.timestep**2
            / param.sparse_param[self.name].inertia
            * param.sparse_param[self.name].gain
            * u[0]
        )
        return {
            "motor_param": {
                self.target_constraint_name: {
                    "speed": speed,
                    "b": param.sparse_param[self.name].inertia / self.timestep,
                }
            }
        }


class GainMotor2(PreStepModifier):
    def __init__(self, name: str, constraint, timestep: float, idx: int):
        self.name = name
        self.constraint = constraint
        self.timestep = timestep
        self.idx = idx
        self.target_dof = 5

    def update_params(self, state, u, param):
        body_ids = tuple(
            param.rigid_body_param.names.index(body) for body in self.constraint.bodies
        )
        constraint_id = param.constraint_param.names.index(self.constraint.name)
        jac = self.constraint.__class__.jacobian(
            param, state, body_ids, constraint_id, self.constraint.constraint_type
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
        return {
            "constraint_param": {
                "target": {self.constraint.name: {5: speed}},
                "compliance": {
                    self.constraint.name: {
                        5: self.timestep / sparse_dict_param[self.name].inertia
                    }
                },
            }
        }

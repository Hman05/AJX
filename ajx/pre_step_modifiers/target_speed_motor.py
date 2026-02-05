from flax import struct
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Literal
from ajx.pre_step_modifiers.base import PreStepModifier
from ajx.constraints import Constraint
from flax import struct
import jax.numpy as jnp
import jax


@struct.dataclass
class TargetSpeedMotor(PreStepModifier):
    name: str
    target_constraint: str
    idx: int

    # def __init__(self, name: str, target_constraint: str, idx: int):
    #     self.name = name
    #     self.target_constraint = target_constraint
    #     self.idx = idx

    def update_params(self, state, u, param):
        return {"motor_param": {self.target_constraint: {"speed": u[self.idx]}}}


@struct.dataclass
class TargetSpeedMotor2(PreStepModifier):
    name: str
    constraint: Constraint
    idx: int

    # def __init__(self, name: str, target_constraint: str, idx: int):
    #     self.name = name
    #     self.target_constraint = target_constraint
    #     self.idx = idx

    def update_params(self, state, u, param):
        lock = u[self.idx] == 0.0
        not_lock = jnp.logical_not(lock)
        default_target = param.get_cp(self.constraint.name).target[5]
        target = default_target * lock + u[self.idx] * not_lock
        return {
            "constraint_param": {
                self.constraint.name: {
                    "target5": target,
                }
            }
        }

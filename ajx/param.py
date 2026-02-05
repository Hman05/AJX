from __future__ import annotations

import copy
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from loguru import logger

from ajx import math
from ajx.definitions import ConstraintParameters
from ajx.definitions import RigidBodyParameters
from util.deepinsert import deepinsert


@register_pytree_node_class
class SimulationParameters:
    # Fixed
    inertial_update_metric: str

    # Dynamic
    gravity: jax.Array
    rigid_body_param: RigidBodyParameters
    constraint_param: ConstraintParameters
    sparse_param: Dict

    def __init__(
        self,
        gravity,
        rigid_body_param,
        constraint_param,
        sparse_param,
    ):
        self.gravity = gravity
        self.rigid_body_param = rigid_body_param
        self.constraint_param = constraint_param
        self.sparse_param = sparse_param
        self.inertial_update_metric = "euclidian"

    def get_cp(self, key):
        i = self.constraint_param.names.index(key)
        return self.constraint_param[i]

    def get_rb(self, key):
        i = self.rigid_body_param.names.index(key)
        return self.rigid_body_param[i]

    def as_dict(self):
        return {
            "gravity": self.gravity,
            "rigid_body_param": self.rigid_body_param,
            "constraint_param": self.constraint_param,
            "sparse_param": self.sparse_param,
        }

    def copy(self):
        return SimulationParameters(
            self.gravity,
            self.rigid_body_param.copy(),
            self.constraint_param.copy(),
            copy.deepcopy(self.sparse_param),
        )

    def insert(self, src):
        new = self.copy()
        new.inertial_update_metric = self.inertial_update_metric
        if isinstance(src, SimulationParameters):
            return src
        if src is None:
            return new
        for key in src.keys():
            if key in "rigid_body_param":
                new.rigid_body_param = new.rigid_body_param.insert(src[key])
            elif key in "constraint_param":
                new.constraint_param = new.constraint_param.insert(src[key])
            elif key in "gravity":
                new.gravity = deepinsert(new.gravity, src[key])
            elif key in "sparse_param":
                new.sparse_param = deepinsert(new.sparse_param, src[key])
            else:
                msg = (
                    f"The provided source ({key}) does not index destination correctly"
                )
                raise Exception(msg)
        return new

    def increment(self, src):
        new = self.copy()
        new.inertial_update_metric = self.inertial_update_metric
        if isinstance(src, SimulationParameters):
            return src
        if src is None:
            return new
        for key in src.keys():
            if key in "rigid_body_param":
                new.rigid_body_param = new.rigid_body_param.increment(
                    src[key], self.inertial_update_metric
                )
            elif key in "constraint_param":
                new.constraint_param = new.constraint_param.increment(src[key])
            elif key in "motor_param":
                new.motor_param = new.motor_param.increment(src[key])
            elif key in "gravity":
                raise NotImplementedError
            elif key in "sparse_param":
                raise NotImplementedError
            else:
                msg = (
                    f"The provided source ({key}) does not index destination correctly"
                )
                raise Exception(msg)
        return new

    def readjust_for_zero_mc(self, constraints):
        # Readjust constraints parameters(frames) and rigid body parameters (inertia) for zero mc
        new = self.copy()
        new.inertial_update_metric = self.inertial_update_metric

        # 1. Adjust Inertia
        def readjust_per_body(rb_param):
            mc = rb_param.mc
            m = rb_param.mass
            I_p = rb_param.inertia
            I_mc = I_p + math.skew(mc) @ math.skew(mc) / m
            triu = jnp.triu_indices(3)
            I_mc_vecsym = I_mc[triu]
            new_data = rb_param.data.at[4:10].set(I_mc_vecsym).at[1:4].set(0.0)
            return new_data

        # new.rigid_body_param.data = vmap(readjust_per_body)(new.rigid_body_param)
        # TODO: Replace with callback to allow JIT compilation
        if isinstance(new.rigid_body_param.data, jax.core.Tracer):
            logger.warning("Warning for invalid inertial parameters is disabled")
        elif jnp.any(jnp.linalg.eig(new.rigid_body_param.inertia)[0].real < 0):
            # An attempt prevent simulations with invalid invertial parameters
            logger.warning("Encountered invalid inertial parameters")
            # print(jnp.linalg.eig(new.rigid_body_param.inertia)[0].real)
            # raise Exception("Invalid inertial parameters")

        # 2. Adjust Frames
        for constraint in constraints:
            idx = self.constraint_param.names.index(constraint.name)
            constraint_param = self.constraint_param[idx]
            if constraint.body_a:
                body_idx = self.rigid_body_param.names.index(constraint.body_a)
                rb_param = self.rigid_body_param[body_idx]
                frame_loc = constraint_param.frame_a.position
                c = rb_param.mc / rb_param.mass
                new_frame_loc = frame_loc - c
                new.constraint_param.data = new.constraint_param.data.at[idx, 0:3].set(
                    new_frame_loc
                )
            if constraint.body_b:
                body_idx = self.rigid_body_param.names.index(constraint.body_b)
                rb_param = self.rigid_body_param[body_idx]
                frame_loc = constraint_param.frame_b.position
                c = rb_param.mc / rb_param.mass
                new_frame_loc = frame_loc - c
                new.constraint_param.data = new.constraint_param.data.at[idx, 7:10].set(
                    new_frame_loc
                )
        return new

    def tree_flatten(self):
        children = (
            self.gravity,
            self.rigid_body_param,
            self.constraint_param,
            self.sparse_param,
        )
        aux_data = self.inertial_update_metric
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        res = cls(*children)
        inertial_update_metric = aux_data
        res.inertial_update_metric = inertial_update_metric
        return res

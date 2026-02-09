from __future__ import annotations

import copy
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from loguru import logger

from ajx import math
from ajx.definitions import ConstraintParameters, RigidBodyParameters, ajx_dataclass
from util.deepinsert import deepinsert


@ajx_dataclass
class SimulationParameters:
    # Dynamic
    gravity: jax.Array
    rigid_body_param: RigidBodyParameters
    constraint_param: ConstraintParameters
    sparse_param: Dict

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
        if isinstance(src, SimulationParameters):
            return src
        if src is None:
            return new
        for key in src.keys():
            if key in self.rigid_body_param.names:
                new.rigid_body_param = new.rigid_body_param.insert(src)
            elif key in self.constraint_param.names:
                new.constraint_param = new.constraint_param.insert(src)
            elif key in "rigid_body_param":
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

    def tree_flatten(self):
        children = (
            self.gravity,
            self.rigid_body_param,
            self.constraint_param,
            self.sparse_param,
        )
        aux_data = ()
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        res = cls(*children)
        return res

from __future__ import annotations

import jax

from ajx.definitions import ConstraintParameters, RigidBodyParameters


from ajx.tree_util import ParameterNode
from flax import struct
from typing import Tuple


@struct.dataclass
class SimulationParameters(ParameterNode):
    # Dynamic
    gravity: jax.Array
    rigid_body_param: RigidBodyParameters
    constraint_param: ConstraintParameters
    sparse_param: ParameterNode


def create_parameter_node(name: str, keys: Tuple[str]):
    namespace = {"__annotations__": {k: object for k in keys}}
    cls = type(name, (ParameterNode,), namespace)
    return struct.dataclass(cls)

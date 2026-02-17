from flax import struct
import jax
import ajx.math as math
from ajx.pre_step_modifiers.base import PreStepModifier
from ajx.tree_util import ParameterNode


@struct.dataclass
class TiltGravityParam(ParameterNode):
    default_gravity: jax.Array
    gravity_rotation: jax.Array


class TiltGravity(PreStepModifier):
    def __init__(self, name: str):
        self.name = name

    def update_params(self, state, u, param):
        defualt_gravity = param[self.name].default_gravity
        quat_rotation = param[self.name].gravity_rotation
        rotated_gravity = math.rotate_vector(quat_rotation, defualt_gravity)
        return {"g": rotated_gravity}

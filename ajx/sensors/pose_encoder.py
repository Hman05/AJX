import jax.numpy as jnp
import ajx.math as math
from ajx.sensors.base import Sensor
from ajx.definitions import Transform


class PoseEncoder(Sensor):
    def __init__(
        self,
        name: str,
        body: str,
        marker_offset: Transform,
    ):
        self.name = name
        self.body = body
        self.observable_names = ["x", "y", "z", "qs", "qx", "qy", "qz"]
        self.residual_names = ["x", "y", "z", "rx", "ry", "rz"]
        self.marker_offset = marker_offset

    def observe(self, state, qdot_next, param):
        idx = param.rigid_body_param.names.index(self.body)
        body_transform = Transform(state.conf.pos[idx], state.conf.rot[idx])
        marker_transform = body_transform.multiply(self.marker_offset)
        return marker_transform.flatten()

    def residual(self, target, prediction):
        pos_delta = target[:3] - prediction[:3]
        rot_delta = math.quat_residual(target[3:], prediction[3:])
        return jnp.concatenate([rot_delta])

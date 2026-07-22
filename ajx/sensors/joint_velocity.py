import jax.numpy as jnp
from ajx.sensors.base import Sensor


class AngularVelocitySensor(Sensor):
    def __init__(self, name: str, hinge):
        self.name = name
        self.hinge = hinge

        self.observable_names = ["omega"]
        self.residual_names = ["omega"]

    def observe(self, state, qdot_next, param):
        omega = self.hinge.get_free_degree_velocity(state, qdot_next, param)
        idx = param.sparse_param.offset_param.names.index(self.name)
        offset = param.sparse_param.offset_param.offset[idx]
        scale = param.sparse_param.offset_param.scale[idx]
        return jnp.stack(
            [
                scale * omega + offset,
            ]
        )

    def residual(self, target, prediction):
        delta = prediction - target
        return jnp.concatenate([delta])


class LinearVelocitySensor(Sensor):
    def __init__(self, name: str, prismatic):
        self.name = name
        self.prismatic = prismatic

        self.observable_names = ["v"]
        self.residual_names = ["v"]

    def observe(self, state, qdot_next, param):
        velocity = self.prismatic.get_free_degree_velocity(state, qdot_next, param)
        idx = param.sparse_param.offset_param.names.index(self.name)
        offset = param.sparse_param.offset_param.offset[idx]
        scale = param.sparse_param.offset_param.scale[idx]
        return jnp.stack(
            [
                scale * velocity + offset,
            ]
        )

    def residual(self, target, prediction):
        delta = prediction - target
        return jnp.concatenate([delta])

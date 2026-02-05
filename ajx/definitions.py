from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Callable, List, Tuple, Any
from dataclasses import dataclass, field
from jax.tree_util import register_pytree_node_class
from flax import struct


def ajx_dataclass(cls):
    """
    A class decorator for creating dataclasses that can be passed to jax transformations,
    similar to flax.struct.dataclass.

    TODO: This implementation is a bit hacky. Compare with flax.struct.dataclass to make it more robust.
    """
    cls = dataclass(cls)
    array_types = ["jnp.array", "jax.Array"]
    string_types = ["Tuple[str]", "str"]
    ajx_types = ["Frame", "Configuration", "GeneralizedVelocity", "State"]
    array_attr_names = [
        attr
        for attr, type_str in cls.__annotations__.items()
        if type_str in array_types
    ]
    ajx_attr_names = [
        attr for attr, type_str in cls.__annotations__.items() if type_str in ajx_types
    ]
    dynamic_attr_names = [*array_attr_names, *ajx_attr_names]
    str_attr_names = [
        attr
        for attr, type_str in cls.__annotations__.items()
        if type_str in string_types
    ]
    static_attr_names = str_attr_names

    def stack(objects: List):
        assert len(objects) > 0
        stacked_attrs = {}
        for key in array_attr_names:
            stacked_attr = jnp.stack([obj.__dict__[key] for obj in objects])
            stacked_attrs[key] = stacked_attr
        for key in ajx_attr_names:
            stacked_attr = objects[0].__class__
            stacked_attrs[key] = stacked_attr
        for key in str_attr_names:
            stacked_attr = tuple(obj.__dict__[key] for obj in objects)
            stacked_attrs[key] = stacked_attr

        return cls(**stacked_attrs)

    def copy(self):
        return cls(**self.__dict__)

    def tree_flatten(self):
        children = tuple(self.__dict__[key] for key in dynamic_attr_names)
        aux_data = tuple(self.__dict__[key] for key in static_attr_names)
        return (children, aux_data)

    def __getitem__(self, key):
        traced_vals = {k: self.__dict__[k][key] for k in dynamic_attr_names}
        if isinstance(key, jax.core.Tracer):
            untraced_vals = {k: None for k in str_attr_names}
        else:
            untraced_vals = {k: self.__dict__[k][key] for k in str_attr_names}
        return cls(**untraced_vals, **traced_vals)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        aux_dict = {key: val for key, val in zip(static_attr_names, aux_data)}
        children_dict = {key: val for key, val in zip(dynamic_attr_names, children)}
        return cls(**aux_dict, **children_dict)

    def __hash__(self):
        # TODO: Is this a bad idea?
        vals = tuple(self.__dict__[key] for key in str_attr_names)
        return hash(vals)

    cls.stack = stack
    cls.copy = copy
    cls.tree_flatten = tree_flatten
    cls.tree_unflatten = tree_unflatten
    cls.__getitem__ = __getitem__
    cls.__hash__ = __hash__

    register_pytree_node_class(cls)

    return cls


@ajx_dataclass
class Frame:
    position: jnp.array
    rotation: jnp.array


@ajx_dataclass
class RigidBody:
    name: Tuple[str]
    geometry: Tuple[str]


@ajx_dataclass
class Configuration:
    pos: jax.Array
    rot: jax.Array


@ajx_dataclass
class GeneralizedVelocity:
    data: jax.Array

    @property
    def vel(self):
        return self.data[..., :3]

    @property
    def ang(self):
        return self.data[..., 3:]


@ajx_dataclass
class State:
    conf: Configuration
    gvel: GeneralizedVelocity


@ajx_dataclass
class ConstraintParameters:
    # Fixed
    names: Tuple[str]

    # Dynamic
    frame_a: Frame
    frame_b: Frame
    compliance: jax.Array  # Viscous compliance for velocity constrained dofs
    damping: jax.Array  # Ignored for velocity constrained dofs
    target: jax.Array  # Target velocity for velocity constrained dofs
    is_velocity: jax.Array  # Bools set to true for velocity constrained dofs

    @classmethod
    def create(
        cls,
        frame_a: Frame,
        frame_b: Frame,
        compliance: float,
        damping: float,
        b: float,
        name: str,
    ):
        holonomic_compliance = jnp.array([compliance] * 5)
        holonomic_damping = jnp.array([damping] * 5)
        viscous_compliance = jnp.array([1.0 / b])
        ignored_damping = jnp.array([0.0])
        target = jnp.zeros(6)
        compliance = jnp.concatenate([holonomic_compliance, viscous_compliance])
        damping = jnp.concatenate([holonomic_damping, ignored_damping])
        is_velocity = jnp.array([False, False, False, False, False, True], dtype=bool)
        return cls(
            name,
            frame_a=frame_a,
            frame_b=frame_b,
            compliance=compliance,
            damping=damping,
            target=target,
            is_velocity=is_velocity,
        )

    @classmethod
    def create_locked(
        cls,
        frame_a: Frame,
        frame_b: Frame,
        compliance: float,
        damping: float,
        offset: float,
        name: str,
    ):
        frame_a_data = frame_a.concat()
        frame_b_data = frame_b.concat()
        holonomic_compliance = jnp.array([compliance] * 6)
        damping = jnp.array([damping] * 6)
        target = jnp.zeros(5)
        offset = jnp.array([offset])
        new_data = jnp.concatenate(
            [frame_a_data, frame_b_data, holonomic_compliance, damping, target, offset]
        )
        return cls(name, new_data)

    def stack_with_constraints(
        param_constraint_pairs: Tuple[Tuple[ConstraintParameters, Any]],
    ):
        if not param_constraint_pairs:
            data = jnp.zeros([0])
            return ConstraintParameters(tuple(), data), tuple()
        frame_a_stacked = Frame.stack(
            [pair[0].frame_a for pair in param_constraint_pairs]
        )
        frame_b_stacked = Frame.stack(
            [pair[0].frame_b for pair in param_constraint_pairs]
        )
        compliance_stacked = jnp.stack(
            [pair[0].compliance for pair in param_constraint_pairs]
        )
        damping_stacked = jnp.stack(
            [pair[0].damping for pair in param_constraint_pairs]
        )
        target_stacked = jnp.stack([pair[0].target for pair in param_constraint_pairs])
        is_velocity_stacked = jnp.stack(
            [pair[0].is_velocity for pair in param_constraint_pairs]
        )

        c_names = tuple([pair[0].names for pair in param_constraint_pairs])
        constraints = tuple(pair[1] for pair in param_constraint_pairs)
        c_names2 = tuple(constraint.name for constraint in constraints)
        assert c_names == c_names2
        return (
            ConstraintParameters(
                c_names,
                frame_a_stacked,
                frame_b_stacked,
                compliance_stacked,
                damping_stacked,
                target_stacked,
                is_velocity_stacked,
            ),
            constraints,
        )

    def insert(self, src):
        new = self.copy()
        for constraint_name, src2 in src.items():
            if not constraint_name in self.names:
                msg = f"The provided source ({constraint_name}) does not index destination correctly"
                raise Exception(msg)
            idx = self.names.index(constraint_name)
            if src2 is None:
                continue
            for prop, val in src2.items():
                if prop == "compliance05":
                    new.data = new.data.at[idx, 14:20].set(val)
                elif prop == "compliance04":
                    new.compliance = new.compliance.at[idx, :4].set(val)
                elif prop == "compliance5":
                    new.compliance = new.compliance.at[idx, 5].set(val)
                elif prop == "damping05":
                    new.data = new.data.at[idx, 20:26].set(val)
                elif prop == "damping04":
                    new.data = new.data.at[idx, 20:25].set(val)
                elif prop == "damping5":
                    new.data = new.data.at[idx, 25].set(val)
                elif prop == "frame_b_pos_x":
                    new.data = new.data.at[idx, 7].set(val)
                elif prop == "frame_b_pos_y":
                    new.data = new.data.at[idx, 8].set(val)
                elif prop == "target5":
                    new.target = new.target.at[idx, 5].set(val)
                else:
                    msg = f"The provided source ({prop}) does not index destination correctly"
                    raise Exception(msg)
        return new


@ajx_dataclass
class RigidBodyParameters:
    # Fixed
    names: Tuple[str]

    # Dynamic
    data: jnp.array

    def __init__(
        self,
        names,
        data,
    ):
        self.names = names
        self.data = data

    @property
    def mass(self):
        return self.data[..., 0]

    @property
    def mc(self):
        return self.data[..., 1:4]

    @property
    def inertia(self):
        inertia = jnp.zeros([3, 3])
        if len(self.data.shape) == 2:
            n_bodies = self.data.shape[0]
            inertia = jnp.zeros([n_bodies, 3, 3])
        triu = jnp.triu_indices(3)
        tril = (triu[1], triu[0])
        if len(self.data.shape) == 2:
            inertia = vmap(lambda x, y: x.at[triu].set(y))(
                inertia, self.data[..., 4:10]
            )
            inertia = vmap(lambda x, y: x.at[triu].set(y))(
                inertia, self.data[..., 4:10]
            )
        else:
            inertia = inertia.at[triu].set(self.data[4:10])
            inertia = inertia.at[tril].set(self.data[4:10])

        return inertia

    @classmethod
    def create(cls, mass: float, inertia_diag: jax.Array, name: str):
        mass = jnp.array(mass).reshape(1)
        mc = jnp.array([0.0, 0.0, 0.0])
        diag_indices = jnp.array([0, 3, 5])
        inertia = jnp.zeros(6).at[diag_indices].set(inertia_diag)

        new_data = jnp.concatenate([mass, mc, inertia])
        return cls(name, new_data)

    def stack_with_rigid_bodies(
        param_rb_pairs: Tuple[Tuple[RigidBodyParameters, RigidBody]],
    ):
        data_stacked = jnp.stack([param_rb[0].data for param_rb in param_rb_pairs])
        rb_names = tuple([param_rb[0].names for param_rb in param_rb_pairs])
        rigid_bodies = tuple(param_rb[1] for param_rb in param_rb_pairs)
        rb_names2 = tuple(rb.name for rb in rigid_bodies)
        assert rb_names == rb_names2
        return RigidBodyParameters(rb_names, data_stacked), rigid_bodies

    def insert(self, src):
        new = self.copy()
        if src is None:
            return new
        if isinstance(src, jax.Array):
            # TODO: Lazy fix for empty dict
            return new
        for rb_name, src2 in src.items():
            if not rb_name in self.names:
                msg = f"The provided source ({rb_name}) does not index destination correctly"
                raise Exception(msg)
            idx = self.names.index(rb_name)
            for prop, val in src2.items():
                if prop == "mass":
                    new.data = new.data.at[idx, 0].set(val)
                elif prop == "mc_x":
                    new.data = new.data.at[idx, 1].set(val)
                elif prop == "mc_y":
                    new.data = new.data.at[idx, 2].set(val)
                elif prop == "mc_z":
                    new.data = new.data.at[idx, 3].set(val)
                elif prop == "inertia_xx":
                    new.data = new.data.at[idx, 4].set(val)
                elif prop == "inertia_xy":
                    new.data = new.data.at[idx, 5].set(val)
                elif prop == "inertia_xz":
                    new.data = new.data.at[idx, 6].set(val)
                elif prop == "inertia_yy":
                    new.data = new.data.at[idx, 7].set(val)
                elif prop == "inertia_yz":
                    new.data = new.data.at[idx, 8].set(val)
                elif prop == "inertia_zz":
                    new.data = new.data.at[idx, 9].set(val)
                else:
                    msg = f"The provided source ({prop}) does not index destination correctly"
                    raise Exception(msg)

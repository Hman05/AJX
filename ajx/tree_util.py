from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct
from dataclasses import asdict, fields
from typing import Dict, Tuple, Union, Sequence
import numbers

global_flag = False


def arr_tree_replace(arr, src, name_idx_maps=()):
    for key, value in src.items():
        if isinstance(key, tuple):
            assert len(key) > 0
            assert all(isinstance(v, str) for v in key)
            next_key = key[1:] if len(key) > 2 else key[1]
            value = {next_key: value}
            key = key[0]

        if isinstance(key, int):
            idx = key
        elif isinstance(key, str):
            has_names = len(name_idx_maps) > 0
            has_idx_in_names = False
            if has_names:
                has_idx_in_names = key in name_idx_maps[0]
            if has_idx_in_names:
                idx = name_idx_maps[0].index(key)
            elif key.isdigit():
                idx = int(key)
            else:
                raise Exception
        else:
            raise Exception
        if isinstance(value, (jax.Array, int, float)):
            arr = arr.at[idx].set(value)
        elif isinstance(value, dict):
            new = arr_tree_replace(arr[idx], value, name_idx_maps[1:])
            arr = arr.at[idx].set(new)
        else:
            raise Exception
    return arr


def arr_tree_retract(arr, src):
    for key, value in src.items():
        if isinstance(value, (jax.Array, int, float)):
            arr = arr.at[key].set(value)
        elif isinstance(value, dict):
            for inner_key, inner_val in value.items():
                arr = arr.at[key, inner_key].set(arr[key, inner_key] + inner_val)
        else:
            raise Exception
    return arr


def tangent_jacfwd(func, argnum=0, has_aux=False):
    def transformed_func(*x):
        zero_tangent = jnp.zeros([x[argnum].tangent_size()])

        def func_inc(inc, x):
            return func(*x[:argnum], x[argnum].retract(inc), *x[argnum + 1 :])

        return jax.jacfwd(func_inc, has_aux=has_aux)(zero_tangent, x)

    return transformed_func


class ParameterNode:
    """
    Base class for a node in a hierarchical parameter structure (SimulationParameters and State).

    Instances may contain array-valued parameters or nested ``ParameterNode``
    objects, forming a JAX-compatible PyTree.
    """

    def tree_replace(self, src: Dict) -> ParameterNode:
        """
        Recursively replace leaves or individual stacked components.

        Parameters
        ----------
        src : Dict
            Mapping that specifies replacement values in the parameter tree.
            Keys may reference attributes in three equivalent ways:

            1. Nested dictionaries (hierarchical structure)
            2. Tuples of strings representing a path
            3. Dot-separated strings representing a path

            For stacked array attributes, individual components may be indexed by either
            integers or symbolic component names. These symbolic names are resolved
            using the node's ``names`` attribute, which must define the ordering of the
            stacked components. If the node does not define a ``names`` attribute,
            symbolic indexing is not supported.

            Examples
            --------
            Nested form
            ~~~~~~~~~~~
            {
                "rigid_body_parameters": {
                    "constraints": {
                        "compliance": {
                            "body": 1e-5
                        }
                    }
                }
            }

            Tuple path form
            ~~~~~~~~~~~~~~~
            {
                ("rigid_body_parameters", "constraints", "compliance", "body"): 1e-5
            }

            Dot-separated form
            ~~~~~~~~~~~~~~~~~~
            {
                "rigid_body_parameters.constraints.compliance.body": 1e-5
            }

        Returns
        -------
        ParameterNode
            A new instance with the specified components replaced.
            The original object remains unchanged (functional update).
        """
        parameter_node_attibutes = [
            key
            for key, value in self.__dict__.items()
            if isinstance(value, ParameterNode)
        ]
        array_attributes = [
            key
            for key, value in self.__dict__.items()
            if isinstance(value, (jax.Array, float))
        ]

        try:
            src = dict(src)
        except (TypeError, ValueError) as e:
            raise TypeError(
                "src must be a mapping or iterable of (key, value) pairs "
                f"convertible to dict, got {type(src).__name__}"
            ) from e

        new = self.copy()
        for src_key, src_val in src.items():
            if isinstance(src_key, str):
                if "." in src_key:
                    src_key = tuple(src_key.split("."))
            if isinstance(src_key, tuple):
                assert len(src_key) > 0
                assert all(isinstance(v, str) for v in src_key)
                # Indexing with {(x, *y): v} is the same as {x: {*y: v}}
                # We should therefore treat this case as having key x and value {*y: v}
                # If *y constains only one string, we should use the string as the new key
                next_key = src_key[1:] if len(src_key) > 2 else src_key[1]
                val = {next_key: src_val}
                key = src_key[0]
            else:
                key = src_key
                val = src_val
            if key not in self.__dict__.keys():
                raise IndexError(
                    "The provided source does not index existing attributes"
                )
            if key in parameter_node_attibutes:
                if isinstance(val, ParameterNode):
                    new.__dict__[key] = val
                    continue
                try:
                    val = dict(val)
                except (TypeError, ValueError):
                    pass

                if isinstance(val, dict):
                    new.__dict__[key] = new.__dict__[key].tree_replace(val)
                elif val is None:
                    continue
                else:
                    raise ValueError("Unsupported type")
            if key in array_attributes:
                if isinstance(val, (jax.Array, float)):
                    new.__dict__[key] = val
                elif isinstance(val, dict):
                    # Arrays are be indexed by names
                    if not hasattr(self, "names"):
                        raise Exception(
                            "Indexing stacked arrays requires a 'names' attriute"
                        )
                    name_idx_maps = [self.names]
                    field_data = type(self).__dataclass_fields__[key]
                    if "second_axis_names" in field_data.metadata:
                        second_axis_names = field_data.metadata["second_axis_names"]
                        name_idx_maps.append(second_axis_names)

                    new.__dict__[key] = arr_tree_replace(
                        self.__dict__[key], val, name_idx_maps
                    )

                else:
                    raise ValueError("Unsupported type")
        return new

    def _mapped_retract(self, delta: jax.Array, keys: Tuple[str]):

        slice_begin = 0
        new = self.copy()
        for key in keys:
            value = self.get_value_at_path(key)

            if isinstance(value, float):
                value = jnp.array(value)
            if isinstance(value, jax.Array):
                slice_size = value.size
                slice_end = slice_begin + slice_size
                reshaped = delta[slice_begin:slice_end].reshape(value.shape)
                new_value = value + reshaped
                new = new.tree_replace({key: new_value})
                slice_begin = slice_end

            elif isinstance(value, ParameterNode):
                slice_size = value.tangent_size()
                slice_end = slice_begin + slice_size
                new_value = value.retract(delta[slice_begin:slice_end])
                new = new.tree_replace({key: new_value})
                slice_begin = slice_end
            else:
                raise Exception

        return new

    def log_map(self, other: ParameterNode) -> jax.Array:
        residuals = []
        for f in fields(self):
            self_value = getattr(self, f.name)
            other_value = getattr(other, f.name)
            if isinstance(self_value, (jax.Array, float)):
                residual = (self_value - other_value).flatten()
            elif isinstance(self_value, ParameterNode):
                residual = self_value.log_map(other_value)
            elif f.metadata.get("pytree_node") is False:
                continue
            else:
                raise Exception
            residuals.append(residual)

        return jnp.concatenate(residuals)

    def tree_retract(self, delta: Dict) -> ParameterNode:
        """
        Recursively applies structured updates to leaves using retraction rules.

        If a leaf supports manifold-aware updates, the provided increment is
        interpreted as an element of the tangent space and mapped back to the
        manifold. Otherwise, a default additive update is applied.

        Parameters
        ----------
        delta: Dict
            Nested dictionary specifying update increments. The structure mirrors
            the hierarchical parameter tree. For stacked component groups,
            individual components may be addressed by their symbolic name.
        Returns:
        -------
        ParameterNode:
            A new instance with updates applied. The original object remains
            unchanged.
        """
        parameter_node_attibutes = [
            key
            for key, value in self.__dict__.items()
            if isinstance(value, ParameterNode)
        ]
        array_attributes = [
            key for key, value in self.__dict__.items() if isinstance(value, jax.Array)
        ]

        new = self.copy()
        for key, val in delta.items():
            if key not in self.__dict__.keys():
                raise IndexError(
                    "The provided source does not index existing attributes"
                )
            if key in parameter_node_attibutes:
                if isinstance(val, dict):
                    new.__dict__[key] = self.__dict__[key].tree_retract(val)
                elif isinstance(val, (jax.Array, float)):
                    # Call retract functions for manifold-aware updates
                    new.__dict__[key] = self.__dict__[key].retract(val)
                else:
                    raise ValueError("Unsupported type")
            if key in array_attributes:
                if isinstance(val, (jax.Array, float)):
                    # Default additive update
                    new.__dict__[key] = self.__dict__[key] + val
                elif isinstance(val, dict):
                    # Stacked arrays can be indexed by name
                    if not hasattr(self, "names"):
                        raise Exception(
                            "Indexing stacked arrays requires a 'names' attriute"
                        )
                    name_idx_map = {k: i for i, k in enumerate(self.names)}
                    arr_src = {name_idx_map.get(k, k): v for k, v in val.items()}
                    new.__dict__[key] = arr_tree_retract(self.__dict__[key], arr_src)

                else:
                    raise ValueError("Unsupported type")
        return new

    def get_metadata(self) -> Tuple:

        kwargs = []
        for f in fields(self):
            value = getattr(self, f.name)
            if f.metadata.get("pytree_node") is False:
                kwargs.append((f.name, value))

        return tuple(kwargs)

    def copy(self) -> ParameterNode:
        """
        Returns a recursive copy of this ParameterNode.

        Array-valued attributes are reused (arrays are immutable in JAX),
        while nested ``ParameterNode`` instances are copied recursively.
        Static metadata fields such as tuples or strings are preserved
        unchanged.
        """

        kwargs = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, (jax.Array, float)):
                kwargs[f.name] = value
            elif isinstance(value, ParameterNode):
                kwargs[f.name] = value.copy()
            elif isinstance(value, (tuple, str)):
                kwargs[f.name] = value
            elif f.metadata.get("pytree_node") is False:
                kwargs[f.name] = value
            else:
                raise Exception

        return type(self)(**kwargs)

    def flatten(self):
        """
        Flattens this ParameterNode.

        The flattening order (i.e. the order of elements in the output list) is deterministic,
        corresponding to a fields-ordered depth-first tree traversal.
        """
        arr = []
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, ParameterNode):
                arr.append(value.flatten())
            elif isinstance(value, jax.Array):
                arr.append(value.flatten())
            elif isinstance(value, float):
                arr.append(jnp.array(value))
        return jnp.concatenate(arr)

    def tangent_size(self):
        """
        Returns the dimension of the tangent space associated with this node.

        This value determines the expected size of flattened update vectors
        passed to ``retract``.
        """
        if hasattr(self, "tangent_restrictions"):
            return self._mapped_tangent_size(self.tangent_restrictions)
        size = 0
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, ParameterNode):
                size += value.tangent_size()
            elif isinstance(value, jax.Array):
                size += value.size
            elif isinstance(value, float):
                size += 1
        return size

    def _mapped_tangent_size(self, keys: Tuple[str]):
        size = 0
        for key in keys:
            value = self.get_value_at_path(key)

            if isinstance(value, float):
                size += 1
            elif isinstance(value, jax.Array):
                size += value.size
            elif isinstance(value, ParameterNode):
                size += value.tangent_size()
            else:
                raise Exception
        return size

    def retract(self, delta: jax.Array):
        """
        Applies a flattened tangent update to this node and returns
        the retracted parameter object.

        The input ``delta`` is interpreted as a vector in the tangent
        space of this node. Its size must match ``tangent_size()``.
        The update is applied recursively in a fixed coordinate order:

        - Array-valued attributes are updated additively.
        - Nested ``ParameterNode`` attributes receive the corresponding
        slice of the tangent vector and apply their own retraction.

        Parameters
        ----------
        delta : jax.Array
            Flattened tangent vector whose size equals ``tangent_size()``.

        Returns
        -------
        ParameterNode:
            A new instance with the update applied. The original object
            remains unchanged.
        """
        assert delta.size == self.tangent_size()

        if hasattr(self, "tangent_restrictions"):
            return self._mapped_retract(delta, self.tangent_restrictions)

        kwargs = {}
        slice_begin = 0

        for f in fields(self):
            value = getattr(self, f.name)

            if isinstance(value, float):
                value = jnp.array(value)

            if isinstance(value, jax.Array):
                slice_size = value.size
                slice_end = slice_begin + slice_size
                reshaped = delta[slice_begin:slice_end].reshape(value.shape)
                kwargs[f.name] = value + reshaped
                slice_begin = slice_end

            elif isinstance(value, ParameterNode):
                slice_size = value.tangent_size()
                slice_end = slice_begin + slice_size
                kwargs[f.name] = value.retract(delta[slice_begin:slice_end])
                slice_begin = slice_end

            else:
                kwargs[f.name] = value

        return type(self)(**kwargs)

    def __getitem__(self, key):
        kwargs = {}

        for f in fields(self):
            value = getattr(self, f.name)

            if isinstance(value, jax.Array):
                kwargs[f.name] = value[key]
            elif isinstance(value, ParameterNode):
                kwargs[f.name] = value[key]
            elif f.metadata.get("pytree_node") is False:
                kwargs[f.name] = value
            else:
                raise Exception

        return type(self)(**kwargs)

    @classmethod
    def concatenate(cls, nodes: Tuple[ParameterNode]):
        """
        Concatenates multiple homogeneous ParameterNode instances into a
        stacked ParameterNode.

        All nodes must share the same field structure. Array-valued fields
        are concatenated along their leading axis, producing a grouped
        representation. Tuple-valued fields are concatenated elementwise.
        Other field types are currently unsupported.

        This operation is typically used to construct component groups
        (e.g., multiple rigid bodies or constraints) from individual
        component parameter objects.

        Parameters
        ----------
        nodes : Tuple[ParameterNode, ...]
            Sequence of ParameterNode instances with identical structure.

        Returns
        -------
        ParameterNode
            A new instance whose array fields represent the stacked
            parameters of all input nodes.
        """
        assert len(nodes) > 0
        first = nodes[0]
        first_fields = fields(first)
        assert all(fields(node) == first_fields for node in nodes[1:])
        assert all(type(node) is cls for node in nodes)
        shared_fields = first_fields
        kwargs = {}
        for f in shared_fields:
            values = [getattr(node, f.name) for node in nodes]
            if isinstance(values[0], tuple):
                concatenated = tuple(item for value in values for item in value)
            elif isinstance(values[0], jax.Array):
                concatenated = jnp.concatenate(values)
            elif isinstance(values[0], ParameterNode):
                concatenated = type(values[0]).concatenate(values)
            else:
                raise Exception("Unsupported type")
            kwargs[f.name] = concatenated
        return cls(**kwargs)

    def create_in_axes(self, mapped_axes: Dict):
        attrs = {}
        for f in fields(self):
            attrs[f.name] = None
            if f.metadata.get("pytree_node") is False:
                attrs[f.name] = self.__dict__[f.name]

        for key, value in mapped_axes.items():
            attrs[key] = value
        return type(self)(**attrs)

    def get_value_at_path(
        self, key: Union[Sequence[str], str]
    ) -> Union[ParameterNode, jax.Array, numbers.Real]:
        """
        Return the value located at a given path in the parameter tree.

        Parameters
        ----------
        key : str or Sequence[str]
            Path to the desired value, provided either as a dot-separated
            string (e.g. ``"a.b.c"``) or a sequence of strings
            (e.g. ``("a", "b", "c")``).

        Returns
        -------
        ParameterNode or jax.Array or numbers.Real
            The value stored at the specified path. Intermediate paths
            return a ``ParameterNode``; terminal paths return a leaf value
            (e.g. ``jax.Array`` or numeric scalar).
        """

        src_key = key
        if isinstance(src_key, str):
            path = tuple(key.split(".")) if key else ()
        elif isinstance(key, (tuple, list)):
            path = tuple(key)
        else:
            raise TypeError(f"Unsupported key type: {type(key).__name__}")

        if len(path) == 0:
            raise ValueError("Empty key/path is not supported")

        head, *tail = path
        if head not in self.__dict__:
            raise KeyError(f"Unknown attribute '{head}'")

        val = self.__dict__[head]
        if tail:
            if isinstance(val, ParameterNode):
                return val.get_value_at_path(tuple(tail))
            elif hasattr(self, "names") and len(tail) == 1:
                idx = self.names.index(tail[0])
                return val[idx]
            raise TypeError(f"Value at '{head}' is not indexable")

        return val

    def __str__(self):
        pass


def flatten_dict_paths(parameter_tree):
    """
    Flatten a nested parameter tree into a single-level dictionary with
    dot-separated path keys.

    Parameters
    ----------
    parameter_tree : dict
        A nested dictionary representing a parameter tree. Internal nodes
        must be dictionaries. Leaves may be ``jax.Array`` instances,
        numeric scalars (``numbers.Real``), or ``ParameterNode`` objects.
        ``ParameterNode`` instances are treated as terminal values and are
        not traversed recursively.

    Returns
    -------
    Dict[str, object]
        A dictionary mapping dot-separated parameter paths (e.g.
        ``"rigid_body_parameters.constraints.compliance.body"``)
        to their corresponding leaf values.

    Raises
    ------
    TypeError
        If an unsupported node or leaf type is encountered during
        traversal.

    Examples
    --------
    >>> parameter_tree = {"a": {"b": 1.0, "c": 2.0}}
    >>> flatten_dict_paths(parameter_tree)
    {'a.b': 1.0, 'a.c': 2.0}
    """

    def _iter_leaf_paths(p, label=None):
        if isinstance(p, dict):
            for k, v in p.items():
                new_label = k if label is None else f"{label}.{k}"
                yield from _iter_leaf_paths(v, new_label)
            return
        elif isinstance(p, (jax.Array, numbers.Real, ParameterNode)):
            yield (label, p)
        else:
            raise TypeError(f"Unsupported type at '{label}': {type(p).__name__}")
        return

    return dict(_iter_leaf_paths(parameter_tree))


def tangent_jacfwd(func, argnum=0, has_aux=False):
    def transformed_func(*x):
        zero_tangent = jnp.zeros([x[argnum].tangent_size()])

        def func_inc(inc, x):
            return func(*x[:argnum], x[argnum].retract(inc), *x[argnum + 1 :])

        return jax.jacfwd(func_inc, has_aux=has_aux)(zero_tangent, x)

    return transformed_func

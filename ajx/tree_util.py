from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct
from dataclasses import asdict, fields
from typing import Dict, Tuple


def arr_tree_replace(arr, src):
    for key, value in src.items():
        if isinstance(value, (jax.Array, int, float)):
            arr = arr.at[key].set(value)
        elif isinstance(value, dict):
            for inner_key, inner_val in value.items():
                arr = arr.at[key, inner_key].set(inner_val)
        else:
            raise Exception
    return arr


class ParameterNode:
    """
    Base class for a node in a hierarchical parameter structure (SimulationParameters and State).

    Instances may contain array-valued parameters or nested ``ParameterNode``
    objects, forming a JAX-compatible PyTree.
    """

    def tree_replace(self, src: Dict) -> ParameterNode:
        """
        Recursively replace leaves or individual stacked components.

        # TODO: Add support for tuple indexation and dot-separated indexation

        Parameters
        ----------
        src : Dict
            Nested dictionary specifying the replacement values. The structure
            mirrors the hierarchical parameter tree. For stacked component
            groups, individual components may be addressed by their symbolic
            name.

            Example
            -------
            {
                "rigid_body_parameters": {
                    "constraints": {
                        "compliance": {
                            "body": 1e-5
                        }
                    }
                }
            }

        Returns
        -------
        ParameterNode
            A new instance with the specified components replaced. The original
            object remains unchanged.
        """
        parameter_node_attibutes = [
            key
            for key, value in self.__dict__.items()
            if isinstance(value, ParameterNode)
        ]
        array_attributes = [
            key for key, value in self.__dict__.items() if isinstance(value, jax.Array)
        ]

        new = self
        for key, val in src.items():
            if key not in self.__dict__.keys():
                raise IndexError(
                    "The provided source does not index existing attributes"
                )
            if key in parameter_node_attibutes:
                if isinstance(val, dict):
                    new.__dict__[key] = self.__dict__[key].tree_replace(val)
                else:
                    raise ValueError("Unsupported type")
            if key in array_attributes:
                if isinstance(val, jax.Array):
                    new.__dict__[key] = val
                elif isinstance(val, dict):
                    # Stacked arrays can be indexed by name
                    if not hasattr(self, "names"):
                        raise Exception(
                            "Indexing stacked arrays requires a 'names' attriute"
                        )
                    name_idx_map = {k: i for i, k in enumerate(self.names)}
                    arr_src = {name_idx_map.get(k, k): v for k, v in val.items()}
                    new.__dict__[key] = arr_tree_replace(self.__dict__[key], arr_src)

                else:
                    raise ValueError("Unsupported type")
        return new

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
            the hierarchical parameter tree. Indexing stacked components is currently
            not supported.
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
                elif isinstance(val, jax.Array):
                    # Call retract functions for manifold-aware updates
                    new.__dict__[key] = self.__dict__[key].retract(val)
                else:
                    raise ValueError("Unsupported type")
            if key in array_attributes:
                if isinstance(val, jax.Array):
                    # Default additive update
                    new.__dict__[key] = self.__dict__[key] + val
                else:
                    raise ValueError("Unsupported type")
        return new

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
                arr.extend(value.flatten())
            elif isinstance(value, jax.Array):
                arr.append(value.flatten())
        return jnp.concatenate(arr)

    def tangent_size(self):
        """
        Returns the dimension of the tangent space associated with this node.

        This value determines the expected size of flattened update vectors
        passed to ``retract``.
        """
        size = 0
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, ParameterNode):
                size += value.tangent_size()
            elif isinstance(value, jax.Array):
                size += value.size
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

        kwargs = {}
        slice_begin = 0

        for f in fields(self):
            value = getattr(self, f.name)

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
            if f.name == "names":
                # TODO: Replace with static fields...
                attrs[f.name] = self.__dict__[f.name]

        for key, value in mapped_axes.items():
            attrs[key] = value
        return type(self)(**attrs)

import jax
import jax.numpy as jnp
from ajx.block_sparse.base import BlockMatrixBase
from jax.tree_util import register_pytree_node_class

from typing import List, Dict, Tuple
from itertools import combinations_with_replacement, product, accumulate


@register_pytree_node_class
class SVBDMatrix(BlockMatrixBase):
    """A symmetric grouped variable block diagonal matrix."""

    def __init__(
        self,
        data: jax.Array,
        block_sizes: List[Tuple[int, int]],
    ):
        self.data = data
        self.block_sizes = block_sizes

    def to_scalar_matrix(self):
        ptr = 0
        dense_groups = []
        for group in self.block_sizes:
            block_dim, n_blocks = group
            size = block_dim * block_dim * n_blocks
            data = self.data[ptr : ptr + size].reshape(n_blocks, block_dim, block_dim)
            dense_group = jax.scipy.linalg.block_diag(*data)
            dense_groups.append(dense_group)
            ptr = ptr + size
        return jax.scipy.linalg.block_diag(*dense_groups)

    def mul_vector(self, vec: jax.Array):
        """
        Multiply this matrix with a vector, assuming it has the correct dimension.
        TODO: Enforce correct vector size (to avoid confusing bugs)
        """
        mat_ptr = 0
        vec_ptr = 0
        mat_mul_vec_groups = []
        for group in self.block_sizes:
            block_dim, n_blocks = group
            mat_size = block_dim * block_dim * n_blocks
            vec_size = block_dim * n_blocks
            self_data = self.data[mat_ptr : mat_ptr + mat_size].reshape(
                n_blocks, block_dim, block_dim
            )
            vec_data = vec[vec_ptr : vec_ptr + vec_size].reshape(n_blocks, block_dim)
            mat_mul_vec = jax.vmap(jnp.matmul)(self_data, vec_data)
            mat_mul_vec_groups.append(mat_mul_vec)

            mat_ptr = mat_ptr + mat_size
            vec_ptr = vec_ptr + vec_size
        return jnp.concatenate(mat_mul_vec_groups, axis=None)

    def tree_flatten(self):
        children = (self.data,)
        aux_data = (self.block_sizes,)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)

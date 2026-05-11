import jax
import jax.numpy as jnp
from ajx.block_sparse.base import BlockMatrixBase


from typing import List, Dict, Tuple
from itertools import combinations_with_replacement, product, accumulate
from jax.tree_util import register_pytree_node_class
from flax import struct
import numpy as np
import numpy.typing as npt


@struct.dataclass
class RowGroup:
    offset: int
    row_size: int
    n_blocks: int
    byte_col_sizes: bytes = struct.field(pytree_node=False)
    byte_col_indices: bytes = struct.field(pytree_node=False)
    byte_col_offsets: bytes = struct.field(pytree_node=False)
    byte_col_sq_offsets: bytes = struct.field(pytree_node=False)
    byte_col_sizes_shape: bytes = struct.field(pytree_node=False)
    byte_col_indices_shape: bytes = struct.field(pytree_node=False)
    byte_col_offsets_shape: bytes = struct.field(pytree_node=False)
    byte_col_sq_offsets_shape: bytes = struct.field(pytree_node=False)

    @property
    def col_sizes(self):
        shape = np.frombuffer(self.byte_col_sizes_shape, dtype=np.int64)
        return np.frombuffer(self.byte_col_sizes, dtype=np.int64).reshape(shape)
    
    @property
    def col_indices(self):
        shape = np.frombuffer(self.byte_col_indices_shape, dtype=np.int64)
        return np.frombuffer(self.byte_col_indices, dtype=np.int64).reshape(shape)
    
    @property
    def col_offsets(self):
        shape = np.frombuffer(self.byte_col_offsets_shape, dtype=np.int64)
        return np.frombuffer(self.byte_col_offsets, dtype=np.int64).reshape(shape)
    
    @property
    def col_sq_offsets(self):
        shape = np.frombuffer(self.byte_col_sq_offsets_shape, dtype=np.int64)
        return np.frombuffer(self.byte_col_sq_offsets, dtype=np.int64).reshape(shape)

    @classmethod
    def create(
        cls, 
        offset: int,
        row_size: int,
        n_blocks: int,
        col_sizes: npt.NDArray[np.int64],
        col_indices: npt.NDArray[np.int64],
        col_offsets: npt.NDArray[np.int64],
        col_sq_offsets: npt.NDArray[np.int64],
    ):
        return RowGroup(
            offset, 
            row_size, 
            n_blocks, 
            np.array(col_sizes).tobytes(), 
            np.array(col_indices).tobytes(), 
            np.array(col_offsets).tobytes(), 
            np.array(col_sq_offsets).tobytes(),
            np.array(col_sizes.shape).tobytes(),
            np.array(col_indices.shape).tobytes(),
            np.array(col_offsets.shape).tobytes(),
            np.array(col_sq_offsets.shape).tobytes(),
            )


@struct.dataclass
class VBRMatrix(BlockMatrixBase):
    """A variable block row matrix format."""

    data: jax.Array
    # Using bytes instead of tuples. Numpy arrays are not allowed by jax
    byte_col_indices: bytes = struct.field(pytree_node=False)
    byte_row_ptr: bytes = struct.field(pytree_node=False)
    byte_row_sizes: bytes = struct.field(pytree_node=False)
    byte_col_sizes: bytes = struct.field(pytree_node=False)
    byte_row_begin_indices: bytes = struct.field(pytree_node=False)
    byte_col_begin_indices: bytes = struct.field(pytree_node=False)

    @property
    def col_indices(self):
        return np.frombuffer(self.byte_col_indices, dtype=np.int64)

    @property
    def row_ptr(self):
        return np.frombuffer(self.byte_row_ptr, dtype=np.int64)

    @property
    def row_sizes(self):
        return np.frombuffer(self.byte_row_sizes, dtype=np.int64)

    @property
    def col_sizes(self):
        return np.frombuffer(self.byte_col_sizes, dtype=np.int64)

    @property
    def row_begin_indices(self):
        return np.frombuffer(self.byte_row_begin_indices, dtype=np.int64)

    @property
    def col_begin_indices(self):
        return np.frombuffer(self.byte_col_begin_indices, dtype=np.int64)

    @classmethod
    def create(
        cls,
        data: jax.Array,
        col_indices: npt.NDArray[np.int64],
        row_ptr: npt.NDArray[np.int64],
        row_sizes: npt.NDArray[np.int64],
        col_sizes: npt.NDArray[np.int64],
    ):
        row_sizes = np.array(row_sizes)
        col_sizes = np.array(col_sizes)
        row_begin_indices = np.cumulative_sum(row_sizes, include_initial=True)
        col_begin_indices = np.cumulative_sum(col_sizes, include_initial=True)

        new = VBRMatrix(
            data,
            np.array(col_indices).tobytes(),
            np.array(row_ptr).tobytes(),
            row_sizes.tobytes(),
            col_sizes.tobytes(),
            row_begin_indices.tobytes(),
            col_begin_indices.tobytes(),
        )
        return new

    @property
    def n_rows(self):
        return len(self.row_sizes)

    @property
    def n_cols(self):
        return len(self.col_sizes)

    @property
    def shape(self):
        rows = sum(r for r in self.row_sizes)
        cols = sum(c for c in self.col_sizes)
        return (rows, cols)
    
    @property
    def row_groups(self):
        """
        Computes the row groups in this matrix from matrix data 
        """
        
        col_sizes_per_row = np.split(
            self.col_sizes[self.col_indices], self.row_ptr[1:-1]
        )

        max_block_size = 10
        col_sizes_hash = np.array(
            [
                np.sum(sizes * max_block_size ** np.arange(len(sizes)))
                for sizes in col_sizes_per_row
            ]
        )
        row_shapes = np.stack([self.row_sizes, col_sizes_hash])
        diff = np.diff(row_shapes, prepend=row_shapes[:, :1] - 1)
        change_idx = np.sum(diff**2, axis=0) != 0
        run_starts = np.flatnonzero(change_idx)
        run_lengths = np.diff(np.append(run_starts, row_shapes.shape[1]))

        col_sizes_in_groups = [col_sizes_per_row[run_start] for run_start in run_starts]
        row_sizes_in_groups = self.row_sizes[run_starts]

        group_sizes = [
            l * r * np.sum(c)
            for l, r, c in zip(run_lengths, row_sizes_in_groups, col_sizes_in_groups)
        ]
        offsets = np.concatenate(([0], np.cumsum(group_sizes)[:-1]))
        col_offsets = np.concatenate(([0], np.cumsum(self.col_sizes)[:-1]))
        col_sq_offsets = np.concatenate(([0], np.cumsum(self.col_sizes**2)[:-1]))

        n_blocks_per_group = (np.array(self.row_ptr)[1:] - np.array(self.row_ptr)[:-1])[
            run_starts
        ]

        col_indices_per_group = np.split(
            self.col_indices, np.array(self.row_ptr[:-1])[run_starts]
        )[1:]

        col_offsets_per_group = np.split(
            col_offsets[self.col_indices], np.array(self.row_ptr[:-1])[run_starts]
        )[1:]

        col_sq_offsets_per_group = np.split(
            col_sq_offsets[self.col_indices], np.array(self.row_ptr[:-1])[run_starts]
        )[1:]

        col_indices_in_groups = [
            c.reshape(-1, b) for b, c in zip(n_blocks_per_group, col_indices_per_group)
        ]
        col_offsets_in_groups = [
            c.reshape(-1, b) for b, c in zip(n_blocks_per_group, col_offsets_per_group)
        ]
        col_sq_offsets_in_groups = [
            c.reshape(-1, b)
            for b, c in zip(n_blocks_per_group, col_sq_offsets_per_group)
        ]

        return [
            (l.item(), RowGroup.create(o, r, n, c, id, cof, csqof))
            for l, o, r, n, c, id, cof, csqof in zip(
                run_lengths,
                offsets,
                row_sizes_in_groups,
                n_blocks_per_group,
                col_sizes_in_groups,
                col_indices_in_groups,
                col_offsets_in_groups,
                col_sq_offsets_in_groups,
            )
        ]

    def tree_flatten(self):
        children = (self.data,)
        aux_data = (self.col_indices, self.row_ptr, self.row_sizes, self.col_sizes)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)
    
    def get_row_from_group(self, group_offset, group_id, row_size, col_sizes):
        local_offset = row_size * np.sum(col_sizes) * group_id
        matrices = []
        for col_size in col_sizes:
            mat = jax.lax.dynamic_slice(
                self.data, (group_offset + local_offset,), (col_size * row_size).item()
            ).reshape(row_size, col_size)
            local_offset += row_size * col_size
            matrices.append(mat)
        return matrices

    def to_scalar_matrix(self):
        full_mat = jnp.zeros(self.shape)
        slice_begin = 0
        col_id = 0
        for k in range(len(self.row_ptr) - 1):
            row_id = k
            for i in range(self.row_ptr[k], self.row_ptr[k + 1]):
                col_id = self.col_indices[i]
                shape = (self.row_sizes[row_id], self.col_sizes[col_id])
                slice_end = slice_begin + shape[0] * shape[1]
                mat = self.data[slice_begin:slice_end].reshape(shape)
                row_ptr = self.row_begin_indices[row_id]
                col_ptr = self.col_begin_indices[col_id]
                full_mat = full_mat.at[
                    row_ptr : row_ptr + shape[0], col_ptr : col_ptr + shape[1]
                ].set(mat)
                slice_begin = slice_end

        return full_mat

    def mul_vector(self, vec):
        res = jnp.zeros(sum(self.row_sizes))
        slice_begin = 0
        for k in range(len(self.row_ptr) - 1):
            row_id = k
            for i in range(self.row_ptr[k], self.row_ptr[k + 1]):
                col_id = self.col_indices[i]
                shape = (self.row_sizes[row_id], self.col_sizes[col_id])
                slice_end = slice_begin + shape[0] * shape[1]
                mat = self.data[slice_begin:slice_end].reshape(shape)

                col_slice_begin = self.col_begin_indices[col_id]
                col_slice_end = self.col_begin_indices[col_id + 1]
                row_slice_begin = self.row_begin_indices[row_id]
                row_slice_end = self.row_begin_indices[row_id + 1]

                vec_data = vec[col_slice_begin:col_slice_end]

                local_result = mat @ vec_data
                res = res.at[row_slice_begin:row_slice_end].set(
                    res[row_slice_begin:row_slice_end] + local_result
                )
                slice_begin = slice_end
        return res

    def vector_mul(self, vec):
        res = jnp.zeros(sum(self.col_sizes))
        slice_begin = 0
        for k in range(len(self.row_ptr) - 1):
            row_id = k
            for i in range(self.row_ptr[k], self.row_ptr[k + 1]):
                col_id = self.col_indices[i]
                shape = (self.row_sizes[row_id], self.col_sizes[col_id])
                slice_end = slice_begin + shape[0] * shape[1]
                mat = self.data[slice_begin:slice_end].reshape(shape)

                col_slice_begin = self.col_begin_indices[col_id]
                col_slice_end = self.col_begin_indices[col_id + 1]
                row_slice_begin = self.row_begin_indices[row_id]
                row_slice_end = self.row_begin_indices[row_id + 1]

                vec_data = vec[row_slice_begin:row_slice_end]

                local_result = vec_data @ mat
                res = res.at[col_slice_begin:col_slice_end].set(
                    res[col_slice_begin:col_slice_end] + local_result
                )
                slice_begin = slice_end

        return res
    
    def grouped_vector_mul(self, vec):
        res = jnp.zeros(sum(self.col_sizes))
        row_groups = self.row_groups
        group_vec_slice_begin = 0

        for num_blocks_rows, group in row_groups:
            row_size = group.row_size
            col_sizes = group.col_sizes
            col_offsets = group.col_offsets

            def vector_mul_block_row(j, res):
                vec_slice_begin = group_vec_slice_begin + j*row_size
                vec_slice = jax.lax.dynamic_slice(vec, (vec_slice_begin,), (row_size,))
                mat = self.get_row_from_group(group.offset, j, row_size, col_sizes)
                for k, local_res in enumerate([vec_slice @ A for A in mat]):
                    idx = jnp.array(col_offsets)[j, k]
                    old = jax.lax.dynamic_slice(res, (idx,), (local_res.shape[0],))
                    res = jax.lax.dynamic_update_slice(res, old + local_res, (idx,))
                return res
            
            res = jax.lax.fori_loop(0, num_blocks_rows, vector_mul_block_row, res)
            group_vec_slice_begin += num_blocks_rows*row_size
        return res


    def plot(self):
        import matplotlib.pyplot as plt

        plt.figure()
        scalar_repr = self.to_scalar_matrix()
        ignore_imshow = False
        while isinstance(scalar_repr, jax.core.Tracer):
            if hasattr(scalar_repr, "val"):
                scalar_repr = scalar_repr.val[0]
            else:
                ignore_imshow = True
                break
        if not ignore_imshow:
            plt.imshow(
                jnp.abs(self.to_scalar_matrix()) < 1e-22,
                cmap="YlGnBu",
                aspect="auto",
                interpolation="none",
            )
            plt.imshow(
                jnp.abs(jnp.abs(self.to_scalar_matrix()) - 1) < 1e-6,
                cmap="Reds",
                aspect="auto",
                interpolation="none",
                alpha=0.6,
            )
        else:
            plt.gca().invert_yaxis()
        plt.tight_layout()

        for i, (row_begin_prev, row_begin) in enumerate(
            zip(self.row_begin_indices, self.row_begin_indices[1:])
        ):
            plt.axhline(row_begin - 0.5, linewidth=2)
            for j in range(self.row_sizes[i]):
                pass
            plt.text(
                0,
                (row_begin_prev + row_begin) * 0.5 - 0.5,
                i,
                verticalalignment="center",
            )

        for i, (col_begin_prev, col_begin) in enumerate(
            zip(self.col_begin_indices, self.col_begin_indices[1:])
        ):
            plt.axvline(col_begin - 0.5)
            plt.text(
                (col_begin_prev + col_begin) * 0.5 - 0.5,
                0,
                i,
                horizontalalignment="center",
            )

        plt.show()


if __name__ == "__main__":
    jnp.set_printoptions(edgeitems=30, linewidth=1000)
    jax.config.update("jax_enable_x64", True)
    key = jax.random.PRNGKey(4)
    key, key1 = jax.random.split(key)
    key, key2 = jax.random.split(key)

    A = jnp.ones([2, 2]) * 1.0
    B = jnp.ones([2, 2]) * 2.0
    C = jnp.ones([4, 1]) * 3.0
    D = jnp.ones([2, 3]) * 4.0

    data = jnp.concatenate([A, B, C, D], axis=None)
    col_indices = (0, 1, 2, 1)
    row_ptr = (0, 2, 3, 4)
    row_sizes = (2, 2, 4)
    col_sizes = (2, 1, 3)
    bcsr_mat = VBRMatrix(data, col_indices, row_ptr, row_sizes, col_sizes)

    vec = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    vec2 = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    res = bcsr_mat.mul_vector(vec)
    res2 = bcsr_mat.vector_mul(vec2)
    print(bcsr_mat.to_scalar_matrix())
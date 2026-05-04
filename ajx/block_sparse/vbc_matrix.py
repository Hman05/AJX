import jax
import jax.numpy as jnp
from ajx.block_sparse.base import BlockMatrixBase
from jax.tree_util import register_pytree_node_class

from flax import struct
import numpy as np
import numpy.typing as npt


@struct.dataclass
class VBCMatrix(BlockMatrixBase):
    """A variable block column matrix format."""

    data: jax.Array
    byte_row_indices: bytes = struct.field(pytree_node=False)
    byte_col_ptr: bytes = struct.field(pytree_node=False)
    byte_row_sizes: bytes = struct.field(pytree_node=False)
    byte_col_sizes: bytes = struct.field(pytree_node=False)
    byte_row_begin_indices: bytes = struct.field(pytree_node=False)
    byte_col_begin_indices: bytes = struct.field(pytree_node=False)

    @property
    def row_indices(self):
        return np.frombuffer(self.byte_row_indices, dtype=np.int64)

    @property
    def col_ptr(self):
        return np.frombuffer(self.byte_col_ptr, dtype=np.int64)

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
        row_indices: npt.NDArray[np.int64],
        col_ptr: npt.NDArray[np.int64],
        row_sizes: npt.NDArray[np.int64],
        col_sizes: npt.NDArray[np.int64],
    ):
        row_sizes = np.array(row_sizes)
        col_sizes = np.array(col_sizes)

        row_begin_indices = np.cumulative_sum(row_sizes, include_initial=True)
        col_begin_indices = np.cumulative_sum(col_sizes, include_initial=True)

        new = VBCMatrix(
            data,
            np.array(row_indices).tobytes(),
            np.array(col_ptr).tobytes(),
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

    def to_scalar_matrix(self):
        full_mat = jnp.zeros(self.shape)
        slice_begin = 0
        col_id = 0
        for k in range(len(self.col_ptr) - 1):
            col_id = k
            for i in range(self.col_ptr[k], self.col_ptr[k + 1]):
                row_id = self.row_indices[i]
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
        res = jnp.zeros(sum(bcsc_mat.row_sizes))
        slice_begin = 0
        for k in range(len(self.col_ptr) - 1):
            col_id = k
            for i in range(self.col_ptr[k], self.col_ptr[k + 1]):
                row_id = self.row_indices[i]
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

    def tree_flatten(self):
        children = (self.data,)
        aux_data = (
            self.row_indices,
            self.col_ptr,
            self.row_sizes,
            self.col_sizes,
        )
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)

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
    row_indices = (0, 1, 2, 1)
    col_ptr = (0, 2, 3, 4)
    row_sizes = (2, 2, 4)
    col_sizes = (2, 1, 3)
    bcsc_mat = VBCMatrix(data, row_indices, col_ptr, row_sizes, col_sizes)
    bcsc_mat.plot()

    vec = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    res = bcsc_mat.mul_vector(vec)
    print(bcsc_mat.to_scalar_matrix())

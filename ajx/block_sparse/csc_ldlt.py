from functools import lru_cache
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ajx.block_sparse.vbc_matrix import VBCMatrix
from ajx.group_operations import grouped_fori_loop


def ldlt_solve(A: VBCMatrix, b: jax.Array, refinement_steps: int = 0) -> jax.Array:
    """
    Solve the linear system Ax = b using an LDLT factorization, where A is stored in a variable block column (VBC) matrix format.

    Limitations
    ----------
    It is assumed that fill-ins are known ahead of time and allocated as zeros in A.

    Consecutive block columns with the same structure are processed by a single `jax.lax.fori_loop`,
    so the compilation time grows with the number of distinct column structures rather than the size of A.

    The solver does not pivot, so A must be symmetric positive definite.

    Parameters
    ----------
    A: VBCMatrix
        The matrix to be factorized. A is assumed to be symmetric.
    b: jax.Array
        The right hand side vector.
    refinement_steps: int
        The number of iterative refinement steps. Each step solves A dx = b - Ax
        using the same factorization, which recovers accuracy lost in the
        factorization at the cost of one extra multiplication and substitution.

    Returns
    -------
    jax.Array:
        The solution x to the linear system Ax = b.
    """
    LDLT = jax.jit(ldlt_factor)(A)
    x = jax.jit(ldlt_substitution)(LDLT, b)

    for _ in range(refinement_steps):
        r = b - jax.jit(symmetric_mul_vector)(A, x)
        x = x + jax.jit(ldlt_substitution)(LDLT, r)
    return x


class ColumnGroup(NamedTuple):
    """
    Consecutive block columns with the same pivot size and the same sizes of the sub-diagonal blocks.
    Each table has one row per column in the group.

    pivot_size: int
        The size of the diagonal block.
    sub_sizes: Tuple[int]
        The row sizes of the sub-diagonal blocks.
    pivot_ptr: (count,)
        Where the diagonal block starts in A.data.
    sub_ptr: (count, n_sub)
        Where each sub-diagonal block starts in A.data.
    update_ptr: (count, n_sub * (n_sub + 1) / 2)
        Where each block updated by the Schur complement starts in A.data, see ldlt_factor.
    pivot_begin: (count,)
        Where the pivot rows start in a vector.
    sub_begin: (count, n_sub)
        Where the rows of each sub-diagonal block start in a vector.
    """

    pivot_size: int
    sub_sizes: Tuple[int, ...]
    pivot_ptr: jax.Array
    sub_ptr: jax.Array
    update_ptr: jax.Array
    pivot_begin: jax.Array
    sub_begin: jax.Array


@lru_cache(maxsize=None)
def _column_groups(byte_row_indices, byte_col_ptr, byte_block_sizes):
    """Symbolic phase: find the ColumnGroups of A from its (hashable) structure."""
    row_indices = np.frombuffer(byte_row_indices, dtype=np.int64)
    col_ptr = np.frombuffer(byte_col_ptr, dtype=np.int64)
    block_sizes = np.frombuffer(byte_block_sizes, dtype=np.int64)
    n_cols = len(col_ptr) - 1

    block_cols = np.repeat(np.arange(n_cols), np.diff(col_ptr))
    block_ptr = np.cumulative_sum(
        block_sizes[row_indices] * block_sizes[block_cols], include_initial=True
    )
    vector_begin = np.cumulative_sum(block_sizes, include_initial=True)

    groups = []
    for k in range(n_cols):
        assert col_ptr[k + 1] - col_ptr[k] > 0  # At least one entry per column
        assert row_indices[col_ptr[k]] == k  # The first entry is the diagonal block
        rows = row_indices[col_ptr[k] + 1 : col_ptr[k + 1]]
        assert np.all(np.diff(rows) > 0), "Row indices must be sorted"

        # The Schur complement of the pivot updates the blocks (rows[j], rows[i]) for j >= i,
        # which are stored in column rows[i]
        update_ptr = []
        for i, r in enumerate(rows):
            rows_in_r = row_indices[col_ptr[r] : col_ptr[r + 1]]
            for c in rows[i:]:
                pos = np.flatnonzero(rows_in_r == c)
                assert pos.size == 1, f"The fill-in block ({c}, {r}) is not allocated"
                update_ptr.append(block_ptr[col_ptr[r] + pos[0]])

        signature = (int(block_sizes[k]), tuple(int(block_sizes[r]) for r in rows))
        column = (
            block_ptr[col_ptr[k]],
            block_ptr[col_ptr[k] + 1 : col_ptr[k + 1]],
            update_ptr,
            vector_begin[k],
            vector_begin[rows],
        )
        if groups and groups[-1][0] == signature:
            groups[-1][1].append(column)
        else:
            groups.append((signature, [column]))

    def table(columns, field, width):
        return np.array([c[field] for c in columns], dtype=np.int32).reshape(len(columns), width)

    result = []
    for (pivot_size, sub_sizes), columns in groups:
        n_sub = len(sub_sizes)
        result.append(
            (
                len(columns),
                pivot_size,
                sub_sizes,
                table(columns, 0, 1)[:, 0],
                table(columns, 1, n_sub),
                table(columns, 2, n_sub * (n_sub + 1) // 2),
                table(columns, 3, 1)[:, 0],
                table(columns, 4, n_sub),
            )
        )
    return tuple(result)


def column_groups(A: VBCMatrix):
    """The ColumnGroups of A, in column order, as (count, group) pairs for grouped_fori_loop."""
    assert np.all(A.row_sizes == A.col_sizes)
    groups = _column_groups(A.byte_row_indices, A.byte_col_ptr, A.byte_row_sizes)
    return [
        (count, ColumnGroup(pivot_size, sub_sizes, *map(jnp.asarray, tables)))
        for count, pivot_size, sub_sizes, *tables in groups
    ]


def _get_block(data, ptr, shape):
    return jax.lax.dynamic_slice(data, (ptr,), (shape[0] * shape[1],)).reshape(shape)


def _set_block(data, ptr, block):
    return jax.lax.dynamic_update_slice(data, block.flatten(), (ptr,))


def _get_rows(x, begin, size):
    return jax.lax.dynamic_slice(x, (begin,), (size,))


def _set_rows(x, begin, rows):
    return jax.lax.dynamic_update_slice(x, rows, (begin,))


def ldlt_factor(A: VBCMatrix) -> VBCMatrix:
    """
    LDLT-factors the symmetric matrix A such that A=LDL^T where L is block lower triangular and D is block diagonal.
    The result is computed in-place for efficiency.
    - The block-lower diagonal part of A contains L.
    - The block-diagonal part of A contains the cholesky-factored blocks of D. These blocks are the result of
       applying the jax.scipy.linalg.cho_factor function on the blocks of D.

    Each column k is eliminated by a sparse Schur complement with the diagonal block P as pivot:
    1. Cholesky factor P in-place
    2. Compute the L-entries L_i = Eta_i @ inv(P) of the sub-diagonal blocks Eta_i
    3. Reduction step: A[r_j, r_i] <- A[r_j, r_i] - Eta_j @ L_i.T for the sub-diagonal rows r_i <= r_j
    4. Replace the sub-diagonal blocks with the L-entries

    row col-> 0   1   2   3   4
     0     |  P  L.T     L.T L.T
     1     | Eta  P
     2     |          P
     3     | Eta     Eta  P
     4     | Eta     Eta Eta  P

    Parameters
    ----------
    A: VBCMatrix
        The matrix to be factorized. A is assumed to be symmetric.

    Returns
    -------
    VBCMatrix:
        The in-place factored version of A.
    """

    def eliminate_column(_, g, group, data):
        p, sizes = group.pivot_size, group.sub_sizes
        chol, _ = jax.scipy.linalg.cho_factor(
            _get_block(data, group.pivot_ptr[g], (p, p)), lower=True
        )
        data = _set_block(data, group.pivot_ptr[g], chol)

        eta = [_get_block(data, group.sub_ptr[g, i], (s, p)) for i, s in enumerate(sizes)]
        # Solve P @ L.T = Eta.T using the cholesky factorization
        lt = [jax.scipy.linalg.cho_solve((chol, True), e.T) for e in eta]

        n = 0
        for i in range(len(sizes)):
            for j in range(i, len(sizes)):
                ptr = group.update_ptr[g, n]
                block = _get_block(data, ptr, (sizes[j], sizes[i]))
                data = _set_block(data, ptr, block - eta[j] @ lt[i])
                n += 1

        for i, l in enumerate(lt):
            data = _set_block(data, group.sub_ptr[g, i], l.T)
        return data

    data = grouped_fori_loop(column_groups(A), eliminate_column, A.data)
    return A.replace(data=data)


def ldlt_substitution(LDLT: VBCMatrix, b: jax.Array) -> jax.Array:
    """
    Solve LDL^T x = b given an in-place factored matrix from ldlt_factor.
    """
    y = forward_substitution(LDLT, b)
    z = diagonal_scaling(LDLT, y)
    return backward_substitution(LDLT, z)


def forward_substitution(A: VBCMatrix, b: jax.Array) -> jax.Array:
    """Solve L y = b, where the diagonal blocks of L are identities."""

    def eliminate_column(_, g, group, y):
        p = group.pivot_size
        y_pivot = _get_rows(y, group.pivot_begin[g], p)
        for i, s in enumerate(group.sub_sizes):
            L_block = _get_block(A.data, group.sub_ptr[g, i], (s, p))
            begin = group.sub_begin[g, i]
            y = _set_rows(y, begin, _get_rows(y, begin, s) - L_block @ y_pivot)
        return y

    return grouped_fori_loop(column_groups(A), eliminate_column, b)


def diagonal_scaling(A: VBCMatrix, y: jax.Array) -> jax.Array:
    """Solve D z = y, where the diagonal blocks of A hold the cholesky factors of D."""

    def scale_column(_, g, group, z):
        p = group.pivot_size
        chol = _get_block(A.data, group.pivot_ptr[g], (p, p))
        begin = group.pivot_begin[g]
        z_pivot = jax.scipy.linalg.cho_solve((chol, True), _get_rows(z, begin, p))
        return _set_rows(z, begin, z_pivot)

    return grouped_fori_loop(column_groups(A), scale_column, y)


def backward_substitution(A: VBCMatrix, z: jax.Array) -> jax.Array:
    """Solve L^T x = z, where the diagonal blocks of L are identities."""

    def solve_column(_, g, group, x):
        # Traverse the columns in reverse
        g = group.pivot_ptr.shape[0] - 1 - g
        p = group.pivot_size
        x_col = _get_rows(x, group.pivot_begin[g], p)
        for i, s in enumerate(group.sub_sizes):
            L_block = _get_block(A.data, group.sub_ptr[g, i], (s, p))
            x_col = x_col - L_block.T @ _get_rows(x, group.sub_begin[g, i], s)
        return _set_rows(x, group.pivot_begin[g], x_col)

    return grouped_fori_loop(reversed(column_groups(A)), solve_column, z)


def symmetric_mul_vector(A: VBCMatrix, x: jax.Array) -> jax.Array:
    """
    Compute A @ x for a symmetric matrix A where only the block lower triangular part is stored.
    The diagonal blocks are assumed to be stored in full.
    """

    def mul_column(_, g, group, res):
        p = group.pivot_size
        begin = group.pivot_begin[g]
        x_col = _get_rows(x, begin, p)
        res_col = _get_block(A.data, group.pivot_ptr[g], (p, p)) @ x_col
        for i, s in enumerate(group.sub_sizes):
            block = _get_block(A.data, group.sub_ptr[g, i], (s, p))
            row_begin = group.sub_begin[g, i]
            res = _set_rows(
                res, row_begin, _get_rows(res, row_begin, s) + block @ x_col
            )
            # Contribution from the (implicitly stored) upper triangular block
            res_col = res_col + block.T @ _get_rows(x, row_begin, s)
        return _set_rows(res, begin, _get_rows(res, begin, p) + res_col)

    return grouped_fori_loop(column_groups(A), mul_column, jnp.zeros_like(x))


def set_diagonal_zero(A):
    slice_begin = 0
    for k in range(len(A.col_ptr) - 1):
        col_id = k
        for i in range(A.col_ptr[k], A.col_ptr[k + 1]):
            row_id = A.row_indices[i]
            shape = (A.row_sizes[row_id], A.col_sizes[col_id])
            slice_end = slice_begin + shape[0] * shape[1]
            if row_id == col_id:
                A.data = A.data.at[slice_begin:slice_end].set(0.0)
            slice_begin = slice_end
    return A


def as_lower_triangular(A):
    slice_begin = 0
    for k in range(len(A.col_ptr) - 1):
        col_id = k
        for i in range(A.col_ptr[k], A.col_ptr[k + 1]):
            row_id = A.row_indices[i]
            shape = (A.row_sizes[row_id], A.col_sizes[col_id])
            slice_end = slice_begin + shape[0] * shape[1]
            if row_id == col_id:
                identity = jnp.eye(shape[0]).flatten()
                A.data = A.data.at[slice_begin:slice_end].set(identity.flatten())
            slice_begin = slice_end
    return A


def pick_diagonal(A):
    slice_begin = 0
    for k in range(len(A.col_ptr) - 1):
        col_id = k
        for i in range(A.col_ptr[k], A.col_ptr[k + 1]):
            row_id = A.row_indices[i]
            shape = (A.row_sizes[row_id], A.col_sizes[col_id])
            slice_end = slice_begin + shape[0] * shape[1]
            if row_id != col_id:
                A.data = A.data.at[slice_begin:slice_end].set(0.0)
            if row_id == col_id:
                diag_block = A.data[slice_begin:slice_end].reshape(shape)
                diag_low = jnp.tril(diag_block)
                full_diag = diag_low @ diag_low.T
                A.data = A.data.at[slice_begin:slice_end].set(full_diag.flatten())
            slice_begin = slice_end
    return A


if __name__ == "__main__":
    jnp.set_printoptions(edgeitems=30, linewidth=1000)
    jax.config.update("jax_enable_x64", True)
    import pickle

    with open("block_mat_problematic.pickle", "rb") as handle:
        A = pickle.load(handle)
        A.data = jnp.float64(A.data)
    with open("rhs.pickle", "rb") as handle:
        rhs = pickle.load(handle)
    import matplotlib.pyplot as plt

    # plt.imshow(
    #     jnp.abs(A.to_scalar_matrix()) < 1e-22,
    #     cmap="YlGnBu",
    #     aspect="auto",
    #     interpolation="none",
    # )
    # plt.show()
    import copy

    # x2 = ldlt_solve(copy.deepcopy(A), rhs)

    # traced = jax.jit(ldlt_solve).trace(copy.deepcopy(A), rhs)
    # lowered = traced.lower()
    # with open("Output.txt", "w") as f:
    #     f.write(lowered.as_text())

    A_upper = set_diagonal_zero(copy.deepcopy(A)).to_scalar_matrix().T
    A_dense = A.to_scalar_matrix() + A_upper

    LDL = ldlt_factor(copy.deepcopy(A))
    LDL_jit = jax.jit(ldlt_factor)(copy.deepcopy(A))
    L_sparse = as_lower_triangular(copy.deepcopy(LDL))
    D_sparse = pick_diagonal(copy.deepcopy(LDL))
    L = L_sparse.to_scalar_matrix()
    D = D_sparse.to_scalar_matrix()
    res = L @ D @ L.T
    result1 = jnp.linalg.norm(res - A_dense)

    zcmp = D
    y = forward_substitution(LDL, copy.deepcopy(rhs))
    rhs_cmp = L @ y
    z = diagonal_scaling(LDL, copy.deepcopy(y))
    y_cmp = D @ z
    x = backward_substitution(LDL, copy.deepcopy(z))
    z_cmp = L.T @ x
    result2 = jnp.linalg.norm(rhs_cmp - rhs)
    result3 = jnp.linalg.norm(y_cmp - y)
    result4 = jnp.linalg.norm(z_cmp - z)
    result5 = jnp.linalg.norm(A_dense @ x - rhs)
    # result6 = jnp.linalg.norm(A_dense @ x2 - rhs)

    x3 = jnp.linalg.solve(A_dense, rhs)
    result7 = jnp.linalg.norm(A_dense @ x3 - rhs)

    x4 = jax.scipy.linalg.solve(A_dense, rhs, assume_a="pos")
    result8 = jnp.linalg.norm(A_dense @ x4 - rhs)

    lu_and_lower = jax.scipy.linalg.cho_factor(A_dense, lower=True)

    x5 = jax.scipy.linalg.cho_solve(lu_and_lower, rhs)
    result9 = jnp.linalg.norm(A_dense @ x5 - rhs)

    print(f"LDLT: {result1}")
    print(f"FWD: {result2}")
    print(f"Scaling: {result3}")
    print(f"BACK: {result4}")
    print(f"FINAL: {result5}")
    # print(f"FINAL_JIT: {result6}")
    print(f"FINAL_LU: {result7}")
    print(f"FINAL_POS: {result8}")
    print(f"FINAL_CHO: {result9}")
    plt.imshow(
        # jnp.log(jnp.abs(res - A_dense) + 1e-10),
        abs(LDL.to_scalar_matrix() - LDL_jit.to_scalar_matrix()) < 1e-10,
        cmap="YlGnBu",
        aspect="auto",
        interpolation="none",
    )
    plt.show()
    # compare = A.to_scalar_matrix()
    pass

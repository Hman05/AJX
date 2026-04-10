import jax.numpy as jnp
import jax
from ajx.block_sparse.vbc_matrix import VBCMatrix
from itertools import product

import numpy as np


def ldlt_solve(A: VBCMatrix, data_ptr: jax.Array, b: jax.Array) -> jax.Array:
    """
    Solve the linear system Ax = b using an LDLT factorization, where A is stored in a variable block row (VBC) matrix format.

    Limitations
    ----------
    It is assumed that fill-ins are known ahead of time and allocated as zeros in A.

    The current implementation relies on Python for-loops. When used under `jax.jit`, these loops are unrolled before compilation,
    which may lead to long compilation times and suboptimal runtime performance.

    Additionally, `jax.numpy.linalg.solve` is highly optimized and will generally outperform this implementation. It also uses pivoting to
    improve numerical conditioning, whereas this solver does not.

    Parameters
    ----------
    A: VBCMatrix
        The matrix to be factorized. A is assumed to be symmetric.
    data_ptr: jax.Array
        A list of indices to where each block starts in the contiguous data array A.data. Also known as rediction scattering indexation (rsi).
    b: jax.Array
        The right hand side vector.

    Returns
    -------
    jax.Array:
        The solution x to the linear system Ax = b.
    """
    LDLT = jax.jit(ldlt_factor, static_argnums=1)(A, data_ptr)
    y = jax.jit(forward_substitution)(LDLT, b)
    z = jax.jit(diagonal_scaling)(LDLT, y)
    x = jax.jit(backward_substitution)(LDLT, z)
    return x


ignore_asserts = True


def inner_reduce(
    row_id,
    col_id,
    A_col_ptr,
    A_data,
    Eta_col,
    Lt,
    row_indices,
    search_range,
    data_ptr,
):
    # Compute the local reduction term
    # S_loc = eta[S_row_id_begin:S_row_id_end] @ Lt[:, S_col_id_begin:S_col_id_end]

    S_loc = Eta_col @ Lt

    # To update the correct slice in memory, we make use of the data_ptr array
    # which jumps us to the correct location of A.data if the block index is known.
    # We know the first block index of the column from the col_ptr array.
    # To find the offset, we need to search for the column.
    # If it is not found, we were given an invalid matrix.
    offset = jnp.nonzero(
        row_indices[search_range[0] : search_range[1]] == row_id,
        size=1,
        fill_value=-1,
    )[0][0]
    idx = A_col_ptr[col_id] + offset
    ptr_slice_begin = data_ptr[idx]
    size = S_loc.shape[0] * S_loc.shape[1]
    new = jax.lax.dynamic_slice(A_data, (ptr_slice_begin,), (size,)) - S_loc.flatten()
    A_data = jax.lax.dynamic_update_slice(A_data, new, (ptr_slice_begin,))
    return A_data


def sparse_schur_reduction(
    A_data: jax.Array,
    Lt_buffer: jax.Array,
    k: int,
    pivot_slice_begin: int,
    A_col_ptr: np.array,
    block_sizes: np.array,
    row_indices: np.array,
    data_ptr: np.array,
):
    """
    Perform a sparse schur complement, using block k on the diagonal as the pivot. This operation can be broken into four steps:
    1. Cholesky factor the pivot in-place
    2. Compute the L-entries in a temp buffer
    3. Reduction step (Schur complement)
    4. Replace the sub-diagonal with the L-entries in the temp buffer

    Parameters
    ----------
    A_data: jax.Array,
        The data of the matrix to be factorized.
    Lt_buffer: jax.Array
        A temporary buffer to store the intermediate result. An attempt to control memory allocation.
    k: int
        The index of the diagonal block to use as pivot.
    pivot_slice_begin: np.array
        The index of the starting location of the pivot data in A_data.
    A_col_ptr: np.array
        Points to the start of each block column in row_indices.
    block_sizes: np.array
        The size of each block (the matrix is symmetric)
    data_ptr: np.array
        Points to the start of each block in A_data.
    Returns
    -------
    jax.Array:
        The solution x to the linear system Ax = b.
    int:
        Points to the end of the subdiagonal in A_data. This is the same as pointing to the pivot with index k+1.
    """

    # Pick the pivot from the A_data
    pivot_shape = (block_sizes[k], block_sizes[k])
    pivot_slice_end = pivot_slice_begin + pivot_shape[0] * pivot_shape[1]
    pivot = A_data[pivot_slice_begin:pivot_slice_end].reshape(pivot_shape)

    # Factorize the pivot in-place.
    # This is done both to efficiently compute the L-factors
    # but also to save work in the diagonal scaling phase.
    chol, _ = jax.scipy.linalg.cho_factor(pivot, lower=True)
    # Store the result in A to save the work for the diagonal scaling phase
    A_data = A_data.at[pivot_slice_begin:pivot_slice_end].set(chol.flatten())

    # The subdiagonal data starts where the pivot ends
    subdiag_slice_begin = pivot_slice_end

    # Find the row indices of the subdiagonal. We call these "reduction indices".
    row_range = range(A_col_ptr[k] + 1, A_col_ptr[k + 1])
    reduction_ids = [row_indices[i] for i in row_range]
    n_red = len(reduction_ids)

    # Form the "L"-entries and store the data inside an "Lt-buffer"
    buffer_offset = 0
    for i in range(n_red):
        # Get the associated row, shape and slice in A_data
        row_id = reduction_ids[i]
        shape = (block_sizes[row_id], block_sizes[k])
        eta_slice_begin = subdiag_slice_begin + buffer_offset
        eta_slice_end = eta_slice_begin + shape[0] * shape[1]
        Eta = A_data[eta_slice_begin:eta_slice_end].reshape(shape)
        # The L factors are given by L = Eta @ inv(P) where P is the pivot
        # We instead solve the system P @ L.T = Eta.T making use of the cholesky factorization
        Lt = jax.scipy.linalg.cho_solve((chol, True), Eta.T)
        # Update the buffer
        buffer_slice_end = buffer_offset + shape[0] * shape[1]
        Lt_buffer = Lt_buffer.at[buffer_offset:buffer_slice_end].set(Lt.flatten())
        buffer_offset = buffer_slice_end
        assert buffer_slice_end < 1000

    # Loop over the recuction indices.
    # We only loop over the lower triangular indices (utilizing symmetry).
    # row col-> 0   1   2   3   4
    #  0     |  P  L.T     L.T L.T
    #  1     | Eta  P
    #  2     |          P
    #  3     | Eta     Eta  P
    #  4     | Eta     Eta Eta  P
    buffer_row_offset = 0
    col_slice_begin = subdiag_slice_begin
    # We start by looping over the rows with nonzero blocks
    for i in range(n_red):
        row_id = reduction_ids[i]

        # Find L[i,k] in Lt_buffer
        shape = (block_sizes[row_id], block_sizes[k])
        buffer_slice_end = buffer_row_offset + shape[0] * shape[1]
        Lt = Lt_buffer[buffer_row_offset:buffer_slice_end].reshape(
            block_sizes[k], block_sizes[row_id]
        )
        buffer_row_offset = buffer_row_offset + shape[0] * shape[1]

        search_range = (A_col_ptr[row_id], A_col_ptr[row_id + 1])  # ptr

        # Next, we loop over the columns with nonzero blocks, excluding the
        # strictly upper triangular blocks

        # To get the right offset, we need to loop through the block_size arrays from 0 to i
        buffer_col_offset = (
            sum(block_sizes[reduction_ids[j]] for j in range(0, i)) * block_sizes[k]
        )
        for j in range(i, n_red):
            col_id = reduction_ids[j]
            # The final step is to update the block entry at (row_id, col_id)
            shape = (block_sizes[col_id], block_sizes[k])

            col_slice_begin = subdiag_slice_begin + buffer_col_offset
            col_slice_end = col_slice_begin + shape[0] * shape[1]
            Eta_col = A_data[col_slice_begin:col_slice_end].reshape(shape)
            # Compute P <- P - L @ Eta_col or Eta <- Eta - L @ Eta

            # Compute the local reduction term
            S_loc = Eta_col @ Lt

            # To update the correct slice in memory, we make use of the data_ptr array
            # which jumps us to the correct location of A.data if the block index is known.
            # We know the first block index of the column from the col_ptr array.
            # To find the offset, we need to search for the column.
            # If it is not found, we were given an invalid matrix.
            offset = jnp.nonzero(
                row_indices[search_range[0] : search_range[1]] == col_id,
                size=1,
                fill_value=-1,
            )[0][0]
            offset = np.nonzero(
                row_indices[search_range[0] : search_range[1]] == col_id
            )[0][0]
            idx = A_col_ptr[row_id] + offset
            ptr_slice_begin = data_ptr[idx]
            size = S_loc.shape[0] * S_loc.shape[1]
            new = A_data[ptr_slice_begin : ptr_slice_begin + size] - S_loc.flatten()
            # new = (
            #     jax.lax.dynamic_slice(A_data, (ptr_slicA_e_begin,), (size,))
            #     - S_loc.flatten()
            # )
            # A_data = jax.lax.dynamic_update_slice(A_data, new, (ptr_slice_begin,))
            A_data = A_data.at[ptr_slice_begin : ptr_slice_begin + size].set(new)

            buffer_col_offset = buffer_col_offset + shape[0] * shape[1]

    # Update the subdiagonal with L
    buffer_slice_begin = 0
    eta_slice_begin = subdiag_slice_begin
    for i in range(n_red):
        row_id = reduction_ids[i]
        shape = (block_sizes[row_id], block_sizes[k])
        eta_slice_end = eta_slice_begin + shape[0] * shape[1]
        buffer_slice_end = buffer_slice_begin + shape[0] * shape[1]
        Lt = Lt_buffer[buffer_slice_begin:buffer_slice_end].reshape(
            block_sizes[k], block_sizes[row_id]
        )
        A_data = A_data.at[eta_slice_begin:eta_slice_end].set(Lt.T.flatten())
        buffer_slice_begin = buffer_slice_end
        eta_slice_begin = eta_slice_end
        assert buffer_slice_end < 1000
    # assert col_slice_end == row_slice_end
    return A_data, pivot_slice_end + buffer_row_offset


def ldlt_factor(A: VBCMatrix, data_ptr):
    """
    LDLT-factors the symmetric matrix A such that A=LDL^T where L is block lower triangular and D is block diagonal.
    The result is computed in-place for efficiency.
    - The block-lower diagonal part of A contains L.
    - The block-diagonal part of A contains the cholesky-factored blocks of D. These blocks are the result of
       applying the jax.scipy.linalg.cho_factor function on the blocks of D.

    Parameters
    ----------
    A: VBCMatrix
        The matrix to be factorized. A is assumed to be symmetric.
    data_ptr: jax.Array
        A list of indices to where each block starts in the contigous data array A.data. Also known as rediction scattering indexation (rsi).

    Returns
    -------
    VBCMatrix:
        The in-place factored version of A.
    """

    # Make sure that we have symmetry in the block shapes
    block_sizes = A.row_sizes

    # TODO: Compute an appropriate size for the Lt-buffer
    Lt_buffer = jnp.zeros(600)
    row_indices = np.array(A.row_indices)

    # Symbolic phase
    # 1. Asserts
    assert np.all(np.array(A.row_sizes) == np.array(A.col_sizes))
    subdiag_shapes = np.zeros((len(A.col_ptr) - 1, 2), dtype=int)
    for k in range(len(A.col_ptr) - 1):
        assert A.col_ptr[k + 1] - A.col_ptr[k] > 0  # At least one entry per column
        assert A.row_indices[A.col_ptr[k]] == k  # The first entry is the diagonal block
        row_range = range(A.col_ptr[k] + 1, A.col_ptr[k + 1])
        row_indices_in_column_k = [A.row_indices[i] for i in row_range]
        # Check that the row indices in column k are sorted
        assert all(
            row_indices_in_column_k[i] <= row_indices_in_column_k[i + 1]
            for i in range(len(row_indices_in_column_k) - 1)
        )

        subdiag_shape = (
            sum(A.row_sizes[i] for i in row_indices_in_column_k),
            A.row_sizes[k],
        )
        subdiag_shapes[k] = np.array(subdiag_shape)

    # possible_blocks = product(set(block_sizes), set(block_sizes))
    # func_list = []
    # for block_size in possible_blocks:
    #     size = block_size[0] * block_size[1]

    A_data_ptr = 0
    # cond_list = {}
    A_data = A.data
    for k in range(len(A.col_ptr) - 1):
        # Select the pivot (block on the diagonal).
        # The pivot is guaranteed (from prior asserts) to be
        # made up of the first slice in the current "column chunk".

        A_data, A_data_ptr = sparse_schur_reduction(
            A_data,
            Lt_buffer,
            k,
            A_data_ptr,
            A.col_ptr,
            block_sizes,
            row_indices,
            data_ptr,
        )
        pass

    return VBCMatrix.create(A_data, A.row_indices, A.col_ptr, A.row_sizes, A.col_sizes)


def forward_substitution(A, b):
    # Make sure that we have symmetry in the block shapes
    if not ignore_asserts:
        assert jnp.all(jnp.array(A.row_sizes) == jnp.array(A.col_sizes))
    block_sizes = A.row_sizes
    y = b
    slice_begin = 0
    slice_end = slice_begin
    for k in range(len(A.col_ptr) - 1):
        col_id = k

        # Make sure that the pivot column has at least one entry
        if not ignore_asserts:
            assert A.col_ptr[k + 1] - A.col_ptr[k] > 0
            # Make sure that the diagonal exists and that there are no entries above the diagonal
            assert A.row_indices[A.col_ptr[k]] == k

        col_slice_begin = A.col_begin_indices[col_id]
        col_slice_end = A.col_begin_indices[col_id + 1]
        y_pivot = y[col_slice_begin:col_slice_end]

        # The pivot is the block on the diagonal (treated as an identity matrix here)
        pivot_shape = (block_sizes[k], block_sizes[k])
        slice_end = slice_begin + pivot_shape[0] * pivot_shape[1]
        # Jump over the pivot
        slice_begin = slice_end

        # Loop over the sub-diagonal
        for i in range(A.col_ptr[k] + 1, A.col_ptr[k + 1]):
            row_id = A.row_indices[i]
            shape = (A.row_sizes[row_id], A.col_sizes[col_id])
            slice_end = slice_begin + shape[0] * shape[1]
            L_block = A.data[slice_begin:slice_end].reshape(shape)

            row_slice_begin = A.col_begin_indices[row_id]
            row_slice_end = A.col_begin_indices[row_id + 1]
            y_row = y[row_slice_begin:row_slice_end]
            y_row_updated = y_row - L_block @ y_pivot
            y = y.at[row_slice_begin:row_slice_end].set(y_row_updated)

            slice_begin = slice_end
    return y


def diagonal_scaling(A, y):
    # Make sure that we have symmetry in the block shapes
    if not ignore_asserts:
        assert jnp.all(jnp.array(A.row_sizes) == jnp.array(A.col_sizes))
    block_sizes = A.row_sizes
    z = y
    slice_begin = 0
    slice_end = slice_begin
    for k in range(len(A.col_ptr) - 1):
        col_id = k
        if not ignore_asserts:
            # Make sure that the pivot column has at least one entry
            assert A.col_ptr[k + 1] - A.col_ptr[k] > 0
            # Make sure that the diagonal exists and that there are no entries above the diagonal
            assert A.row_indices[A.col_ptr[k]] == k

        col_slice_begin = A.col_begin_indices[col_id]
        col_slice_end = A.col_begin_indices[col_id + 1]
        z_pivot = z[col_slice_begin:col_slice_end]

        # The pivot is the block on the diagonal (treated as an identity matrix here)
        pivot_shape = (block_sizes[k], block_sizes[k])
        slice_end = slice_begin + pivot_shape[0] * pivot_shape[1]
        # The diagonal block is expected to be cholesky factored
        chol = A.data[slice_begin:slice_end].reshape(pivot_shape)
        z_scaled = jax.scipy.linalg.cho_solve((chol, True), z_pivot)
        z = z.at[col_slice_begin:col_slice_end].set(z_scaled)
        slice_begin = slice_end

        # Jump over the sub-diagonal
        for i in range(A.col_ptr[k] + 1, A.col_ptr[k + 1]):
            row_id = A.row_indices[i]
            shape = (A.row_sizes[row_id], A.col_sizes[col_id])
            slice_end = slice_end + shape[0] * shape[1]
        slice_begin = slice_end
    return z


def backward_substitution(A, z):
    # Make sure that we have symmetry in the block shapes
    if not ignore_asserts:
        assert jnp.all(jnp.array(A.row_sizes) == jnp.array(A.col_sizes))
    block_sizes = A.row_sizes
    x = z
    # We now have to traverse the data in reverse...
    slice_end = A.data.shape[0]
    slice_begin = slice_end
    for k in reversed(range(len(A.col_ptr) - 1)):
        col_id = k
        if not ignore_asserts:
            # Make sure that the pivot column has at least one entry
            assert A.col_ptr[k + 1] - A.col_ptr[k] > 0
            # Make sure that the diagonal exists and that there are no entries above the diagonal
            assert A.row_indices[A.col_ptr[k]] == k

        col_slice_begin = A.col_begin_indices[col_id]
        col_slice_end = A.col_begin_indices[col_id + 1]

        # Loop over the sub-diagonal in reverse
        x_col = x[col_slice_begin:col_slice_end]
        for i in reversed(range(A.col_ptr[k] + 1, A.col_ptr[k + 1])):
            row_id = A.row_indices[i]
            shape = (A.row_sizes[row_id], A.col_sizes[col_id])
            slice_begin = slice_end - shape[0] * shape[1]
            L_block = A.data[slice_begin:slice_end].reshape(shape)

            row_slice_begin = A.col_begin_indices[row_id]
            row_slice_end = A.col_begin_indices[row_id + 1]
            x_row = x[row_slice_begin:row_slice_end]
            # Update x_col
            x_col = x_col - L_block.T @ x_row

            slice_end = slice_begin
        # Store the result
        x = x.at[col_slice_begin:col_slice_end].set(x_col)

        # The pivot is the block on the diagonal (treated as an identity matrix here)
        pivot_shape = (block_sizes[k], block_sizes[k])
        slice_begin = slice_end - pivot_shape[0] * pivot_shape[1]
        # Jump over the pivot
        slice_end = slice_begin
    return x


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


def create_block_ptr(A):
    rsi = jnp.zeros(len(A.row_indices), dtype=int)
    ptr = 0
    for k in range(len(A.col_ptr) - 1):
        col_id = k
        for i in range(A.col_ptr[k], A.col_ptr[k + 1]):
            row_id = A.row_indices[i]
            size = A.row_sizes[row_id] * A.col_sizes[col_id]
            rsi = rsi.at[i].set(ptr)
            ptr = ptr + size
    return rsi


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

    ptr = create_block_ptr(A)

    # x2 = ldlt_solve(copy.deepcopy(A), ptr, rhs)

    # traced = jax.jit(ldlt_solve).trace(copy.deepcopy(A), ptr, rhs)
    # lowered = traced.lower()
    # with open("Output.txt", "w") as f:
    #     f.write(lowered.as_text())

    A_upper = set_diagonal_zero(copy.deepcopy(A)).to_scalar_matrix().T
    A_dense = A.to_scalar_matrix() + A_upper

    LDL = ldlt_factor(copy.deepcopy(A), ptr)
    LDL_jit = jax.jit(ldlt_factor)(copy.deepcopy(A), ptr)
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

import jax
import jax.numpy as jnp
import os

from ajx.group_operations import sparse_blockrow_mul_blockdiag, sparse_blockrow_mul_vec
from ajx.constraints.base import ConstraintType

@jax.jit
def gauss_seidel_dense_naive(A, b, x0, Nit):
    """
    Implementation of a Gauss Seidel solver operating on dense matrices.
    INPUTS:
        A: system matrix, n x n jax array.
        b: right hand side, n x 1 jax array.
        x0: initial guess, n x 1 jax array.
        Nit: number of gauss seidel iterations to use.
    OUTPUTS:
        x: solution, n x 1 jax array.
    """
    L = jnp.tril(A, k=0)
    U = jnp.triu(A, k=1)
    x = x0

    def gauss_seidel_step(i, state):
        L = state[0]
        U = state[1]
        b = state[2]
        x = state[3]
        x = jax.scipy.linalg.solve_triangular(L, b - U @ x, lower=True)
        return (L, U, b, x)

    _, _, _, x = jax.lax.fori_loop(0, Nit, gauss_seidel_step, (L, U, b, x))

    assert jnp.allclose(A @ x, b, rtol=1e-5, atol=1e-8)

    return x


@jax.jit(static_argnames=("h", "Nit"))
def projected_gauss_seidel_dense(
    gvel, lbda0, G, M_inv, Sigma, h, f_ext, q, lbda_limits, Nit
):
    """
    Solves dense system on the form
    | M  -G_k^T   | | u_k+1 | = | M @ v_k + h*f_ext |
    | G_k  Sigma  | | lbda  | = | q |

    INPUTS:
        gvel: ndof x 1, jax array, current generalized velocity.
        lbda0: nc x 1, jax array
        G: nc x ndof, jax array
        M_inv: ndof x ndof, jax array
        Sigma: nc x 1, jax array, diagonal entries of Sigma matrix
        h: timestep size
        f_ext: ndof x 1, external force applied to rigid bodies.
        q: nc x 1, jax array, right hand side data.
        lbda_limits: nc x 2, jax array, multiplier limits. Each row has multiplier limits, (lbda_min, lbda_max).
        Nit: number of iterations
    """

    nc = G.shape[0]

    # To precompute M^-1 @ G^T
    M_inv_GT = M_inv @ G.T

    # To precompute diagonal elements of schur complement matrix S = G @ M_inv @ G.T + Sigma
    S_diag = jnp.einsum("ik,ki->i", G, M_inv_GT) + Sigma

    def constraint_body(c, state):
        """
        This is the inner loop body, handling the update of lbda for each constraint
        """
        u, lbda = state
        r = q[c] - jnp.dot(G[c, :], u) - Sigma[c] * lbda[c]
        delta_lbda = jnp.divide(r, S_diag[c])

        # Projection step
        old_lbda = lbda[c]
        lbda = lbda.at[c].set(
            jnp.clip(lbda[c] + delta_lbda, lbda_limits[c, 0], lbda_limits[c, 1])
        )

        # Update step with correct delta lambda
        delta_lbda_update = lbda[c] - old_lbda
        u = u + M_inv_GT[:, c] * delta_lbda_update

        return (u, lbda)

    def pgs_body(j, state):
        """
        This is the outer loop body, handling the 'Nit' iterations
        """
        return jax.lax.fori_loop(0, nc, constraint_body, state)

    lbda = lbda0
    u = gvel + h * M_inv @ f_ext + M_inv_GT @ lbda

    u, lbda = jax.lax.fori_loop(0, Nit, pgs_body, (u, lbda))
    return u, lbda


@jax.jit(static_argnames=("G_block_row_size", "h", "Nit"))
def projected_gauss_seidel_block_dense(
    gvel, lbda0, G, G_block_row_size, M_inv, Sigma, h, f_ext, q, Nit, constraint_metadata
):
    """
    Solves dense system on the form
    | M  -G_k^T   | | u_k+1 | = | M @ v_k + h*f_ext |
    | G_k  Sigma  | | lbda  | = | q |

    This solver supports constraints that constrain the same number of degrees of freedom.
    It is specified through the G_block_row_size argument.

    INPUTS:
        gvel: n_bodies x 1, jax array, current generalized velocity.
        lbda0: (nc*block_row_size) x 1, jax array
        G_data: jax array of size (nc*block_row_size) x n_bodies
        G_block_row_size: number of rows in G per constraint block, a positive integer.
        M_inv: n_bodies x n_bodies, jax array
        Sigma: (nc*block_row_size) x 1, jax array, diagonal entries of Sigma matrix
        h: timestep size
        f_ext: n_bodies x 1, external force applied to rigid bodies.
        q: (nc*block_row_size) x 1, jax array, right hand side data.
        lbda_limits: (nc*block_row_size) x 2, jax array, multiplier limits. Each row has multiplier limits, (lbda_min, lbda_max).
        Nit: number of iterations
    """

    nc = G.shape[0] // G_block_row_size  # Number of constraints
    constraint_type = constraint_metadata["constraint_type"]

    # To precompute M^-1 @ G^T
    M_inv_GT = M_inv @ G.T

    # To precompute necessary elements of schur complement matrix S = G @ M_inv @ G.T + Sigma.
    def schur_body(c, schur):
        row_start = c * G_block_row_size

        Gi = jax.lax.dynamic_slice(G, (row_start, 0), (G_block_row_size, G.shape[1]))
        Sigma_block = jax.lax.dynamic_slice(Sigma, (row_start,), (G_block_row_size,))

        # To compute and update with the Schur block Sii, corrsponding to constraint 'c'.
        S_block = Gi @ M_inv @ Gi.T + jnp.diag(Sigma_block)
        schur = jax.lax.dynamic_update_slice(schur, S_block, (row_start, 0))
        return schur

    schur_blocks = jnp.zeros([nc * G_block_row_size, G_block_row_size])
    schur_blocks = jax.lax.fori_loop(0, nc, schur_body, schur_blocks)

    def constraint_body(c, state):
        """
        This is the inner loop body, handling the update of 'lbda' and 'u' for each constraint
        """
        u, lbda = state

        # To find the start index for constraint block.
        row_start = c * G_block_row_size

        qi = jax.lax.dynamic_slice(q, (row_start,), (G_block_row_size,))
        Gi = jax.lax.dynamic_slice(G, (row_start, 0), (G_block_row_size, G.shape[1]))
        Gi_u = jnp.dot(Gi, u)
        sigma_ii = jax.lax.dynamic_slice(Sigma, (row_start,), (G_block_row_size,))
        lbda_i = jax.lax.dynamic_slice(lbda, (row_start,), (G_block_row_size,))
        Sigma_lbda_i = sigma_ii * lbda_i

        ri = qi - Gi_u - Sigma_lbda_i
        Sii = jax.lax.dynamic_slice(
            schur_blocks, (row_start, 0), (G_block_row_size, G_block_row_size)
        )
        delta_lbda_i = jnp.linalg.solve(Sii, ri)

        # Projection step
        lbda = jax.lax.dynamic_update_slice(lbda, update_multipliers(lbda_i, delta_lbda_i, constraint_type[c]), (row_start,))

        # Update step with correct delta lambda
        delta_lbda_update = (
            jax.lax.dynamic_slice(lbda, (row_start,), (G_block_row_size,)) - lbda_i
        )
        u = (
            u
            + jax.lax.dynamic_slice(
                M_inv_GT, (0, row_start), (M_inv_GT.shape[0], G_block_row_size)
            )
            @ delta_lbda_update
        )

        return (u, lbda)

    def pgs_body(j, state):
        """
        This is the outer loop body, handling the 'Nit' iterations
        """
        return jax.lax.fori_loop(0, nc, constraint_body, state)

    lbda = lbda0
    u = gvel + h * M_inv @ f_ext + M_inv_GT @ lbda

    u, lbda = jax.lax.fori_loop(0, Nit, pgs_body, (u, lbda))
    return u, lbda


def update_multipliers(lbda, delta_lbda, constraint_type):
    """
    This function is responsible for updating the multipliers of a constraint in the projected gauss seidel solver. If the constraint is of type HINGE, we projected the multipliers according to a friction law lbda_motor <= mu*r*||lbda_load||, where mu is a friction coefficient, and r is an effective radius.
    INPUTS:
        lbda: array of size 6x1, the multipliers for this constraint.
        delta_lbda: array of size 6x1, the multiplier update for this constraint.
    OUTPUTS:
        lbda: array of size 6x1, the updated multipliers for this constraint.
    """
    lbda = lbda + delta_lbda
    
    # These are passed as environment variables, and the user is responsible for these values.
    mu = float(os.environ.get("MU", "0.0"))
    r = float(os.environ.get("EFFECTIVE_RADIUS", "1.0"))
    
    def hinge_fn(lbda):
        hinge_load = jnp.linalg.norm(lbda[:3], ord=2)
        lbda_motor = jnp.clip(lbda[5], -mu*r*hinge_load, mu*r*hinge_load)     # The 6th multiplier corresponds to the HINGE-axis
        return lbda.at[5].set(lbda_motor)
    def not_hinge_fn(lbda):
        return lbda
    
    lbda = jax.lax.cond(constraint_type==ConstraintType.HINGE.value, hinge_fn, not_hinge_fn, lbda)
    return lbda


@jax.jit(static_argnames=("h", "Nit"))
def projected_gauss_seidel_sparse(
    gvel, lbda0, G, M_inv, Sigma, h, f_ext, q, lbda_limits, Nit
):
    """
    Solves sparse system on the form
    | M  -G_k^T   | | u_k+1 | = | M @ v_k + h*f_ext |
    | G_k  Sigma  | | lbda  | = | q |

    INPUTS:
        gvel: ndof x 1, jax array, current generalized velocity.
        lbda0: nc x 1, jax array
        G: nc x ndof, ajx.block_sparse.VBRMatrix
        M_inv: ajx.block_sparse.SVBDMatrix
        Sigma: jax array, from which a diagonal matrix can be formed
        h: timestep size
        f_ext: ndof x 1, external force applied to rigid bodies.
        q: nc x 1, jax array, right hand side data.
        lbda_limits: nc x 2, jax array, multiplier limits. Each row has multiplier limits, (lbda_min, lbda_max).
        Nit: number of iterations
    """

    schur_block_diag = get_schur_block_diagonal_elements(G, M_inv, Sigma)
    group_row_offsets = get_group_row_offsets(
        G
    )  # Row offset for each group in the actual dense matrix G

    def constraint_body(group_index, j, group, state):
        """
        This routine is intended to calculate one PGS-iteration per constraint
        """
        u, lbda = state

        # Indexing the rows of the jth block in group
        row_start = (
            group_row_offsets[group_index] + j * group.row_size
        )  # Row start index in full matrix
        Gi = G.get_row_from_group(group.offset, j, group.row_size, group.col_sizes)
        qi = jax.lax.dynamic_slice(q, (row_start,), (group.row_size,))
        lbda_i = jax.lax.dynamic_slice(lbda, (row_start,), (group.row_size,))
        sigma_i = jax.lax.dynamic_slice(Sigma, (row_start,), (group.row_size,))

        Gi_u = sparse_blockrow_mul_vec(
            Gi, u, group.col_sizes, jnp.array(group.col_offsets)[j]
        )
        ri = qi - Gi_u - sigma_i * lbda_i
        Sii = jax.lax.dynamic_slice(
            schur_block_diag[group_index],
            (j * group.row_size, 0),
            (group.row_size, group.row_size),
        )

        # Solve for delta lambda and project the multipliers
        delta_lbda_i = jnp.linalg.solve(Sii, ri)
        lbda_lower_limit = jax.lax.dynamic_slice(
            lbda_limits, (row_start, 0), (group.row_size, 1)
        ).flatten()
        lbda_upper_limit = jax.lax.dynamic_slice(
            lbda_limits, (row_start, 1), (group.row_size, 1)
        ).flatten()
        lbda = jax.lax.dynamic_update_slice(
            lbda,
            jnp.clip(lbda_i + delta_lbda_i, lbda_lower_limit, lbda_upper_limit),
            (row_start,),
        )

        delta_lbda_update = (
            jax.lax.dynamic_slice(lbda, (row_start,), (group.row_size,)) - lbda_i
        )
        u = update_generalized_velocity(
            u,
            M_inv,
            Gi,
            delta_lbda_update,
            group.col_sizes,
            jnp.array(group.col_offsets)[j],
            jnp.array(group.col_sq_offsets)[j],
        )

        return (u, lbda)

    def pgs_body(j, state):
        """
        This is the outer loop body, handling the 'Nit' iterations.
        The grouped_fori_loop, returns the updated state = (u, lbda) after one pgs-iteration.
        """
        u, lbda = state
        return pgs_grouped_fori_loop(G.groups, constraint_body, (u, lbda))

    # To initialize the multipliers and the generalized velocity
    lbda = lbda0
    u = gvel + h * M_inv.mul_vector(f_ext) + M_inv.mul_vector(G.vector_mul(lbda))

    # This is the entry point for the PGS-solver
    u, lbda = jax.lax.fori_loop(0, Nit, pgs_body, (u, lbda))
    return u, lbda


def get_schur_block_diagonal_elements(G, M_inv, Sigma):
    """
    INPUTS:
        G: nc x ndof, ajx.block_sparse.VBRMatrix
        M_inv: ajx.block_sparse.SVBDMatrix, inverse mass matrix
        Sigma: jax array, from which a diagonal matrix can be formed.
    OUTPUTS:
        schur_block_diag: list of len(G.groups) with jax arrays of shape (group.row_size*num_block_rows, group.row_size)
    """

    # Pre-allocate initial state for jit-compatibility. group.col_offsets seems to be an array of size (num_block_rows, num_blocks) with col_offset for each block in the block row.
    initial_state = tuple(
        jnp.zeros((group.row_size * num_block_rows, group.row_size))
        for num_block_rows, group in G.groups
    )
    group_row_offsets = get_group_row_offsets(
        G
    )  # Row offset for each group in the actual dense matrix G

    def single_schur_block_body(group_index, j, group, state):
        row_start = (
            group_row_offsets[group_index] + j * group.row_size
        )  # Row start index in full matrix

        Gi = G.get_row_from_group(
            group.offset, j, group.row_size, group.col_sizes
        )  # To get data in G[row_start:row_]
        Gi_M_inv = sparse_blockrow_mul_blockdiag(
            Gi,
            M_inv.data,
            group.col_sizes,
            jnp.array(group.col_sq_offsets)[j],
        )

        schur_block = sum(
            [A @ B.T for A, B in zip(Gi_M_inv, Gi)]
        )  # G @ M_inv @ G.T for subset of rows
        sigma_i = jax.lax.dynamic_slice(Sigma, (row_start,), (group.row_size,))
        schur_block = schur_block.at[jnp.diag_indices(group.row_size)].add(sigma_i)

        state_i = jax.lax.dynamic_update_slice(
            state[group_index], schur_block, (j * group.row_size, 0)
        )
        state = state[:group_index] + (state_i,) + state[group_index + 1 :]
        return state

    schur_block_diag = pgs_grouped_fori_loop(
        G.groups, single_schur_block_body, initial_state
    )
    return schur_block_diag


def update_generalized_velocity(
    u, M_inv, Gi, delta_lbda_update, col_sizes, col_offsets, col_sq_offsets
):
    """
    INPUTS:
        u: jax array of size nb x 1, generalized velocity.
        M_inv: ajx.block_sparse.SVBDMatrix, inverse mass matrix
        Gi: list of constraint jacobian blocks
        delta_lbda_update: jax array of size r x 1, multiplier update increment
        col_sizes: list of column sizes for each of the constraint jacobian blocks (See RowGroup class in VBRMatrix)
        col_offsets: list of column offset for each block (See RowGroup class in VBRMatrix)
        col_sq_offsets: list of squared column offset for each block (See RowGroup class in VBRMatrix)
    OUTPUTS:
        U: updated generalized velocity
    """
    assert len(col_sq_offsets) == len(col_sizes)

    Gt_lbda_i = [
        A.T @ delta_lbda_update for A in Gi
    ]  # Now we need to index which rigid bodies are effected in u and M_inv

    assert len(Gt_lbda_i) == len(col_sizes)

    vel_update_blocks = [
        jax.lax.dynamic_slice(
            M_inv.data,
            (col_sq_offsets[i],),
            (col_sizes[i] * col_sizes[i],),
        ).reshape(col_sizes[i], col_sizes[i])
        @ Gt_lbda_i[i]
        for i in range(len(Gt_lbda_i))
    ]

    for j, (start_idx, size) in enumerate(zip(col_offsets, col_sizes)):
        new_u = jax.lax.dynamic_slice(u, (start_idx,), (size,)) + jnp.array(
            vel_update_blocks[j]
        )
        u = jax.lax.dynamic_update_slice(u, new_u, (start_idx,))
    return u


def pgs_grouped_fori_loop(groups, body_fun, init_val):
    val = init_val
    for group_index, (count, group_data) in enumerate(groups):

        def body_fun_aug(i, carry):
            return body_fun(group_index, i, group_data, carry)

        val = jax.lax.fori_loop(0, count, body_fun_aug, val)

    return val


def get_group_row_offsets(G):
    """
    Calculates the row offset where each group starts in the actual dense matrix G.
    INPUTS:
        G: nc x ndof, ajx.block_sparse.VBRMatrix
    OUTPUTS:
        row_offsets: jax array of size number_of_groups x 1.
    """
    num_rows_per_group = tuple(
        num_block_rows * g.row_size for num_block_rows, g in G.groups
    )
    row_offsets = jnp.cumulative_sum(jnp.array((0,) + num_rows_per_group))[:-1]
    return row_offsets


def check_finite(x):
    if not jnp.all(jnp.isfinite(x)):
        raise ValueError("x contains NaN or Inf!")

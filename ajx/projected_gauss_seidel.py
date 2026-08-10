import jax
import jax.numpy as jnp
import os

from ajx.group_operations import sparse_blockrow_mul_blockdiag, sparse_blockrow_mul_vec


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
            jnp.clip(lbda[c] + delta_lbda, lbda_limits[0, c], lbda_limits[1, c])
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
        lbda_limits: 2 x nc, jax array, multiplier limits. Each column has multiplier limits, (lbda_min, lbda_max).
        Nit: number of iterations
    """

    lbda_lower_limits = lbda_limits[0, :]
    lbda_upper_limits = lbda_limits[1, :]

    group_row_offsets = get_group_row_offsets(
        G
    )  # Row offset for each group in the actual dense matrix G

    # To cache precomputed data. It is unclear though if it improves performance.
    group_meta_data = []
    for _, group_data in G.row_groups:
        local_data = {
            "n_blocks": group_data.n_blocks,
            "row_size": group_data.row_size,
            "col_sizes": group_data.col_sizes,
            "col_offsets": jnp.array(group_data.col_offsets),
            "col_sq_offsets": jnp.array(group_data.col_sq_offsets),
        }
        group_meta_data.append(local_data)
    group_meta_data = tuple(group_meta_data)

    # To precompute blocks Gi, and inverse schur diagonal blocks.
    G_blocks, schur_block_diag_inv = precompute_row_blocks_data(G, M_inv, Sigma, group_meta_data)

    def constraint_body(group_index, j, group, state):
        """
        This routine is intended to calculate one PGS-iteration per constraint
        """
        u, lbda = state
        
        group_cache_data = group_meta_data[group_index]
        group_col_offsets = group_cache_data["col_offsets"]
        group_col_sq_offsets = group_cache_data["col_sq_offsets"]
        group_row_size = group_cache_data["row_size"]
        group_col_sizes = group_cache_data["col_sizes"]

        # Indexing the rows of the jth block in group
        row_start = (
            group_row_offsets[group_index] + j * group_row_size
        )  # Row start index in full matrix

        #Gi = G.get_row_from_group(group.offset, j, group_row_size, group_col_sizes)  # To get data in G from block row j
        Gi = tuple(blocks[j] for blocks in G_blocks[group_index])
        qi = jax.lax.dynamic_slice(q, (row_start,), (group_row_size,))
        lbda_i = jax.lax.dynamic_slice(
            lbda, (row_start,), (group_row_size,)
        )  # lbda[row_slice_idx]
        sigma_i = jax.lax.dynamic_slice(Sigma, (row_start,), (group_row_size,))

        Gi_u = sparse_blockrow_mul_vec(Gi, u, group_col_sizes, group_col_offsets[j])
        ri = qi - Gi_u - sigma_i * lbda_i
        Sii_inv = schur_block_diag_inv[group_index][j]

        # Solve for delta lambda and project the multipliers
        delta_lbda_i = Sii_inv @ ri
        lbda_lower_limit = jax.lax.dynamic_slice(
            lbda_lower_limits, (row_start,), (group_row_size,)
        )
        lbda_upper_limit = jax.lax.dynamic_slice(
            lbda_upper_limits, (row_start,), (group_row_size,)
        )
        lbda_i_clipped = jnp.clip(
            lbda_i + delta_lbda_i, lbda_lower_limit, lbda_upper_limit
        )
        lbda = jax.lax.dynamic_update_slice(lbda, lbda_i_clipped, (row_start,))

        delta_lbda_update = lbda_i_clipped - lbda_i
        u = update_generalized_velocity(
            u,
            M_inv,
            Gi,
            delta_lbda_update,
            group_col_sizes,
            group_col_offsets[j],
            group_col_sq_offsets[j],
        )
        return (u, lbda)

    def pgs_body(j, state):
        """
        This is the outer loop body, handling the 'Nit' iterations.
        The grouped_fori_loop, returns the updated state = (u, lbda) after one pgs-iteration.
        """
        u, lbda = state
        return pgs_grouped_fori_loop(G.row_groups, constraint_body, (u, lbda))

    # To initialize the multipliers and the generalized velocity
    lbda = lbda0
    u = (
        gvel
        + h * M_inv.mul_vector(f_ext)
        + M_inv.mul_vector(G.grouped_vector_mul(lbda))
    )

    # This is the entry point for the PGS-solver
    u, lbda = jax.lax.fori_loop(0, Nit, pgs_body, (u, lbda))

    return u, lbda


def precompute_row_blocks_data(G, M_inv, Sigma, group_meta_data):
    """
    Precomputes the row blocks Gi, qi, sigma_i, and Sii^(-1) (schur block inverses).
    INPUT:
        G: nc x ndof, ajx.block_sparse.VBRMatrix
        cache_data:  
    OUTPUT:
        block_cache: single tuple of the same length as the number of groups containing a tuple of Gi blocks per group
    """

    group_row_offsets = get_group_row_offsets(
        G
    )  # Row offset for each group in the actual dense matrix G

    G_row_blocks = []
    schur_block_diag_inv = []
    for group_index, (num_block_rows, group) in enumerate(G.row_groups):
        gmd = group_meta_data[group_index]
        def extract(j):
            Gi = G.get_row_from_group(group.offset, j, gmd["row_size"], gmd["col_sizes"])
            Gi_M_inv = sparse_blockrow_mul_blockdiag(Gi, M_inv.data, gmd["col_sizes"], gmd["col_sq_offsets"][j])
            Sii = jax.vmap(lambda A, B: A @ B.T)(jnp.stack(Gi_M_inv), jnp.stack(Gi)).sum(axis=0)
            row_start = group_row_offsets[group_index] + j * gmd["row_size"]
            sigma_i = jax.lax.dynamic_slice(Sigma, (row_start,), (gmd["row_size"],))
            return Gi, Sii + jnp.diag(sigma_i)
        
        Gi, Sii = jax.vmap(extract)(jnp.arange(num_block_rows))
        G_row_blocks.append(Gi)
        schur_block_diag_inv.append(jnp.linalg.inv(Sii))
    
    return tuple(G_row_blocks), tuple(schur_block_diag_inv)


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

    vel_update_blocks = [
        jax.lax.dynamic_slice(
            M_inv.data, (col_sq_offsets[i],), (col_sizes[i] ** 2,)
        ).reshape(col_sizes[i], col_sizes[i])
        @ (Gi[i].T @ delta_lbda_update)
        for i in range(len(Gi))
    ]
    for j, (start_idx, size) in enumerate(zip(col_offsets, col_sizes)):
        new_u = jax.lax.dynamic_slice(u, (start_idx,), (size,)) + vel_update_blocks[j]
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
        num_block_rows * g.row_size for num_block_rows, g in G.row_groups
    )
    row_offsets = jnp.cumulative_sum(jnp.array((0,) + num_rows_per_group))[:-1]
    return row_offsets

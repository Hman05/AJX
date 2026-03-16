import jax
import jax.numpy as jnp


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

@jax.jit
def gauss_seidel_dense(gvel, lbda0, G, M_inv, Sigma, h, f_ext, q, Nit):
    """
    Solves dense system on the form
    | M  -G_k^T   | | u_k+1 | = | M @ v_k + h*f_ext |
    | G_k  Sigma  | | lbda  | = | q |

    INPUTS:
        gvel: ndof x 1, jax array, current generalized velocity.
        lbda0: nc x 1, jax array
        G: nc x ndof, jax array
        M_inv: ndof x ndof, jax array
        Sigma: nc x nc, jax array, diagonal entries of Sigma matrix
        h: timestep size
        f_ext: ndof x 1, external force applied to rigid bodies.
        Nit: number of iterations
    """

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
        lbda = lbda.at[c].add(delta_lbda)
        u = u + M_inv_GT[:, c] * delta_lbda

        return (u, lbda)

    def gs_body(j, state):
        """
        This is the outer loop body, handling the 'Nit' iterations
        """
        nc = G.shape[0]
        return jax.lax.fori_loop(0, nc, constraint_body, state)

    lbda = lbda0
    u = gvel + h*M_inv @ f_ext + M_inv_GT @ lbda
    
    u, lbda = jax.lax.fori_loop(0, Nit, gs_body, (u, lbda))
    return u, lbda


def gauss_seidel_sparse(lbda0, G, M_inv, Sigma, p, q, Nit):
    """
    INPUTS:
        lbda0: nc x 1, jax array
        G: nc x ndof, ajx.block_sparse.VBRMatrix
        M_inv: ajx.block_sparse.SVBDMatrix
        Sigma: jax array, from which a diagonal matrix can be formed
        p: ndof x 1, jax_array
        q: nc x 1, jax_array
        Nit: number of iterations
    """

    j = 0
    res = 0
    tol = 1e-6
    lbda = lbda0

    schur_diag = 0

    GTlbda = G.vector_mul(
        lbda
    )  # This seems to be the way to compute G^T @ lbda from other usages
    u = M_inv.mul_vector(p) + M_inv.mul_vector(GTlbda)

    # The row dimension in G equals the number of constraints
    num_constraints = G.shape[0]
    while j < Nit and res > tol:
        for c in range(num_constraints):
            tmp = G.mul_vector(u)  # Naive and slow
            r = q[c] - tmp[c] - Sigma[c] * lbda[c]

def check_finite(x):
    if not jnp.all(jnp.isfinite(x)):
        raise ValueError("x contains NaN or Inf!")

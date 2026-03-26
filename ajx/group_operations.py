import jax
import jax.numpy as jnp
from jax import lax


def grouped_fori_loop(groups, body_fun, init_val):
    count_offset = 0
    val = init_val

    for count, group_data in groups:

        def body_fun_aug(i, carry):
            return body_fun(count_offset, i, group_data, carry)

        val = jax.lax.fori_loop(0, count, body_fun_aug, val)
        count_offset += count
    return val


def sparse_blockrow_mul_vec(mat, vec, col_sizes, col_offsets):
    n_blocks = col_sizes.shape[0]
    res = [
        lax.dynamic_slice(
            vec,
            (col_offsets[i],),
            (col_sizes[i],),
        )
        for i in range(n_blocks)
    ]
    return sum([A @ v for A, v in zip(mat, res)])


def sparse_blockrow_mul_blockdiag(blockrow, blockdiag, col_sizes, col_sq_offsets):
    n_blocks = col_sizes.shape[0]
    res = [
        lax.dynamic_slice(
            blockdiag,
            (col_sq_offsets[i],),
            (col_sizes[i] * col_sizes[i],),
        ).reshape(col_sizes[i], col_sizes[i])
        for i in range(n_blocks)
    ]
    return [A @ B for A, B in zip(blockrow, res)]


if __name__ == "__main__":
    jnp.set_printoptions(edgeitems=30, linewidth=1000, precision=2, suppress=True)
    from ajx.example_environments.dlo import DLO
    from ajx.example_environments.dlo import DLO, DLOSettings
    from ajx.simulation import SimulationSettings, Solver
    from ajx.constraints import ConstraintType

    environment = DLO(
        sim_settings=SimulationSettings(0.01, True, Solver.DENSE_LINEAR),
        env_settings=DLOSettings(
            n_bodies=2,
            body_length=0.02,
            constraint_type=ConstraintType.SE3.value,
            loose_end=False,
        ),
    )
    initial_state = environment.state_from_angles(environment.default_param)
    M, M_inv, G, Sigma_data, b_data = environment.sim._assemble_blocks(
        initial_state, environment.default_param
    )

    def step(i, j, group, carry):
        Gi = G.get_row_from_group(group.offset, j, group.row_size, group.col_sizes)
        u = initial_state.gvel.flatten()
        Gi_u = sparse_blockrow_mul_vec(
            Gi, u, group.col_sizes, jnp.array(group.col_offsets)[j]
        )

        Gi_M_inv = sparse_blockrow_mul_blockdiag(
            Gi,
            M_inv.data,
            group.col_sizes,
            jnp.array(group.col_sq_offsets)[j],
        )
        Gi_M_inv_GiT = sum([A @ B.T for A, B in zip(Gi_M_inv, Gi)])
        Sigma_entries = lax.dynamic_slice(
            Sigma_data, (group.offset + j * group.row_size,), (group.row_size,)
        )
        S = Gi_M_inv_GiT  # + jnp.diag(Sigma_entries)
        jax.debug.print("carry: {}", jnp.diag(S))
        carry = carry.at[i + j].set(S)
        return carry

    S_dense = (
        G.to_scalar_matrix()
        @ M_inv.to_scalar_matrix()
        @ G.to_scalar_matrix().T
        # + jnp.diag(Sigma_data)
    )

    S_sparse = jnp.zeros([5, 6, 6])
    result = grouped_fori_loop(G.groups, step, S_sparse)
    S_dense2 = jax.scipy.linalg.block_diag(*result)

    # Should be zero
    delta = jnp.diag(S_dense2) - jnp.diag(S_dense)
    print("|delta|:", jnp.linalg.norm(delta))

    # Test
    j = 1
    Gi_test = G.to_scalar_matrix()[6 * (j + 1) : 6 * (j + 2)]
    group = G.groups[1][1]
    Gi = G.get_row_from_group(group.offset, j, group.row_size, group.col_sizes)
    Gi_M_inv = sparse_blockrow_mul_blockdiag(
        Gi,
        M_inv.data,
        group.col_sizes,
        jnp.array(group.col_sq_offsets)[j],
    )
    Gi_M_inv_GiT = sum([A @ B.T for A, B in zip(Gi_M_inv, Gi)])

    Gi2 = jnp.concatenate(Gi, axis=1)
    Gi_test = G.to_scalar_matrix()[6 * (j + 1) : 6 * (j + 2)]

    res1 = Gi_M_inv_GiT
    res2 = Gi_test @ M_inv.to_scalar_matrix() @ Gi_test.T
    pass

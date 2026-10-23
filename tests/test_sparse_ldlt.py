from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from ajx.block_sparse.csc_ldlt import column_groups, ldlt_solve
from ajx.block_sparse.svbd_matrix import SVBDMatrix
from ajx.block_sparse.vbr_matrix import VBRMatrix
from ajx.simulation import Simulation
from ajx.symbolic import get_constraint_sparsity


class FakeConstraint:
    """Only the parts of a constraint that the sparsity analysis reads."""

    def __init__(self, bodies, degrees):
        self.bodies = bodies
        self.degrees = degrees

    def get_constrained_degrees(self):
        return self.degrees


def random_system(rigid, scalar, constraints, seed=0):
    """Random G, M_inv and Sigma with the structure of the given bodies and constraints."""
    rng = np.random.default_rng(seed)
    size, col_indices, row_ptr, row_sizes, col_sizes, _ = get_constraint_sparsity(
        rigid, scalar, constraints, tuple(rigid), tuple(scalar)
    )
    G = VBRMatrix.create(
        jnp.array(rng.normal(size=size)), col_indices, row_ptr, row_sizes, col_sizes
    )
    blocks = [rng.normal(size=(d, d)) for d in col_sizes]
    M_inv = SVBDMatrix(
        jnp.concatenate([(X @ X.T + len(X) * np.eye(len(X))).ravel() for X in blocks]),
        ((6, len(rigid)), (1, len(scalar))),
    )
    Sigma = jnp.array(rng.uniform(1e-3, 1e-2, size=sum(row_sizes)))
    S = Simulation._schur_reduction(
        SimpleNamespace(constraint_list=constraints), G, M_inv, Sigma
    )
    G_dense = G.to_scalar_matrix()
    S_dense = G_dense @ M_inv.to_scalar_matrix() @ G_dense.T + jnp.diag(Sigma)
    return S, S_dense


def chain(n):
    return [f"b{i}" for i in range(n + 1)], [], [
        FakeConstraint((f"b{i}", f"b{i + 1}"), 6) for i in range(n)
    ]


SYSTEMS = {
    "chain": chain(6),
    # Rigid and scalar bodies have different block sizes
    "mixed_rigid_scalar": (
        ["r0"],
        ["s0", "s1"],
        [
            FakeConstraint(("r0",), 6),
            FakeConstraint(("s0", "s1"), 1),
            FakeConstraint(("r0", "s0"), 1),
        ],
    ),
    # c1 lists its bodies in descending order
    "unsorted_bodies": (
        ["r0", "r1", "r2"],
        [],
        [FakeConstraint(("r0", "r1"), 5), FakeConstraint(("r2", "r1"), 5)],
    ),
    # The first column has 17 sub-diagonal 6x6 blocks
    "star": (
        ["hub"] + [f"b{i}" for i in range(18)],
        [],
        [FakeConstraint(("hub", f"b{i}"), 6) for i in range(18)],
    ),
}


@pytest.mark.parametrize("system", SYSTEMS)
def test_schur_reduction_matches_dense(system):
    S, S_dense = random_system(*SYSTEMS[system])
    np.testing.assert_allclose(
        jnp.tril(S.to_scalar_matrix()), jnp.tril(S_dense), atol=1e-12
    )


@pytest.mark.parametrize("refinement_steps", [0, 2])
@pytest.mark.parametrize("system", SYSTEMS)
def test_ldlt_solve_matches_dense(system, refinement_steps):
    S, S_dense = random_system(*SYSTEMS[system])
    b = jnp.linspace(-1.0, 1.0, S_dense.shape[0])
    x = ldlt_solve(S, b, refinement_steps)
    np.testing.assert_allclose(x, jnp.linalg.solve(S_dense, b), rtol=1e-9)


def test_chain_columns_form_a_constant_number_of_groups():
    # Compilation time scales with the number of groups, not the chain length
    for n in (5, 50):
        S, _ = random_system(*chain(n))
        assert [count for count, _ in column_groups(S)] == [n - 1, 1]

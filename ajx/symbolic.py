import numpy as np
from ajx.constraints import Constraint
from ajx.definitions import RigidBody, ScalarBody
from typing import Tuple


def get_schur_fillin_sparsity(constraint_list: Tuple[Constraint], lower: bool = True):
    constraint_node = [(i, c.bodies) for i, c in enumerate(constraint_list)]

    # Pre-Step 1: Create a directed constraint-body graph, represented as a dict
    edges_directed_map = {}
    for i, bodies_inner in constraint_node:
        edges_directed_map[i] = []
        for j, bodies_outer in constraint_node:
            if i == j:
                continue
            if i > j and lower:
                continue
            if set(bodies_inner).intersection(bodies_outer):
                edges_directed_map[i].append(j)

    # Pre-Step 2: Compute column sparsity pattern
    ncn = len(constraint_list)
    child = {i: [] for i in range(ncn)}
    cols = {}
    for j in range(ncn):
        cols[j] = set(edges_directed_map[j])
        for k in child[j]:
            cols[j] = cols[j].union(cols[k]) - set([j])
        if cols[j]:
            l = min(i for i in cols[j])
            child[l].append(j)

    # Pre-Step 3: Construct sparsity pattern for S
    block_sizes = [c.get_constrained_degrees() for c in constraint_list]
    entries_list = []
    rsi_dict = {}
    row_indices = []
    col_ptr = [0]
    cumulative_pos = 0
    counter = 0
    for col, rows in cols.items():
        triple = (col, col, cumulative_pos)
        rsi_dict[(col, col)] = cumulative_pos
        cumulative_pos += block_sizes[col] * block_sizes[col]
        entries_list.append(triple)
        row_indices.append(col)
        counter += 1
        for row in sorted(rows):
            triple = (row, col, cumulative_pos)
            rsi_dict[(row, col)] = cumulative_pos
            cumulative_pos += block_sizes[row] * block_sizes[col]
            entries_list.append(triple)
            row_indices.append(row)
            counter += 1
        col_ptr.append(counter)

    S_sparsity_pattern = (
        cumulative_pos,
        row_indices,
        col_ptr,
        block_sizes,
        block_sizes,
        rsi_dict,
    )

    return S_sparsity_pattern


def get_constraint_sparsity(
    rigid_body_list: Tuple[RigidBody],
    scalar_body_list: Tuple[ScalarBody],
    constraint_list: Tuple[Constraint],
    rigid_body_names: Tuple[str],
    scalar_body_names: Tuple[str],
):

    nc = len(constraint_list)
    nb = len(rigid_body_list)
    nsb = len(scalar_body_list)
    # Pre-Step 4: Compute reduction scattering indexation and allocate data for G
    row_sizes = np.array([c.get_constrained_degrees() for c in constraint_list])
    col_sizes_rb = np.ones(nb, dtype=int) * 6
    col_sizes_sb = np.ones(nsb, dtype=int) * 1
    col_sizes = np.concatenate([col_sizes_rb, col_sizes_sb])

    G_rsi_dict = (
        {}
    )  # Maps the row (constraint_id) and column (body_id) to the correct location in memory
    col_indices = []
    row_ptr = [0]
    cumulative_pos = 0
    counter = 0
    for row in range(nc):
        constraint = constraint_list[row]
        for body in constraint.bodies:
            dim = -1
            if body in rigid_body_names:
                body_id = rigid_body_names.index(body)
                dim = 6
            elif body in scalar_body_names:
                body_id = scalar_body_names.index(body) + nb
                dim = 1
            else:
                raise Exception(f"{body} not found")

            col_indices.append(body_id)
            G_rsi_dict[(row, body_id)] = cumulative_pos
            cumulative_pos += dim * row_sizes[row]
            counter += 1

        row_ptr.append(counter)

    G_sparsity_pattern = (
        cumulative_pos,
        col_indices,
        row_ptr,
        row_sizes,
        col_sizes,
        G_rsi_dict,
    )
    return G_sparsity_pattern

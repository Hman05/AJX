import numpy as np
from ajx.constraints import Constraint
from ajx.definitions import RigidBody
from typing import Tuple


def get_schur_fillin_sparsity(constraint_list: Tuple[Constraint], lower: bool = True):
    constraint_node = [(i, c.body_a, c.body_b) for i, c in enumerate(constraint_list)]

    edges_directed_map = {}
    for i, a, b in constraint_node:
        edges_directed_map[i] = []
        for j, c, d in constraint_node:
            if i == j:
                continue
            if i > j and lower:
                continue
            if a == c or b == c or a == d or b == d:
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
    block_sizes = [c.dof for c in constraint_list]
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
    constraint_list: Tuple[Constraint],
    rigid_body_names: Tuple[str],
):

    nc = len(constraint_list)
    nb = len(rigid_body_list)
    # Pre-Step 4: Compute reduction scattering indexation and allocate data for G
    row_sizes = np.array([c.dof for c in constraint_list])
    col_sizes = np.ones(nb, dtype=int) * 6

    G_rsi_dict = {}
    col_indices = []
    row_ptr = [0]
    cumulative_pos = 0
    counter = 0
    for row in range(nc):
        constraint = constraint_list[row]
        if constraint.body_a in rigid_body_names:
            body_a_id = rigid_body_names.index(constraint.body_a)
        else:
            body_a_id = None
        if constraint.body_b in rigid_body_names:
            body_b_id = rigid_body_names.index(constraint.body_b)
        else:
            body_b_id = None
        if not constraint.is_attached_to_world:
            col_indices.append(body_a_id)
            col_indices.append(body_b_id)
            G_rsi_dict[(row, body_a_id)] = cumulative_pos
            G_rsi_dict[(row, body_b_id)] = cumulative_pos + 6 * row_sizes[row]
            cumulative_pos += 12 * row_sizes[row]
            counter += 2
        else:
            col_indices.append(body_b_id)
            G_rsi_dict[(row, body_b_id)] = cumulative_pos
            cumulative_pos += 6 * row_sizes[row]
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

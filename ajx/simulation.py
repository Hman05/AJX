import jax.numpy as jnp
import numpy as np
import jax
import ajx.math as math

from ajx.definitions import (
    RigidBody,
    ScalarBody,
    State,
    Configuration,
    GeneralizedVelocity,
)
from ajx.pre_step_modifiers import PreStepModifier
from ajx.constraints import (
    TwoBodyConstraint,
    OneBodyConstraint,
    Constraint,
    TwoBodyShaftConstraint,
    GearConstraint,
)
from ajx.sensors import Sensor
from typing import Dict, List, Tuple, Optional
from loguru import logger

from ajx.block_sparse.csc_ldlt import ldlt_solve
from ajx.block_sparse.vbc_matrix import VBCMatrix
from ajx.block_sparse.vbr_matrix import VBRMatrix
from ajx.param import SimulationParameters
from ajx.definitions import RigidBodyParameters, ScalarBodyParameters

from flax import struct
from functools import partial
from jax import jit
from enum import Enum

from ajx.symbolic import (
    get_constraint_sparsity,
    get_schur_fillin_sparsity,
)
from flax import struct


class Solver(Enum):
    DENSE_LINEAR = 1
    SPARSE_LINEAR = 2


@struct.dataclass
class SimulationSettings:
    timestep: float
    use_gyroscopic: bool = False
    solver: Solver = Solver.DENSE_LINEAR


@struct.dataclass
class Simulation:
    """
    A class to represent the physics simulation.

    Attributes
    ----------
    timestep: float
        The size of the timestep in seconds.
    rigid_body_list: Tuple[RigidBody]
        A list of the rigid bodies that are present in the simulation.
    constraint_list: Tuple[Constraint]
        A list of the constraints that are present in the simulation.
    pre_step_modifiers: Tuple[PreStepModifier]
        A list of modifier objects (friction models, motors, symmetric inertia, etc.).
    use_gyroscopic: bool
        If false, the force calculation will exclude the fictisious forces due to configuration dependent interia.

    Methods
    -------
    pre_step:
        Returns the velocity update, multipliers and solver state given the system's
        state, parameters and control signal.
    post_step:
        Update the state using the given velocity update.
    inverse_dynamics:
        Solve for the forces to step from one state to a target state.
    """

    settings: SimulationSettings
    rigid_body_list: Tuple[RigidBody]
    constraint_list: Tuple[Constraint]
    sensor_list: Tuple[Sensor] = struct.field(default_factory=lambda: ())
    pre_step_modifiers: Tuple[PreStepModifier] = struct.field(
        default_factory=lambda: ()
    )
    scalar_body_list: Tuple[ScalarBody] = struct.field(default_factory=lambda: ())

    @property
    def h_inv(self):
        return 1 / self.settings.timestep

    @property
    def h(self):
        return self.settings.timestep

    @partial(jit, static_argnums=0)
    def pre_step(
        self, state: State, u: jax.Array, param: SimulationParameters
    ) -> Tuple[Dict[str, jax.Array], jax.Array, int]:
        """
        Returns the velocity update, multipliers and solver state given the system's
        state, parameters and control signal.

        Parameters
        ----------
        state: State
            The system's state.
        u: jax.Array
            The current control signal.
        param: SimulationParameters
            The system's parameters.

        Returns:
        -------
        Dict[str, jax.Array]:
            The velocity update.
        jax.Array:
            The multipliers.
        int:
            Solver state.
        """
        for component in self.pre_step_modifiers:
            state, param = component.update_params(state, u, param)
        f_ext = self._gravity_gyro_force3D(state, param)

        return state, self._force_solver(state, f_ext, param)

    @partial(jit, static_argnums=0)
    def post_step(self, state: State, gvel_next: GeneralizedVelocity) -> State:
        """
        Update the state using the given velocity update.

        Parameters
        ----------
        state: State
            The system's state.
        gvel_next: GeneralizedVelocity
            The velocity update
            TODO: Should be replaced with the full result from pre_step (intermediates)

        Returns:
        -------
        State:
            The updated state.
        """

        def body_step(gvel_next_data, conf_pos, conf_rot):
            vel_next = gvel_next_data[:3]
            ang_next = gvel_next_data[3:]
            pos_next = conf_pos + self.h * vel_next
            delta_rot = jit(math.from_rotation_vector)(ang_next * self.h)

            rot_next = jit(math.quat_mul)(delta_rot, conf_rot)
            rot_next = jit(math.normalize)(rot_next)

            return pos_next, rot_next

        pos_next, rot_next = jax.vmap(body_step)(
            gvel_next.data, state.conf.pos, state.conf.rot
        )
        scalar_next = state.conf.scalar + self.h * gvel_next.scalar
        conf_next = Configuration(
            pos=pos_next,
            rot=rot_next,
            scalar=scalar_next,
        )
        state_next = state.replace(
            conf=conf_next,
            gvel=gvel_next,
        )

        return state_next

    def observe(
        self, state: State, gvel_next: GeneralizedVelocity, param: SimulationParameters
    ) -> jax.Array:
        """
        Observe the state using the simulation sensors

        Parameters
        ----------
        state: State
            The system's state.
        gvel_next: GeneralizedVelocity
            The velocity update
            TODO:  Should be replaced with the full result from pre_step (intermediates)

        Returns:
        -------
        jax.Array:
            An flat array containing all observations in the same order as the simulation sensors.
        """

        observation_list = [jnp.zeros([0])]
        for sensor in self.sensor_list:
            observation = sensor.observe(state, gvel_next, param)
            observation_list.append(observation)

        return jnp.concatenate(observation_list)

    @partial(jit, static_argnums=0)
    def inverse_dynamics(
        self,
        state: State,
        gvel_target: GeneralizedVelocity,
        action: jax.Array,
        param: SimulationParameters,
    ) -> jax.Array:
        """
        Solve for the force(s) required to step from one state to a target velocity.

        Parameters
        ----------
        state: State
            The system's state.
        target_state: State
            The target state to be reached.
        param: SimulationParameters
            The system's parameters stored as a jax pytree with a dictionary at the top level.

        Returns:
        -------
        jax.Array:
            An array containing the required force(s).
        """
        for component in self.pre_step_modifiers:
            state, param = component.update_params(state, action, param)

        f_ext = self._gravity_gyro_force3D(state, param)
        M_stacked, _, G, Sigma_data, b_data = self._assemble_blocks(state, param)
        G_dense = G.to_scalar_matrix()
        # M = jax.scipy.linalg.block_diag(*M_stacked)

        # lbda = 1 / Sigma_data * (b_data - G.mul_vector(qdot_target.data.flatten()))
        lbda = 1 / Sigma_data * (b_data - G_dense @ gvel_target.data.flatten())

        M_vdelta = jax.vmap(jnp.matmul)(
            M_stacked, gvel_target.data - state.gvel.data
        ).flatten()
        # p_ext = M_vdelta - G.vector_mul(lbda)
        p_ext = M_vdelta - G_dense.T @ lbda

        delta = p_ext - f_ext.flatten() * self.h

        return delta

    def effective_mass(
        self,
        state: State,
        action: jax.Array,
        param: Dict,
    ):
        """
        Compute the effective mass of the system at a given state. The effective mass
            M_e = M + G.T @ Sigma^(-1) G
        where M is the mass marix, G is the combined constraint Jacobian, and Sigma
        is the combined regularization.

        Parameters
        ----------
        state: State
            The system's state.
        action: jax.Array
            The current action.
        param: SimulationParameters
            The system's parameters stored as a jax pytree with a dictionary at the top level.

        Returns:
        -------
        jax.Array:
            A 2D array containing the effective mass M_e.
        """
        for component in self.pre_step_modifiers:
            state, param = component.update_params(state, action, param)

        M_stacked, M_inv_stacked, G, Sigma_data, b_data = self._assemble_blocks(
            state, param
        )
        G_dense = G.to_scalar_matrix()
        M = jax.scipy.linalg.block_diag(*M_stacked)
        M_Sigma = M + G_dense.T @ jnp.diag(1 / Sigma_data) @ G_dense
        return M_Sigma

    @partial(jit, static_argnums=0)
    def _gravity_gyro_force3D(self, state, param):
        """Compute the external force to apply to each rigid body"""
        g = param.gravity

        def force_per_body(rb_param, conf_rot, gvel_data):
            mass = rb_param.mass
            gyroscopic_torque = jnp.array([0.0, 0.0, 0.0])
            ang = gvel_data[3:]
            if self.settings.use_gyroscopic:
                rotation = math.rotation_matrix(conf_rot)
                inertia = rb_param.get_inertia_matrix()
                world_inertia = rotation @ inertia @ rotation.T
                gyroscopic_torque = -jnp.cross(ang, world_inertia @ ang)
            force_ext = g * mass
            # This force terms should always be zero and are only used to get derivative information
            mc = math.rotate_vector(conf_rot, rb_param.mc)
            gyroscopic_linear_force = -jnp.cross(ang, jnp.cross(ang, mc))
            torque_ext = jnp.cross(mc, g)

            genelarized_force = jnp.concatenate(
                [force_ext + gyroscopic_linear_force, torque_ext + gyroscopic_torque]
            )
            return genelarized_force

        gforces = jax.vmap(force_per_body)(
            param.rigid_body_param, state.conf.rot, state.gvel.data
        )
        scalar_body_forces = jnp.zeros_like(state.conf.scalar)
        combined_gforces = jnp.concatenate([gforces.flatten(), scalar_body_forces])

        return combined_gforces

    @partial(jit, static_argnums=0)
    def _force_solver(
        self, state: State, f_ext: jax.Array, param: SimulationParameters
    ) -> Tuple[Tuple[State, jax.Array], int]:

        logger.trace("Tracing force_solver")
        gvel = state.gvel.flatten()

        M_stacked, M_inv_stacked, G, Sigma_data, b_data = self._assemble_blocks(
            state, param
        )
        n_rb_dof = 6 * state.gvel.data.shape[0]

        if self.settings.solver == Solver.SPARSE_LINEAR:
            S_sparse, rhs, M_inv_f, rsi_dict = self._assemble_schur(
                G,
                M_inv_stacked.flatten(),
                Sigma_data,
                f_ext.flatten(),
                gvel,
                b_data,
                lower=True,
            )
            rsi_list = tuple(rsi_dict.values())

            lbda = ldlt_solve(S_sparse, rsi_list, rhs)
            GTlbda = G.vector_mul(lbda)
        elif self.settings.solver == Solver.DENSE_LINEAR:
            M_inv = jax.scipy.linalg.block_diag(*M_inv_stacked[0], *M_inv_stacked[1])
            G_dense = G.to_scalar_matrix()
            S = G_dense @ M_inv @ G_dense.T + jnp.diag(Sigma_data)
            M_inv_f = M_inv @ f_ext.flatten()
            rhs = b_data - G_dense @ gvel - self.h * G_dense @ M_inv_f
            lbda = jax.scipy.linalg.solve(S, rhs)
            GTlbda = G_dense.T @ lbda
        else:
            raise NotImplementedError
        GTlbda6dof = GTlbda[:n_rb_dof]
        GTlbda1dof = GTlbda[n_rb_dof:]
        constraint_imp_6dof = jax.vmap(jnp.matmul)(
            M_inv_stacked[0], GTlbda6dof.reshape(-1, 6)
        )
        constraint_imp_scalar = jax.vmap(jnp.matmul)(
            M_inv_stacked[1], GTlbda1dof.reshape(-1, 1)
        )
        constraint_imp = jnp.concatenate(
            [constraint_imp_6dof.flatten(), constraint_imp_scalar.flatten()]
        )
        qdot_next = state.gvel.flatten() + constraint_imp + self.h * M_inv_f
        code = 0
        gvel_next = GeneralizedVelocity(
            qdot_next[:n_rb_dof].reshape(-1, 6), qdot_next[n_rb_dof:]
        )
        return (gvel_next, lbda), code

    @partial(jit, static_argnums=0)
    def _assemble_mass_matrix(
        self, state: State, param: SimulationParameters
    ) -> Tuple[jax.Array, jax.Array]:
        def assemble_mass_block(
            rb_param: RigidBodyParameters, rot: jax.Array
        ) -> Tuple[jax.Array, jax.Array]:
            m = rb_param.mass
            J = rb_param.get_inertia_matrix()
            R = jit(math.rotation_matrix)(rot)
            mc = R @ rb_param.mc
            inertia_block = R @ J @ R.T

            mass_block = jnp.diag(jnp.array([m, m, m]))
            mc_skew = math.skew(mc)

            M = jnp.block([[mass_block, -mc_skew], [mc_skew, inertia_block]])
            return M, jnp.linalg.inv(M)

        M_stack, M_inv_stack = jax.vmap(assemble_mass_block)(
            param.rigid_body_param, state.conf.rot
        )
        M_scalar = param.scalar_body_param.inertia[:, None, None]
        M_scalar_inv = 1 / param.scalar_body_param.inertia[:, None, None]
        return (M_stack, M_scalar), (M_inv_stack, M_scalar_inv)

    @partial(jit, static_argnums=0)
    def _assemble_blocks(
        self,
        state: State,
        param: SimulationParameters,
    ) -> Tuple[jax.Array, jax.Array, VBRMatrix, jax.Array, jax.Array]:
        """
        Assemble the blocks in the saddle point system

        | M   -G.T  ||  v   |   | a |
        |           ||      |   |   |
        | G   Sigma || lbda | = | b |

        and return M, M_inv, G, Sigma, b
        """
        # Sort constraint_list into multiple lists where the number of dofs are the same
        # Group constraints based on their structure. If consecutive constraints have the same structure, they are grouped.
        # Constraint groups are evaluated in parallel. Note
        constraint_group_list = []
        group = []
        first_id = 0
        previous_identifier = None
        for i, constraint in enumerate(self.constraint_list):
            # To be expanded
            identifier = constraint.__class__.__name__
            if not identifier == previous_identifier:
                if previous_identifier:
                    constraint_group_list.append((previous_identifier, first_id, group))
                group = []
                first_id = i
            group.append((constraint))
            previous_identifier = identifier
        if self.constraint_list:
            constraint_group_list.append((identifier, first_id, group))

        # Use constraint graph to compute sparsity information
        (
            G_size,
            G_col_indices,
            G_row_ptr,
            G_row_sizes,
            G_col_sizes,
            G_rsi,
        ) = get_constraint_sparsity(
            self.rigid_body_list,
            self.scalar_body_list,
            self.constraint_list,
            param.rigid_body_param.names,
            param.scalar_body_param.names,
        )
        G_rsi_list = list(G_rsi.values())
        G_row_indices = np.cumsum(G_row_sizes) - G_row_sizes
        G_data = jnp.zeros(G_size)
        Sigma_data = jnp.zeros(sum(G_row_sizes))
        b_data = jnp.zeros(sum(G_row_sizes))
        row_slice_begin = 0

        # Assemble mass matrices
        M, M_inv = self._assemble_mass_matrix(state, param)

        # Loop over each constraint group. If the number of groups are kept low, the
        # fact that a regular for-loop is used should not matter much
        for identifier, first_index, constraint_group in constraint_group_list:
            # These dicts are required if we want to replace the for-loop above with a jax.lax loop
            n_bodies = {
                "TwoBodyShaftConstraint": TwoBodyShaftConstraint.get_num_bodies(),
                "TwoBodyConstraint": TwoBodyConstraint.get_num_bodies(),
                "OneBodyConstraint": OneBodyConstraint.get_num_bodies(),
                "GearConstraint": GearConstraint.get_num_bodies(),
            }
            n_constraint_param_objects = {
                "TwoBodyShaftConstraint": 2,
                "TwoBodyConstraint": 1,
                "OneBodyConstraint": 1,
                "GearConstraint": 1,
            }
            c_func = {
                "TwoBodyShaftConstraint": TwoBodyShaftConstraint.func,
                "TwoBodyConstraint": TwoBodyConstraint.func,
                "OneBodyConstraint": OneBodyConstraint.func,
                "GearConstraint": GearConstraint.func,
            }
            jac_func = {
                "TwoBodyShaftConstraint": TwoBodyShaftConstraint.jacobian,
                "TwoBodyConstraint": TwoBodyConstraint.jacobian,
                "OneBodyConstraint": OneBodyConstraint.jacobian,
                "GearConstraint": GearConstraint.jacobian,
            }
            operand_shapes = {
                "TwoBodyShaftConstraint": (6, 6, 1),
                "TwoBodyConstraint": (6, 6),
                "OneBodyConstraint": (6,),
                "GearConstraint": (1,),
            }
            constrained_degrees = {
                "TwoBodyShaftConstraint": 1,
                "TwoBodyConstraint": 6,
                "OneBodyConstraint": 6,
                "GearConstraint": 1,
            }
            constraint_param_dict = {
                "TwoBodyShaftConstraint": param.scalar_constraint_param,
                "TwoBodyConstraint": param.constraint_param,
                "OneBodyConstraint": param.constraint_param,
                "GearConstraint": param.scalar_constraint_param,
            }
            index_subtract = {
                "TwoBodyShaftConstraint": param.constraint_param.compliance.shape[0],
                "TwoBodyConstraint": 0,
                "OneBodyConstraint": 0,
                "GearConstraint": param.constraint_param.compliance.shape[0],
            }
            index_subtract2_dict = {
                "TwoBodyShaftConstraint": (
                    param.constraint_param.compliance.shape[0],
                    0,
                ),
                "TwoBodyConstraint": (0,),
                "OneBodyConstraint": (0,),
                "GearConstraint": (param.constraint_param.compliance.shape[0],),
            }
            index_subtract2 = index_subtract2_dict[identifier]

            # Get the body indices (integers) from body names (strings)
            rb_names_ext = (
                *param.rigid_body_param.names,
                *param.scalar_body_param.names,
            )
            body_ids = tuple(
                jnp.array(
                    [
                        rb_names_ext.index(constraint.bodies[i])
                        for constraint in constraint_group
                    ]
                )
                for i in range(n_bodies[identifier])
            )
            # Get the constraint indices (integers) from constraint names (strings)
            constraint_names_ext = (
                *param.constraint_param.names,
                *param.scalar_constraint_param.names,
            )
            constraint_ids = tuple(
                jnp.array(
                    [
                        constraint_names_ext.index(constraint.names[i])
                        - index_subtract2[i]
                        for constraint in constraint_group
                    ]
                )
                for i in range(n_constraint_param_objects[identifier])
            )
            # Stack constraint types as a jnp.array
            constraint_types = jnp.array(
                [constraint.constraint_type for constraint in constraint_group]
            )

            # Compute Jacobians and constraint offsets
            G_blocks = jax.vmap(jac_func[identifier], (None, None, 0, 0, 0))(
                param,
                state,
                body_ids,
                constraint_ids,
                constraint_types,
            )
            default_offsets = jax.vmap(c_func[identifier], (None, None, 0, 0, 0))(
                param,
                state,
                body_ids,
                constraint_ids,
                constraint_types,
            )

            # Copy the Jacobian data from the constraint group to the full Jacobian
            ptr = G_rsi_list[G_row_ptr[first_index]]
            row_slice_begin = G_row_indices[first_index]
            G_data = G_data.at[ptr : ptr + G_blocks.flatten().shape[0]].set(
                G_blocks.flatten()
            )

            # Get the index slice corresponding to this constraint group
            # This slice is used to constraint parameters
            first_index = first_index - index_subtract[identifier]
            group_size = len(constraint_group)
            idx_slice = slice(first_index, first_index + group_size)

            # Get the constraint parameters
            constraint_param = constraint_param_dict[identifier]
            tau = constraint_param.damping[idx_slice]
            epsilon = constraint_param.compliance[idx_slice]
            target = constraint_param.target[idx_slice]
            viscous_compliance = epsilon
            alpha = 1 / (1 + 4 * tau * self.h_inv)
            holonomic_regularization = 4 * epsilon * self.h_inv**2 * alpha
            nonholonomic_regularization = viscous_compliance * self.h_inv

            not_locked = constraint_param.is_velocity[idx_slice]
            is_locked = jnp.logical_not(not_locked)

            offsets = default_offsets - target

            regularization = (
                holonomic_regularization * is_locked
                + nonholonomic_regularization * not_locked
            )

            # Copy regularization of this constraint group to the full regularization vector
            row_slice_end = (
                row_slice_begin + constrained_degrees[identifier] * group_size
            )
            Sigma_data = Sigma_data.at[row_slice_begin:row_slice_end].set(
                regularization.flatten()
            )

            # Compute Jacobian times velocity (TODO: Consider abstraction in GeneralizedVelocity)
            # proj_vel = G @ u = G1 @ u + G2 @ u + ... = [G1x ux G1z uz].T +
            jnp_body_ids = jnp.stack(body_ids, axis=1)

            def stack_multiply(G_blocks, body_ids, gvel, gscalar):
                dims = operand_shapes[identifier]
                sizes = (
                    jnp.concatenate(
                        [
                            jnp.ones([gvel.shape[0]], dtype=jnp.int64) * 6,
                            jnp.ones([gscalar.shape[0]], dtype=jnp.int64),
                        ]
                    )
                    - 6
                )
                offsets = jnp.cumulative_sum(sizes, include_initial=True)
                gflat = jnp.concatenate([gvel.flatten(), gscalar])
                offset = 0
                results = []
                for i, dim in enumerate(dims):
                    cd = constrained_degrees[identifier]
                    G = G_blocks[offset * cd : (offset + dim) * cd].reshape(cd, dim)

                    body_id = body_ids[i]
                    vel_begin = offsets[body_id]
                    vel_dim = operand_shapes[identifier][i]
                    vel = jax.lax.dynamic_slice(gflat, (vel_begin,), (vel_dim,))
                    results.append(G @ vel)

                return jnp.stack(results)

            proj_vels = jax.vmap(stack_multiply, in_axes=(0, 0, None, None))(
                G_blocks, jnp_body_ids, state.gvel.data, state.gvel.scalar
            )
            proj_vel = jnp.sum(proj_vels, axis=1)

            # Compute the rhs vector
            rhs_holonomic = -4 * self.h_inv * alpha * offsets + alpha * proj_vel
            rhs_velocity = target
            rhs = rhs_holonomic * is_locked + rhs_velocity * not_locked

            # Copy rhs of this constraint group to the full rhs vector
            b_data = b_data.at[row_slice_begin:row_slice_end].set(rhs.flatten())

        G = VBRMatrix(G_data, G_col_indices, G_row_ptr, G_row_sizes, G_col_sizes)

        return M, M_inv, G, Sigma_data, b_data

    def _assemble_schur(
        self,
        G,
        M_inv,
        Sigma_data,
        f_ext,
        gvel,
        b_data,
        lower=True,
    ):
        """Form S = G @ M_inv @ G.T + Sigma as a VBCMatrix"""
        (
            schur_size,
            row_indices,
            col_ptr,
            block_sizes,
            block_sizes,
            rsi_dict,
        ) = get_schur_fillin_sparsity(
            self.constraint_list,
            lower,
        )
        S_data = jnp.zeros(schur_size)

        # Want to form S = G @ M_inv @ G.T + Sigma
        # Result is formed by "M_inv weighted block-dot products" of combinations of rows of G

        slice_begin1 = 0
        sigma_slice_begin = 0
        for i in range(len(G.row_ptr) - 1):
            r1_slice = (G.row_ptr[i], G.row_ptr[i + 1])
            cols1 = G.col_indices[r1_slice[0] : r1_slice[1]]
            slice_end1 = slice_begin1 + sum(
                G.col_sizes[c] * G.row_sizes[i] for c in cols1
            )
            data1 = G.data[slice_begin1:slice_end1]
            slice_begin2 = 0
            sigma_slice_end = sigma_slice_begin + block_sizes[i]
            for j in range(len(G.row_ptr) - 1):
                r2_slice = (G.row_ptr[j], G.row_ptr[j + 1])
                cols2 = G.col_indices[r2_slice[0] : r2_slice[1]]
                slice_end2 = slice_begin2 + sum(
                    G.col_sizes[c] * G.row_sizes[j] for c in cols2
                )
                if (i, j) in rsi_dict:
                    data2 = G.data[slice_begin2:slice_end2]
                    res, intersection_empty = _bdot_over_intersection(
                        cols1,
                        data1,
                        cols2,
                        data2,
                        M_inv,
                        G.row_sizes[i],
                        G.row_sizes[j],
                        G.col_sizes,
                    )
                    if i == j:
                        res = res + jnp.diag(
                            Sigma_data[sigma_slice_begin:sigma_slice_end]
                        )
                        pass
                    if not intersection_empty:
                        rsi_slice_begin = rsi_dict[(i, j)]
                        rsi_slice_end = rsi_slice_begin + res.shape[0] * res.shape[1]
                        S_data = S_data.at[rsi_slice_begin:rsi_slice_end].set(
                            res.flatten()
                        )
                        # S_data[rsi_slice_begin:rsi_slice_end] = res.flatten()
                slice_begin2 = slice_end2
            sigma_slice_begin = sigma_slice_end
            slice_begin1 = slice_end1
        S = VBCMatrix(S_data, row_indices, col_ptr, block_sizes, block_sizes)

        M_inv_slice_begin = 0
        f_slice_begin = 0
        M_inv_f = jnp.zeros_like(f_ext)
        body_sizes = np.ones([len(self.rigid_body_list)], dtype=int) * 6
        for i, size in enumerate(body_sizes):
            M_inv_slice_end = M_inv_slice_begin + size**2
            f_slice_end = f_slice_begin + size
            M_inv_block = M_inv[M_inv_slice_begin:M_inv_slice_end].reshape(size, size)
            f_block = f_ext[f_slice_begin:f_slice_end]
            result_flat = (M_inv_block @ f_block).flatten()
            M_inv_f = M_inv_f.at[f_slice_begin:f_slice_end].set(result_flat)
            M_inv_slice_begin = M_inv_slice_begin
            f_slice_begin = f_slice_end

        M_inv_a = gvel - self.h * M_inv_f
        b_star = b_data - G.mul_vector(M_inv_a)
        return S, b_star, M_inv_f, rsi_dict


import numpy as np


def _bdot_over_intersection(
    a_indices,
    a_values,
    b_indices,
    b_values,
    c_values,
    row_size1,
    row_size2,
    col_sizes,
):
    res = jnp.zeros([row_size1, row_size2])
    intersection_empty = True
    m = len(a_indices)
    n = len(b_indices)
    slices_begin3 = np.cumsum(np.array(col_sizes) ** 2) - col_sizes[0] ** 2
    slices_end3 = np.cumsum(np.array(col_sizes) ** 2)

    slice_begin1 = 0
    slice_begin2 = 0
    i, j = 0, 0
    while i < m and j < n:
        slice_end1 = slice_begin1 + row_size1 * col_sizes[i]
        slice_end2 = slice_begin2 + row_size2 * col_sizes[j]

        if a_indices[i] == b_indices[j]:
            reduce_size = col_sizes[a_indices[i]]
            slice_begin3 = slices_begin3[a_indices[i]]
            slice_end3 = slices_end3[a_indices[i]]
            A = a_values[slice_begin1:slice_end1].reshape(row_size1, reduce_size)
            B = b_values[slice_begin2:slice_end2].reshape(row_size2, reduce_size)
            C = c_values[slice_begin3:slice_end3].reshape(reduce_size, reduce_size)
            res_value = A @ C @ B.T
            res += res_value
            intersection_empty = False
        ii = i
        jj = j
        if a_indices[i] <= b_indices[j]:
            ii = i + 1
            slice_begin1 = slice_end1
        if a_indices[i] >= b_indices[j]:
            jj = j + 1
            slice_begin2 = slice_end2
        i = ii
        j = jj

    return res, intersection_empty

import unittest

from ajx.example_environments import Pendulum
from ajx.simulation import SimulationSettings, Solver
import numpy as np


class TestSolversCompileAndRun(unittest.TestCase):

    def test_dense_linear_solver(self):

        # To create the pendulum environment, set default parameters, and get an initial state
        env = Pendulum(SimulationSettings(timestep=0.01, solver=Solver.DENSE_LINEAR), has_quadratic_damping=False)
        env_param = env.default_param.tree_replace({})
        state = env.state_from_angle(1.0, env_param)

        # To check that we can run 3 steps without error
        for i in range(3):
            state, observations = env.step(state, None, env_param)

    def test_sparse_linear_solver(self):

        # To create the pendulum environment, set default parameters, and get an initial state
        env = Pendulum(SimulationSettings(timestep=0.01, solver=Solver.SPARSE_LINEAR), has_quadratic_damping=False)
        env_param = env.default_param.tree_replace({})
        state = env.state_from_angle(1.0, env_param)

        # To check that we can run 3 steps without error
        for i in range(3):
            state, observations = env.step(state, None, env_param)

    def test_dense_pgs_solver(self):

        # To create the pendulum environment, set default parameters, and get an initial state
        env = Pendulum(SimulationSettings(timestep=0.01, solver=Solver.DENSE_PGS), has_quadratic_damping=False)
        env_param = env.default_param.tree_replace({})
        state = env.state_from_angle(1.0, env_param)

        # To check that we can run 3 steps without error
        for i in range(3):
            state, observations = env.step(state, None, env_param)

    def test_sparse_pgs_solver(self):

        # To create the pendulum environment, set default parameters, and get an initial state
        env = Pendulum(SimulationSettings(timestep=0.01, solver=Solver.SPARSE_PGS), has_quadratic_damping=False)
        env_param = env.default_param.tree_replace({})
        state = env.state_from_angle(1.0, env_param)

        # To check that we can run 3 steps without error
        for i in range(3):
            state, observations = env.step(state, None, env_param)

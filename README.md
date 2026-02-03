# AJX
AJX is a minimal multibody dynamics simulator written in Python and built on top of the library JAX. AJX exposes the simulator step (forward dynamics) and inverse dynamics as JAX-composable functions, enabling GPU parallelization and derivative transformations.

AJX represents each rigid body using maximal coordinates and computes constraint forces implicitly via the SPOOK stepper. A core design principle of AJX is the separation of state and logic. The goal is to maintain a functional programming style that mirrors standard mathematical notation. Accordingly, a rigid body object is distinct from its parameters and its state (position and velocity).
This design makes data layout explicit and ensures contiguous storage, which is critical for memory efficiency, performance, and JAX transformations.

The intention is to keep the core library small.

## High level project structure
This repository consists of five top-level directories which aim to be as independent as possible.
- **`ajx`** contains the actual physics library. It is supposed to be fully self-contained.
- **`scenes`** contains executable scripts to explore environments interactively in real-time. Based on the [panda3D](https://www.panda3d.org/) library.
- **`environments`** contains simulation environments

Files inside **`evaluation`** and **`util`** should eventually be moved.

## Getting Started
This is a Python project and should work with any recent Python version.
The following Python packages have to be installed to run the files
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [jax](https://jax.readthedocs.io/en/latest/index.html)
- [flax](https://flax.readthedocs.io/en/v0.5.3/index.html) (should ideally remove this dependency)
- [loguru](https://loguru.readthedocs.io/en/stable/)
- [panda3D](https://www.panda3d.org/) (is used in **`scenes`**)

## Running the project
There are several executable scripts located in various places. It is important to set the **PYTHONPATH** environment variable to the workspace directory to avoid import errors.

Almost all executable scripts are located in the **`experiments`** directory. 
The only exceptions are the scripts in the **`scenes`** directory. All scripts found under **`experiments`** are explained below.

- **`enviroments`** is meant to be a library of multibody systems and the files are not meant to be executed.
- **`simulate`** contains scripts to simulate various environments, store the artificial data, and explore the results.

## Parameters
Contiguous buffer (1D array of floats/doubles) for rigid body parameters (mass, mc, inertia), contiguous buffer (1D array of doubles) for constraint parameters (frames, regularization, targets), and dict (fixed pytree) for \texttt{sparse data}.

## State 
The state of the simulation is given by the configuration and generalized velocity of all rigid bodies. There is currently no support for extending the state with custom variables. The state variables are stored contiguously and grouped the way they are processed. The three groups are generalized velocity, position, and rotation (quaternions).

## Constraints
Instead of separate velocity constraints and holonomic constraints, AJX handles them together
- Constraints between two bodies always restrict 6 degrees of freedom. 
- (For now, may be useful to add 1 dof restrictions.)
- May be a mix of holonomic and velocity sub-constraints
This system has the following benefits
- Trivial indexation when all constraints are of the same size.
- Parallelization
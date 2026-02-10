# AJX
AJX is a minimal multibody dynamics simulator written in Python and built on top of the library JAX. AJX exposes the simulator step (forward dynamics) and inverse dynamics as JAX-composable functions, enabling GPU parallelization and derivative transformations.

AJX represents each rigid body using maximal coordinates and computes constraint forces implicitly via the SPOOK stepper. A core design principle of AJX is the separation of state and logic. The goal is to maintain a somewhat functional programming style that mirrors standard mathematical notation. Accordingly, a rigid body object is distinct from its parameters and its state (position and velocity).
This design makes data layout explicit and ensures contiguous storage, which is critical for memory efficiency, performance, and JAX transformations.

The intention is to keep the core library small.

## High level project structure
This repository consists of five top-level directories which aim to be as independent as possible.
- **`ajx`** contains the actual physics library. It is supposed to be fully self-contained.
- **`scenes`** contains executable scripts to explore environments interactively in real-time. Based on the [panda3D](https://www.panda3d.org/) library.
- **`environments`** contains simulation environments
- **`notebooks`** contains example Jupyter notebooks
- **`util`** contains example Jupyter notebooks

Files inside **`evaluation`** and **`util`** should eventually be moved.

## Getting Started
This is a Python project and should work with any recent Python version.
The following Python packages have to be installed to run the files
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [jax](https://jax.readthedocs.io/en/latest/index.html)
- [flax](https://flax.readthedocs.io/en/v0.5.3/index.html)
- [loguru](https://loguru.readthedocs.io/en/stable/)
- [panda3D](https://www.panda3d.org/) (used in **`scenes`**)
- [matplotlib](https://matplotlib.org/) (used in **`notebooks`**)

## Running the project
In this repository, there are executable scripts **`scenes`** and Jupyter notebooks under **`notebooks`**. To run the scripts under **`scenes`**, it
is important to set the **PYTHONPATH** environment variable to the workspace directory to avoid import errors.
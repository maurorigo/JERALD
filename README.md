# JERALD: high-fidelity dark matter, stellar mass and neutral hydrogen maps from fast N-body simulations

* [**Installation**](#installation)
* [**Testing**](#testing)
* [**Documentation**](#documentation)

JERALD is a code, based on the [Lagrangian Deep Learning](https://arxiv.org/abs/2010.02926) idea of Dai and Seljak, to paint baryons and related quantities on top of N-body only simulations with machine learning.

The package is based on [JAX](https://github.com/google/jax) and [mpi4jax](https://github.com/mpi4jax/mpi4jax/tree/master), and it uses MPI to implement parallel Cloud-In-Cell (CIC) painting and interpolation algorithms as well as parallel forward and backward Fourier transforms via [FFTW](https://www.fftw.org/).

## Installation
To install the package, clone the repo and run:
```
pip install .
```
On the C++ side, installation requires FFTW with MPI support (which should generally be available on clusters). The installation requires ```PkgConfig``` to link FFTW (which should also be available on clusters). In the future I may provide a simple ```FindFFTW.cmake``` file to allow compilation also when ```PkgConfig``` is missing.

On the Python side installation requires JAX and mpi4jax (see the related READMEs for installation), as well as [pybind11](https://github.com/pybind/pybind11) for binding C++ code to Python.
CUDA support isn't required, as for the time being the package only runs on CPU. In the future this may change.
On a cluster, I suggest loading any MPI modules and then installing mpi4jax via pip, to allow mpi4py to build on the existing MPI installation.

## Testing
A simple test that computes the model loss and its derivative with respect to its parameters (checked against the original LDL code) is available in ```losstest.py```. To run it with a single MPI process, execute
```
python losstest.py
```
Otherwise, to run it with ```Nproc``` processes, use
```
mpirun -n Nproc python losstest.py
```

## Documentation
For now, the documentation for each method is in the code (and it should be pretty complete).


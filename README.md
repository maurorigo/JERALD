# jaxLDL: JAX implementation of Lagrangian Deep Learning

* [**Installation**](#installation)
* [**Testing**](#testing)
* [**Documentation**](#documentation)

jaxLDL is the [JAX](https://github.com/google/jax) implementation of the [code](https://github.com/biweidai/LDL) performing [Lagrangian Deep Learning](https://arxiv.org/abs/2010.02926), a machine learning method to paint baryons and related quantities on top of N-body only simulations.

The package is based on JAX and [mpi4jax](https://github.com/mpi4jax/mpi4jax/tree/master), and it uses MPI to implement parallel Cloud-In-Cell (CIC) painting and interpolation algorithms as well as parallel forward and backward Fourier transforms via [FFTW](https://www.fftw.org/).

## Installation
To install the package, create a folder named ```src``` first. Then the package can be built using
```
pip install .
```
On the C++ side, installation requires FFTW with MPI support (which should generally be available on clusters), while on the Python side it requires JAX and mpi4jax (see the related READMEs for installation), as well as [pybind11](https://github.com/pybind/pybind11) for binding C++ code to Python.
CUDA support isn't required, as for the time being the package only runs on CPU. In the future this may change.
On a cluster, I suggest loading any MPI modules and then installing mpi4jax via pip, to allow mpi4py to build on the existing MPI installation.

## Testing
A simple test that computes the LDL loss and its derivative with respect to the LDL parameters (checked against the original code) is available in ```losstest.py```. To run it with a single MPI process, execute
```
python losstest.py
```
Otherwise, to run it with ```Nproc``` processes, use
```
mpirun -n Nproc python losstest.py
```

## Documentation
For now, the documentation for each method is in the code (and it should be pretty complete).


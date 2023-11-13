import jax.numpy as jnp
import numpy as np
import jax
from functools import partial
from jax import jit, vmap
from mpi4py import MPI
import mpi4jax

import jaxpops_bind as jpo

class PMesh(object):
    """
    Class to deal with particle mesh things, with methods that can be differentiated using standard JAX, parallelized using MPI
    """
    
    def __init__(self, Nmesh, BoxSize, comm=MPI.COMM_WORLD):
        """
            Nmesh (int or tuple): int N specifying the grid size (same for x, y and z) or tuple (Nx, Ny, Nz)
            BoxSize (float or tuple): Size of the simulation box (same or different for x, y and z)
            comm: MPI communicator
        """
        if np.size(Nmesh)==1:
            self.Nmesh = jnp.array((Nmesh, Nmesh, Nmesh), dtype=int)
        else:
            self.Nmesh = jnp.array(Nmesh, dtype=int)

        if np.size(BoxSize)==1:
            self.BoxSize = jnp.array((BoxSize, BoxSize, BoxSize))
        else:
            self.BoxSize = jnp.array(BoxSize)

        self.comm = comm
        self.commrank = self.comm.Get_rank()
        self.commsize = self.comm.Get_size()

        # Compute fft partitioning of data (local x length and local start)
        self.localL, self.localS = jpo.buildplan(self.Nmesh, comm)
        self._Nmesh = (int(self.Nmesh[0]), int(self.Nmesh[1]), int(self.Nmesh[2])) # For kvals
        self.fftss = (int(self.localS), int(self.localS + self.localL)) # Start/stop for kvals x
        self.fftdims = (int(self.localL), int(self.Nmesh[1]), int(self.Nmesh[2]))
        
        # Values for painting/readout
        self.edges = self.computeEdges()
        self.paintdims = (int(self.edges[0, self.commrank, 1] - self.edges[0, self.commrank, 0]),
                int(self.edges[1, self.commrank, 1] - self.edges[1, self.commrank, 0]),
                int(self.edges[2, self.commrank, 1] - self.edges[2, self.commrank, 0]))
        
        # Useful for LDL transfer function (might do a void returning function later)
        self.kvals = self.compute_wavevectors()
        self.knorms = jnp.linalg.norm(self.kvals, axis=-1) # kvals is (Nx, Ny, Nz)


    def computeEdges(self):
        """
            Computes lower and upper limits along x, y, z for the local part of the field
        """
        lls, _ = mpi4jax.allgather(self.localL, comm=self.comm)
        lss, _ = mpi4jax.allgather(self.localS, comm=self.comm)
        edges = np.zeros((3, self.commsize, 2), dtype=int)
        edges[2, :, 1] = self.Nmesh[2]
        edges[1, :, 1] = self.Nmesh[2]
        for i in range(self.commsize):
            edges[0, i, 0] = lss[i]
            edges[0, i, 1] = lss[i] + lls[i]
        return jnp.array(edges)
    
    @partial(jit, static_argnums=(0, 3))
    def paint(self, pos, mass, lyidx=0):

        return jpo.ppaint(pos, mass, self.Nmesh, self.BoxSize, self.edges, self.paintdims, self.comm, lyidx)

    @partial(jit, static_argnums=(0, 5))
    def readout(self, pos, mesh, boxsize=None, comm=None, lyidx=0):

        if boxsize is None:
            boxsize = self.BoxSize

        if comm is None:
            comm = self.comm

        return jpo.preadout(pos, mesh, self.Nmesh, boxsize, self.edges, self.paintdims, comm, lyidx)

    @partial(jit, static_argnums=0)
    def r2c(self, localreal):
        """
        Real to complex, direct 3D Fourier transform

        Parameters:
            mesh (array-like): Values at mesh points (real)

        Returns:
            out (array-like): Fourier transform of input mesh (complex)
        """

        return jpo.pfft(localreal, self.fftdims) / jnp.prod(self.Nmesh)

    @partial(jit, static_argnums=0)
    def c2r(self, localcplx):
        """
        Complex to real, inverse 3D Fourier transform

        Parameters:
            mesh (array-like): Values at mesh points (complex)

        Returns:
            out (array-like): Inverse Fourier transform of input mesh (real)
        """
        return jpo.pifft(localcplx, self.fftdims)

    def mesh_coordinates(self):
        """
        Computes grid coordinates as a (Nx, Ny, Nz, 3) array
        !!! OUTDATED !!!
        """
        x = jnp.arange(self.Nmesh[0]) / self.Nmesh[0] * self.BoxSize[0]
        y = jnp.arange(self.Nmesh[1]) / self.Nmesh[1] * self.BoxSize[1]
        z = jnp.arange(self.Nmesh[2]) / self.Nmesh[2] * self.BoxSize[2]
        X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
        out = jnp.stack((X, Y, Z), axis=-1)

        return out

    def flattened_coordinates(self):
        """
        Computes grid coordinates as a (Nx*Ny*Nz, 3) array
        !!! OUTDATED !!!
        """
        cellsz = self.BoxSize / self.Nmesh
        out = jnp.empty((self.Nmesh.prod(), 3))
        lenrange = jnp.arange(self.Nmesh.prod())
        out = out.at[:, 2].set(jnp.mod(lenrange, self.Nmesh[2]) * cellsz[2])
        out = out.at[:, 1].set(jnp.mod(jnp.floor(lenrange / self.Nmesh[2]), self.Nmesh[1]) * cellsz[1])
        out = out.at[:, 0].set(jnp.mod(jnp.floor(lenrange / self.Nmesh[2] / self.Nmesh[1]), self.Nmesh[0]) * cellsz[1])

        return out
    
    @partial(jit, static_argnums=0)
    def compute_wavevectors(self):
        """
        Returns the wavevectors associated to the mesh for this rank
        """
        kx = jnp.fft.fftfreq(self._Nmesh[0], self.BoxSize[0]/self.Nmesh[0]) * 2*np.pi
        ky = jnp.fft.fftfreq(self._Nmesh[1], self.BoxSize[1]/self.Nmesh[1]) * 2*np.pi
        kz = jnp.fft.fftfreq(self._Nmesh[2], self.BoxSize[2]/self.Nmesh[2]) * 2*np.pi
        kx = kx[self.fftss[0]:self.fftss[1]] # Local freqs
        kz = kz[:int(self._Nmesh[2]/2)+1] # Symmetry of FFT

        Kx, Ky, Kz = jnp.meshgrid(kx, ky, kz, indexing='ij')
        return jnp.stack((Kx, Ky, Kz), axis=-1)


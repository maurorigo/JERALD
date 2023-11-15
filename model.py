import jax.numpy as jnp
import numpy as np
import jax
from functools import partial
from jax import jit, vmap, grad, value_and_grad, custom_vjp
from mpi4py import MPI
import mpi4jax


@custom_vjp
def c2f(obj):
    """
        This compensates the fact that we only have int(Nmesh[2]/2)+1 of the fft
        along z due to symmetry. The problem is not present on the forward pass,
        but when computing derivatives it becomes significant.
    """
    return obj

def c2f_fwd(obj):
    return obj, ()

def c2f_bwd(res, v):
    # Assuming the second dimension of the fft is the correct one,
    # but this can be done more general if needed
    # Don't ask me why exactly it's like this
    # The indices that aren't self conjugate need this treatment 
    sz = v.shape[1]
    v = v.at[:, :, 1:int(sz/2)].add(jnp.conj(v[:, :, 1:int(sz/2)]))
    return (v,)

c2f.defvjp(c2f_fwd, c2f_bwd)

@partial(custom_vjp, nondiff_argnums=(1,))
def allbcast(val, comm):
    """
        Pretty much the inverse of allreduce. When dealing with different parts of
        the field in different ranks, the backward pass has its own value of these
        variables (I guess?), this sets it back to correct.
        In the forward pass, all values should already be the same, so just return val
    """
    return val

def allbcast_fwd(val, comm):
    return val, ()

def allbcast_bwd(comm, res, v):
    v, _ = mpi4jax.allreduce(v, op=MPI.SUM, comm=comm)
    return (v,)

allbcast.defvjp(allbcast_fwd, allbcast_bwd)


class LDLModel(object):
    """
    Model for Lagrangian Deep Learning
    """

    def __init__(self, pm):
        """
            pm (PMesh object): PMesh that performs fft, paint, readout
        """
        self.pm = pm
        self.ready = False

    def potential_gradient(self, params, X):
        """
        Computes gradient of LDL potential following code from https://github.com/biweidai/LDL/tree/master

        Parameters:
            params (tuple): Parameters of LDL
            X (array-like): (N, 3) array representing particle positions, needed to compute density

        Returns:
            Gradient of LDL potential with shape (Nx, Ny, Nz, 3)
        """
        fact, gamma, kh, kl, n = params

        # Density
        delta = self.pm.paint(X, 1.) * fact # Normalized density rho/bar(rho)
        gamma = allbcast(gamma, self.pm.comm)
        delta = (delta+1e-8) ** gamma

        # Density in Fourier space
        deltak = self.pm.r2c(delta)
        
        # Green's operator in Fourier space
        kl = allbcast(kl, self.pm.comm)
        kh = allbcast(kh, self.pm.comm)
        n = allbcast(n, self.pm.comm)
        kvals = self.pm.kvals # (Nx, Ny, Nz, 3)
        knorms = self.pm.knorms # (Nx, Ny, Nz)
        # Using the next two lines instead of the two above makes compilation faster (and not give
        # some warnings) on small numbers of MPI ranks.
        #kvals = self.pm.compute_wavevectors()
        #knorms = jnp.linalg.norm(kvals, axis=-1) 
        knorms = jnp.where(knorms==0, 1e-8, knorms) # Correct norms that are null
        filterk = -jnp.exp(-knorms**2/kl**2) * jnp.exp(-kh**2/knorms**2) * knorms**n # Idk why - sign, but it's just a multiplicative factor
        filterk = c2f(filterk) # Compensates the fact that we only have ~half of the fft (for back pass)

        potk = deltak * filterk

        # Now we need to compute the spacial derivative of this object on each point
        # Note that we only have the values at some grid points, so we need to use a finite difference approach (see overleaf)
        # Compare the FT of this (convolution) with an order 2 finite difference formula (https://en.wikipedia.org/wiki/Numerical_differentiation)
        step = self.pm.BoxSize / self.pm.Nmesh
        coeff = 1 / (6 * step) * (8 * jnp.sin(kvals * step) - jnp.sin(2 * kvals * step)) # (Nx, Ny, Nz, 3)

        # This gives gradient at grid positions, but we need it at particle positions, so we interpolate (see overleaf)
        potgradk = -1j * coeff[:, :, :, 0] * potk # x component (using same array to save memory)
        potgradx = self.pm.c2r(potgradk)
        potgradk = -1j * coeff[:, :, :, 1] * potk # y component
        potgrady = self.pm.c2r(potgradk)
        potgradk = -1j * coeff[:, :, :, 2] * potk # z component
        potgradz = self.pm.c2r(potgradk)

        # Use CIC interpolation
        return jnp.stack((self.pm.readout(X, potgradx),
            self.pm.readout(X, potgrady),
            self.pm.readout(X, potgradz)), axis=-1)
    
    def displace(self, params, X, Nstep=1):
        """
        Displaces particles according to LDL model.

        Parameters:
            params (array-like): Parameters for LDL
            X: (array-like): Particle positions (divided into different ranks)
        
        Returns:
            Displaced particle positions
        """

        sz, _ = mpi4jax.allreduce(len(X), op=MPI.SUM, comm=self.pm.comm)
        fact = self.pm.Nmesh.prod() / sz # Normalization for density

        for i in range(Nstep):
            
            alpha = params[5*i]
            gamma = params[5*i+1]
            kh = params[5*i+2]
            kl = params[5*i+3]
            n = params[5*i+4]
            
            alpha = allbcast(alpha, self.pm.comm)
            X += alpha * self.potential_gradient((fact, gamma, kh, kl, n), X)
            
        return X

    @staticmethod
    @jit
    def ReLU(x):
        # Just ReLU in JAX
        return jnp.where(x >= 0, x, 0)

    @partial(jit, static_argnums=(0, 3, 4))
    def LDL(self, params, X, Nstep=1, baryon=False):

        sz, _ = mpi4jax.allreduce(len(X), op=MPI.SUM, comm=self.pm.comm)
        fact = self.pm.Nmesh.prod() / sz

        X = self.displace(params, X, Nstep=Nstep)

        # Paint particle overdensity field
        delta = self.pm.paint(X, 1.) * fact

        if baryon:
            mu = params[5*Nstep]
            b1 = params[5*Nstep+1]
            b0 = params[5*Nstep+2]
        
            # Field transformation
            mu = allbcast(mu, self.pm.comm)
            b1 = allbcast(b1, self.pm.comm)
            b0 = allbcast(b0, self.pm.comm)
            return self.ReLU(b1 * (delta+1e-8) ** mu + b0) # Definition of b0 is different from the paper
        else:
            return delta

    def set_loss_params(self, Nstep, baryon=False, n=1., index=1., field2=None, mask=None, L1=True):
        """
        Just sets some variables for the loss function

        Parameters:
            Nstep (int): Number of displacement steps
            baryon (bool): Baryonic observables or no
            n (float): Smoothing kernel exponent, see LDL paper
            index (float): Exponent for LDL map, see LDL paper
            field2 (array-like): Additional (Nx, Ny, Nz) map to be multiplied to LDL output for combined observables
            mask (array-like): (Nx, Ny, Nz) array of zeros and ones for selecting values for training
            L1 (bool): L1 or L2 loss
        """
        self.ready = True
        self.Nstep = Nstep
        self.baryon = baryon
        self.n = n
        self.index = index
        self.field2 = field2
        self.mask = mask
        self.L1 = L1

    @partial(jit, static_argnums=(0))
    def _loss(self, params, X, target):
        """
        LDL loss function

        Parameters:
            params (array-like): Parameters of LDL model
            X (array-like): (N, 3) array representing input particle positions
            target (array-like): (Nx, Ny, Nz) target map

        Returns:
            loss (float): LDL model loss
        """

        # Compute the LDL map
        F = self.LDL(params, X, Nstep=self.Nstep, baryon=self.baryon) ** self.index

        # Optionally multiply by second field (WILL MOVE TO DIFFERENT METHOD MAYBE)
        if self.field2 is not None:
            F = F * self.field2

        # Residue field
        residue = F - target

        #smooth the field
        knorms = self.pm.knorms
        smoothingkernel = jnp.where(knorms==0, 1., knorms**(-self.n) + 1.)
        residuek = self.pm.r2c(residue)
        residuek *= smoothingkernel
        residue = jnp.abs(self.pm.c2r(residuek))
        if self.mask is not None:
            residue *= self.mask
            Npixel = jnp.sum(self.mask)
        else:
            Npixel = residue.size

        # Collect losses from all processes
        if self.L1:
            loss = jnp.sum(residue)
        else:
            loss = jnp.sum(residue**2)

        loss, _ = mpi4jax.allreduce(loss, op=MPI.SUM, comm=self.pm.comm)
        Npixel, _ = mpi4jax.allreduce(Npixel, op=MPI.SUM, comm=self.pm.comm)
        return loss / Npixel

    def loss(self, params, X, target):
        """
        Loss with exception handling, see _loss for the parameters
        """
        if not self.ready:
            raise Exception("Cannot compute loss before setting its parameters.")

        return self._loss(params, X, target)

    def loss_gradient(self, params, X, target):
        """
        Gradient of loss function with exception handling, see _loss for the parameters
        """
        if not self.ready:
            raise Exception("Cannot compute loss gradient before setting loss parameters.")

        return grad(self._loss, argnums=0)(params, X, target)

    def loss_and_grad(self, params, X, target):
        """
        Loss and its gradient with exception handling, see _loss for the parameters
        """
        if not self.ready:
            raise Exception("Cannot compute loss gradient before setting loss parameters.")

        return value_and_grad(self._loss, argnums=0)(params, X, target)


class RMSprop(object):
    """
    RMSprop optimizer class
    """
    def __init__(self, Nparams, LR=0.01, beta=0.99):
        """
            Nparams (int): number of parameters to optimize for
            LR (float): learning rate
            beta (float): exponential decay rate of RMSprop
        """
        self.LR = LR
        self.beta = beta
        self.v = jnp.zeros(Nparams)
        self.itr = 1

    def step(self, params, gradient):
        """
        Computes a RMSprop update of the parameters

        Parameters:
            params (array-like): Parameters to optimize
            gradient (array-like): Gradient of loss with respect to parameters

        Returns:
            Updated parameters
        """
        self.v = self.beta * self.v + (1. - self.beta) * gradient**2
        vh = jnp.sqrt(self.v / (1. - self.beta**self.itr))
        self.itr += 1
        return params - self.LR * gradient / (vh + 1e-8)

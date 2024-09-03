import jax.numpy as jnp
import numpy as np
import jax
from functools import partial
from jax import jit, vmap, grad, value_and_grad, custom_vjp
from mpi4py import MPI
import mpi4jax
import gc

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
        Also works for a jnp array of parameters (recommended in order to guarantee correct order of ops)
    """
    return val

def allbcast_fwd(val, comm):
    return val, ()

def allbcast_bwd(comm, res, v):
    v, _ = mpi4jax.allreduce(v, op=MPI.SUM, comm=comm)
    return (v,)

allbcast.defvjp(allbcast_fwd, allbcast_bwd)

@jit
def ReLU(x, low=1e-12):
    # Just ReLU in jax
    return jnp.where(x > low, x, low)

@jit
def lReLU(x, alpha=.1):
    # Leaky ReLU in jax
    return jnp.where(x > 0, x, alpha*x)


@jax.tree_util.register_pytree_node_class
class LDLModel(object):
    """
    Model for Lagrangian Deep Learning
    """
    def __init__(self, X, target, pm, Nstep=1, baryon=False, n=1., index=1., L1=None, field2=None, masktrain=None, maskvalid=None):
        """
            pm (PMesh object): PMesh that performs fft, paint, readout
            X (array-like): (?, 3) array of local input positions
            target (array-like): (?, Ny, Nz) array of local target map
            Nstep (int): Number of displacement steps
            baryon (bool): Baryonic observables or no
            n (float): Smoothing kernel exponent, see LDL paper
            index (float): Exponent for LDL map, see LDL paper
            field2 (array-like): Additional (?, Ny, Nz) map to be multiplied to LDL output
            masktrain (array-like): (?, Ny, Nz) local array of 0s and 1s for selecting training pixels
            maskvalid (array-like): As above, for validation
            L1 (bool): L1 or L2 norm loss

            NOTE: initializing everything here to make compilation of class faster via pytrees,
            avoiding constant folding of huge arrays which can lead to massive slowdowns.
        """
        self.pm = pm
        self.X = X
        self.target = target
        # Useful later
        self.kvals = self.pm.compute_wavevectors()
        self.knorms = jnp.linalg.norm(self.kvals, axis=-1)

        # Loss parameters
        self.Nstep = Nstep
        self.baryon = baryon
        self.n = n
        self.index = index
        self.field2 = field2
        self.masktrain = masktrain
        self.maskvalid = maskvalid
        if L1 is None:
            self.L1 = baryon # Paper uses True for baryons and False otherwise
        else:
            self.L1 = L1

    @jit
    def potential_gradient(self, params, X):
        """
        Computes gradient of LDL potential following https://github.com/biweidai/LDL/tree/master

        Parameters:
            params (tuple): Parameters of LDL
            X (array-like): (N, 3) array representing particle positions

        Returns:
            Gradient of LDL potential with shape (Nx, Ny, Nz, 3)
        """
        fact, gamma, kh, kl, n = params

        # Density
        delta = self.pm.paint(X, 1.) * fact # Normalized density rho/bar(rho)
        delta = (delta+1e-8) ** gamma

        # Density in Fourier space
        deltak = self.pm.r2c(delta)
        
        # Green's operator in Fourier space
        knorms = jnp.where(self.knorms==0, 1e-8, self.knorms) # Correct norms that are null
        filterk = -jnp.exp(-knorms**2/kl**2) * jnp.exp(-kh**2/knorms**2) * knorms**n # Attractive so -
        filterk = c2f(filterk) # Compensates the fact that we only have ~half of the fft (for back pass)

        potk = deltak * filterk

        # Now we need to compute the spacial derivative of this object on each point
        # Note that we only have the values at some grid points, so we need to use a finite difference approach (see overleaf)
        # Compare the FT of this (convolution) with an order 2 finite difference formula (https://en.wikipedia.org/wiki/Numerical_differentiation)
        step = self.pm.BoxSize / self.pm.Nmesh
        #coeff = 1 / (6 * step) * (8 * jnp.sin(self.kvals * step) - jnp.sin(2 * self.kvals * step)) # (Nx, Ny, Nz, 3)
        coeff = 1 / (30 * step) * (45 * jnp.sin(self.kvals * step) - 9 * jnp.sin(2 * self.kvals * step) + jnp.sin(3 * self.kvals * step))

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

        """potgradk = -1j * coeff[:, :, :, 0] * potk # x component (using same array to save memory)
        potgradx = self.pm.c2r(potgradk)
        potgradk = -1j * coeff[:, :, :, 1] * potk # y component
        potgrady = self.pm.c2r(potgradk)
        potgradk = -1j * coeff[:, :, :, 2] * potk # z component
        potgradz = self.pm.c2r(potgradk)
        return self.pm.readout3D(X, jnp.stack((potgradx, potgrady, potgradz), axis=0))"""

    @partial(jit, static_argnums=3)
    def displace(self, params, X, Nstep):
        """
        Displaces particles according to LDL model.

        Parameters:
            params (array-like): Parameters for LDL
            X (array-like): Particle positions (divided into different ranks)
            Nstep (int): Number of displacement layers
        
        Returns:
            Displaced particle positions
        """

        sz, _ = mpi4jax.allreduce(len(X), op=MPI.SUM, comm=self.pm.comm)
        fact = self.pm.Nmesh.prod() / sz # Normalization for density

        def body(i, X):
            
            alpha = params[5*i]
            gamma = params[5*i+1]
            kh = params[5*i+2]
            kl = params[5*i+3]
            n = params[5*i+4]
            
            return X + alpha * self.potential_gradient((fact, gamma, kh, kl, n), X)
            
        # Using fori_loop because with normal for JAX can't compute the derivative wrt gamma
        # in case len(X) is not divisible by pm.comm.size and with Nstep=1 (but with 2 yes,
        # for some reason)
        return jax.lax.fori_loop(0, Nstep, body, X)

    @partial(jit, static_argnums=(3, 4))
    def LDL(self, params, X, Nstep, baryon):

        sz, _ = mpi4jax.allreduce(len(X), op=MPI.SUM, comm=self.pm.comm)
        fact = self.pm.Nmesh.prod() / sz

        X = self.displace(params, X, Nstep=Nstep)

        # Paint particle overdensity field
        delta = self.pm.paint(X, 1.) * fact
        
        if baryon:
            mu = params[5*Nstep+0]
            b1 = params[5*Nstep+1]
            b0 = params[5*Nstep+2]
            b2 = params[5*Nstep+3]
        
            # Field transformation
            return lReLU(b1 * ReLU(delta + b2)**mu + b0)
        else:
            return delta


    @jit
    def loss(self, params):
        """
        LDL loss function
        No validation, could be much faster if validation is not needed

        Parameters:
            params (array-like): Parameters of LDL model

        Returns:
            loss (float): LDL model loss
        """
        # First allbcast the parameters
        params = allbcast(params, self.pm.comm)

        # Compute the LDL map
        F = self.LDL(params, self.X, Nstep=self.Nstep, baryon=self.baryon) ** self.index

        # Optionally multiply by second field
        if self.field2 is not None:
            F = F * self.field2

        # Residue field
        residue = F - self.target

        # Smooth the field
        # !!! NO C2F HERE AS PER ORIGINAL CODE !!!
        smoothingkernel = jnp.where(self.knorms==0, 1., self.knorms**(-self.n) + 1.)
        #smoothingkernel = jnp.exp(-self.n * self.knorms)
        #smoothingkernel = (self.knorms + 1.)**(-self.n)
        residuek = self.pm.r2c(residue)
        residuek *= smoothingkernel
        residue = jnp.abs(self.pm.c2r(residuek))
        
        # Apply mask
        if self.masktrain is not None:
            residue *= self.masktrain
            Npixel = jnp.sum(self.masktrain)
        else:
            Npixel = residue.size

        # Compute loss (could also change L1 to a float exponent directly)
        if self.L1:
            loss = jnp.sum(residue)
        else:
            loss = jnp.sum(residue**2)

        # Collect losses and number of pixels from all ranks
        loss, _ = mpi4jax.allreduce(loss, op=MPI.SUM, comm=self.pm.comm)
        Npixel, _ = mpi4jax.allreduce(Npixel, op=MPI.SUM, comm=self.pm.comm)
        
        gc.collect() # Collect garbage
        return loss / Npixel

    def loss_grad(self, params):
        out = grad(self.loss)(params)
        gc.collect()
        return out

    def loss_and_grad(self, params):
        out1, out2 = value_and_grad(self.loss)(params)
        gc.collect()
        return out1, out2

    @jit
    def lossv(self, params):
        """
        Same as above, but if validation mask was provided,
        also computes validation loss and returns it.
        """
        params = allbcast(params, self.pm.comm)
        
        F = self.LDL(params, self.X, Nstep=self.Nstep, baryon=self.baryon) ** self.index
        
        if self.field2 is not None:
            F *= self.field2

        residue = F - self.target
        
        #smoothingkernel = jnp.where(self.knorms==0, 1., self.knorms**(-self.n) + 1.)
        #residuek = self.pm.r2c(residue)
        #residuek *= smoothingkernel
        #residue = jnp.abs(self.pm.c2r(residuek))
        residue = jnp.abs(residue)
        
        if self.masktrain is not None:
            residuel = residue * self.masktrain
            Npixel = jnp.sum(self.masktrain)
        else:
            residuel = residue
            Npixel = residue.size
        
        if self.L1:
            loss = jnp.sum(residuel)
        else:
            loss = jnp.sum(residuel**2)

        loss, _ = mpi4jax.allreduce(loss, op=MPI.SUM, comm=self.pm.comm)
        Npixel, _ = mpi4jax.allreduce(Npixel, op=MPI.SUM, comm=self.pm.comm)
        loss /= Npixel

        # Optionally compute and store validation loss
        lossv = 0.
        if self.maskvalid is not None:
            residue *= self.maskvalid
            Npixelv = jnp.sum(self.maskvalid)
            if self.L1:
                lossv = jnp.sum(residue)
            else:
                lossv = jnp.sum(residue**2)
            lossv, _ = mpi4jax.allreduce(lossv, op=MPI.SUM, comm=self.pm.comm)
            Npixelv, _ = mpi4jax.allreduce(Npixelv, op=MPI.SUM, comm=self.pm.comm)
            lossv /= Npixelv

        gc.collect()
        return loss, lossv

    def _lossv(self, params, lossvp):
        # Trick to also return lossv and be able to compute gradient
        o1, o2 = self.lossv(params)
        gc.collect()
        return o1 + lossvp*o2

    def lossv_and_grad(self, params):
        out1, (out3, out2) = value_and_grad(self._lossv, argnums=(0, 1))(params, 0.)
        gc.collect()
        return out1, out2, out3
    
    def tree_flatten(self):
        children = (self.X, self.target, self.field2, self.masktrain, self.maskvalid)
        aux_data = (self.pm, self.Nstep, self.baryon, self.n, self.index, self.L1)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*(children[:2]), *aux_data, *(children[2:]))


class RMSProp(object):
    """
    RMSprop optimizer class
    """
    def __init__(self, LR=0.01, beta=0.999, eps=1e-8, eps_root=0.):
        """
            LR (float, array-like or callable): learning rate
            beta (float): exponential decay rate of RMSprop
        """
        self.LR = LR
        self.beta = beta
        self.eps = eps
        self.eps_root = eps_root

    def init(self, params0):
        self.v = jnp.zeros(len(params0))
        self.itr = 0

    def step(self, params, gradient):
        """
        Computes a RMSprop update of the parameters

        Parameters:
            params (array-like): Parameters to optimize
            gradient (array-like): Gradient of loss with respect to parameters

        Returns:
            Updated parameters
        """
        if callable(self.LR):
            LR = self.LR(self.itr)
        else:
            LR = self.LR
        self.v = (1 - self.beta) * gradient**2 + self.beta * self.v
        self.itr += 1
        vh = jnp.sqrt(self.v / (1. - self.beta**self.itr) + self.eps_root)
        update = gradient / (vh + self.eps)
        return params - LR * update

class Adam(object):
    """
    Adam optimizer class
    """
    def __init__(self, LR=0.01, b1=0.9, b2=0.999, eps=1e-8, eps_root=0.):
        self.LR = LR
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.eps_root = eps_root

    def init(self, params0):
        self.mu = jnp.zeros(len(params0))
        self.nu = jnp.zeros(len(params0))
        self.itr = 0

    def step(self, params, gradient):
        """
        Computes an Adam update of the parameters

        Parameters:
            params (array-like): Parameters to optimize
            gradient (array-like): Gradient of loss with respect to parameters

        Returns:
            Updated parameters
        """
        if callable(self.LR):
            LR = self.LR(self.itr)
        else:
            LR = self.LR
        self.mu = self.b1 * self.mu + (1. - self.b1) * gradient
        self.nu = self.b2 * self.nu + (1. - self.b2) * gradient**2
        self.itr += 1
        muh = self.mu / (1. - self.b1**self.itr)
        nuh = self.nu / (1. - self.b2**self.itr)
        update = muh / (jnp.sqrt(nuh + self.eps_root) + self.eps)
        return params - LR * update

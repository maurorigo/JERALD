import jax.numpy as jnp
import numpy as np
import jax
from functools import partial
from jax import jit, grad, value_and_grad, custom_vjp
from mpi4py import MPI
import mpi4jax
import gc
import warnings

@custom_vjp
def c2f(obj):
    """
        This function compensates the fact that we only have int(Nmesh[2]/2)+1
        of the fft along z due to symmetry. The problem is not present on the
        forward pass, but when computing derivatives it becomes significant.
    """
    return obj

def c2f_fwd(obj):
    return obj, ()

def c2f_bwd(res, v):
    # Assuming the second dimension of the fft is the correct one,
    # but this can be done more general if needed
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
        variables (I believe), this sets it back to correct.
        In the forward pass, all values should already be the same, so just return val
        !!!  Also works for a jnp array of parameters (recommended in order to guarantee
        !!!  correct order of ops)
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
class JERALDModel(object):
    """
    JERALD model for dark matter, stellar mass and neutral hydrogen
    """
    def __init__(self, X, target, pm, Nstep=2, kind="dm", n=None, index=1., L1=None, field2=None, masktrain=None, maskvalid=None, starmap=None):
        """
            X (array-like): (?, 3) array of local input positions
            target (array-like): (?, Ny, Nz) array of local target map
            pm (PMesh object): PMesh that performs fft, paint, readout
            Nstep (int): Number of displacement steps
            kind (str): Quantity to model ("dm", "sm", "HI", potentially also LDL ones, using sm model)
            n (float): Smoothing kernel exponent (if None no smoothing)
            index (float): Exponent for second map (for compatibility with LDL)
            field2 (array-like): Additional (?, Ny, Nz) map to be multiplied to output
                                 (for compatibility with LDL)
            masktrain (array-like): (?, Ny, Nz) local array of 0s and 1s for selecting training pixels
            maskvalid (array-like): As above, for validation
            L1 (bool): L1 or L2 norm loss
            starmap (array-like): (?, Ny, Nz) array of local stellar mass map (required for HI)

            !! NOTE: X and target are only needed at initialization for training. When evaluating only,
            they can be passed directly to model.evaluate() or model.displace()

            NOTE: initializing everything here to make compilation of class faster via pytrees,
            avoiding constant folding of huge arrays which can lead to massive slowdowns.
        """
        self.pm = pm
        self.X = X
        self.target = target

        # Useful later
        self.kvals = self.pm.compute_wavevectors()
        self.knorms = jnp.linalg.norm(self.kvals, axis=-1)
        self.knorms = jnp.where(self.knorms==0, 1e-8, self.knorms) # To avoid nan's when k**-1

        # Loss parameters
        self.Nstep = Nstep
        self.kind = kind
        # For unknown cases just use the stellar mass model (as LDL basically)
        if self.kind not in ["dm", "sm", "HI"]:
            self.kind = "sm"
        self.n = n
        self.index = index
        self.field2 = field2
        self.masktrain = masktrain
        self.maskvalid = maskvalid
        if L1 is None:
            self.L1 = baryon # Paper uses True for baryons and False otherwise
        else:
            self.L1 = L1

        self.starmap = starmap
        if starmap:
            self.starmapk = self.pm.r2c(self.starmap)
        elif kind == "HI":
            raise Exception("Star map required for HI target")

        # Number of parameters in the potential for different targets
        if self.kind in ["dm", "sm"]:
            self.Npot = 5
        elif self.kind == "HI":
            self.Npot = 10

    @jit
    def potential_gradient_dm(self, params, X):
        """
        Computes gradient of potential for JERALD for dark matter

        Parameters:
            params (tuple): Parameters of Green's function and delta exponent
            X (array-like): (N, 3) array of particle positions

        Returns:
            Gradient of LDL potential with shape (Nx, Ny, Nz, 3)
        """
        fact, alpha, gamma, kh, kl, nu = params

        # Density
        rho = self.pm.paint(X, 1.) * fact # Normalized density rho/bar(rho)
        # Power of density in Fourier space
        rhok = self.pm.r2c((rho+1e-8) ** gamma)
        
        # Green's operator in Fourier space
        greenk = -jnp.exp(-self.knorms/kl -kh/self.knorms) * self.knorms**nu # Attractive so -
        greenk = c2f(greenk) # Compensates the fact that we only have ~half of the fft (for back pass)

        potk = 1e-2 * alpha * rhok * greenk

        # Now we need to compute the spacial derivative of this object on each point
        # Note that we only have the values at some grid points, so we need to use a finite difference approach
        # Compare the FT of this (convolution) with an order (2 commented) 3 finite difference formula (https://en.wikipedia.org/wiki/Numerical_differentiation)
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

        # !!! WIP: This is faster but for now I'm passing gradient components stacked and it's
        # expensive from a memory point of view, I'll change this in the future
        # return self.pm.readout3D(X, jnp.stack((potgradx, potgrady, potgradz), axis=0))

    @jit
    def potential_gradient_sm(self, params, X):
        """
        Same as above, for stellar mass
        """
        fact, alpha, gamma, kh, kl, nu = params

        rho = self.pm.paint(X, 1.) * fact
        rhok = self.pm.r2c((rho+1e-8) ** gamma)

        greenk = -jnp.exp(-self.knorms**2/kl**2 -kh**2/self.knorms**2) * self.knorms**nu
        greenk = c2f(greenk)

        potk = 1e-2 * alpha * rhok * greenk

        step = self.pm.BoxSize / self.pm.Nmesh
        coeff = 1 / (30 * step) * (45 * jnp.sin(self.kvals * step) - 9 * jnp.sin(2 * self.kvals * step) + jnp.sin(3 * self.kvals * step))

        potgradk = -1j * coeff[:, :, :, 0] * potk
        potgradx = self.pm.c2r(potgradk)
        potgradk = -1j * coeff[:, :, :, 1] * potk
        potgrady = self.pm.c2r(potgradk)
        potgradk = -1j * coeff[:, :, :, 2] * potk
        potgradz = self.pm.c2r(potgradk)

        return jnp.stack((self.pm.readout(X, potgradx),
            self.pm.readout(X, potgrady),
            self.pm.readout(X, potgradz)), axis=-1)

    @jit
    def potential_gradient_HI(self, params, X):
        """
        Same as above, for HI (this time more parameters are needed)
        """
        fact, alpha1, gamma1, kh1, kl1, nu1, alpha2, gamma2, kh2, kl2, nu2 = params

        rho = self.pm.paint(X, 1.) * fact
        rhok = self.pm.r2c((rho+1e-8) ** gamma1)

        greenk1 = -jnp.exp(-self.knorms**2/kl1**2 -kh1**2/self.knorms**2) * self.knorms**nu1
        greenk1 = c2f(greenk1)

        # Same as above but for stellar mass map
        starmapk = self.pm.r2c(self.starmap ** gamma2)

        greenk2 = jnp.exp(-self.knorms**2/kl2**2 -kh2**2/self.knorms**2) * self.knorms**nu2
        greenk2 = c2f(greenk2)

        potk = 1e-3 * (alpha1 * greenk1 * rhok + alpha2 * greenk2 * starmapk)

        step = self.pm.BoxSize / self.pm.Nmesh
        coeff = 1 / (30 * step) * (45 * jnp.sin(self.kvals * step) - 9 * jnp.sin(2 * self.kvals * step) + jnp.sin(3 * self.kvals * step))

        potgradk = -1j * coeff[:, :, :, 0] * potk
        potgradx = self.pm.c2r(potgradk)
        potgradk = -1j * coeff[:, :, :, 1] * potk
        potgrady = self.pm.c2r(potgradk)
        potgradk = -1j * coeff[:, :, :, 2] * potk
        potgradz = self.pm.c2r(potgradk)

        return jnp.stack((self.pm.readout(X, potgradx),
            self.pm.readout(X, potgrady),
            self.pm.readout(X, potgradz)), axis=-1)

    @partial(jit, static_argnums=3)
    def displace(self, params, X, Nstep):
        """
        Displaces particles according to JERALD model

        Parameters:
            params (array-like): Model parameters
            X (array-like): Particle positions (divided into different ranks)
            Nstep (int): Number of displacement layers
        
        Returns:
            Displaced particle positions
        """

        sz, _ = mpi4jax.allreduce(len(X), op=MPI.SUM, comm=self.pm.comm)
        fact = self.pm.Nmesh.prod() / sz # Normalization for density

        def body(i, X):
            
            alpha = params[Npot*i+0]
            gamma = params[Npot*i+1]
            kh = params[Npot*i+2]
            kl = params[Npot*i+3]
            nu = params[Npot*i+4]

            if self.kind == "dm":
                return X + self.potential_gradient_dm((fact, alpha, gamma, kh, kl, nu), X)
            elif self.kind == "sm":
                return X + self.potential_gradient_sm((fact, alpha, gamma, kh, kl, nu), X)
            elif self.kind == "HI":
                alpha2 = params[Npot*i+5]
                gamma2 = params[Npot*i+6]
                kh2 = params[Npot*i+7]
                kl2 = params[Npot*i+8]
                nu2 = params[Npot*i+9]
                return X + self.potential_gradient_HI((fact, alpha, gamma, kh, kl, nu, alpha2, gamma2, kh2, kl2, nu2), X)
            
        # Using fori_loop because with normal for JAX can't compute the derivative wrt gamma
        # in case len(X) is not divisible by pm.comm.size and with Nstep=1 (but with 2 yes,
        # for some reason, not sure why)
        return jax.lax.fori_loop(0, Nstep, body, X)

    @partial(jit, static_argnums=(3, 4))
    def evaluate(self, params, X, Nstep):
        """
        Evaluates dm, sm or HI map according to the JERALD model

        Parameters:
            params (array-like): Model parameters
            X (array-like): Particle positions (divided into different ranks)
            Nstep (int): Number of displacement layers

        Returns:
            dm, sm or HI map
        """

        sz, _ = mpi4jax.allreduce(len(X), op=MPI.SUM, comm=self.pm.comm)
        fact = self.pm.Nmesh.prod() / sz

        X = self.displace(params, X, Nstep=Nstep)

        # Paint particle overdensity field
        rho = self.pm.paint(X, 1.) * fact
        
        if self.kind == "HI":
            mu = params[self.Npot*Nstep+0]
            w = params[self.Npot*Nstep+1]
            b = params[self.Npot*Nstep+2]

            beta1 = params[self.Npot*Nstep+3]
            khi = params[self.Npot*Nstep+4]
            kli = params[self.Npot*Nstep+5]
            nui = params[self.Npot*Nstep+6]
            beta2 = params[self.Npot*Nstep+7]
            gamma3 = params[self.Npot*Nstep+8]

            xi = params[self.Npot*Nstep+9]
            eta = params[self.Npot*Nstep+10]

            # Field transformation
            starmapk = self.pm.r2c(self.starmap)
            filterk = jnp.exp(-self.knorms/kli -khi/self.knorms)*self.knorms**nui + xi
            filterk = c2f(filterk)

            depletion = ReLU(beta1 * self.pm.c2r(starmapk*filterk) + eta) + beta2*self.starmap**gamma3

            return ReLU(w * (rho+1e-8)**mu - depletion + b)

        elif self.kind == "sm":
            mu = params[self.Npot*Nstep+0]
            w = params[self.Npot*Nstep+1]
            b = params[self.Npot*Nstep+2]
        
            # Field transformation
            return ReLU(w * (rho+1e-8)**mu + b)

        elif self.kind == "dm":
            return rho


    @jit
    def loss(self, params):
        """
        JERALD loss function
        No validation, could be much faster if validation is not needed

        Parameters:
            params (array-like): Parameters of model

        Returns:
            loss (float): Model loss
        """
        # First allbcast the parameters
        params = allbcast(params, self.pm.comm)

        # Compute the map (index is for compatibility with LDL code)
        F = self.evaluate(params, self.X, Nstep=self.Nstep) ** self.index

        # Optionally multiply by second field (for compatibility with LDL)
        if self.field2:
            F = F * self.field2

        # Residue field
        diff = F - self.target

        # Smooth the field
        if self.n:
            # !!! NO C2F HERE AS PER ORIGINAL CODE !!!
            smoothingkernel = jnp.where(self.knorms==0, 1., self.knorms**(-self.n) + 1.)
            diffk = self.pm.r2c(diff)
            diffk *= smoothingkernel
            diff = jnp.abs(self.pm.c2r(diffk))
        else:
            diff = jnp.abs(diff)
        
        # Apply mask
        if self.masktrain:
            diff *= self.masktrain
            Npixel = jnp.sum(self.masktrain)
        else:
            Npixel = diff.size

        # Compute loss (could also change L1 to a float exponent directly)
        if self.L1:
            loss = jnp.sum(diff)
        else:
            loss = jnp.sum(diff**2)

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
        
        F = self.evaluate(params, self.X, Nstep=self.Nstep) ** self.index
        
        if self.field2:
            F *= self.field2

        diff = F - self.target
        
        if self.n:
            smoothingkernel = jnp.where(self.knorms==0, 1., self.knorms**(-self.n) + 1.)
            diffk = self.pm.r2c(diff)
            diffk *= smoothingkernel
            diff = jnp.abs(self.pm.c2r(diffk))
        else:
            diff = jnp.abs(diff)

        if self.masktrain:
            diffl = diff * self.masktrain
            Npixel = jnp.sum(self.masktrain)
        else:
            diffl = diff
            Npixel = diff.size
        
        if self.L1:
            loss = jnp.sum(diffl)
        else:
            loss = jnp.sum(diffl**2)

        loss, _ = mpi4jax.allreduce(loss, op=MPI.SUM, comm=self.pm.comm)
        Npixel, _ = mpi4jax.allreduce(Npixel, op=MPI.SUM, comm=self.pm.comm)
        loss /= Npixel

        # Optionally compute and store validation loss
        lossv = 0.
        if self.maskvalid:
            diff *= self.maskvalid
            Npixelv = jnp.sum(self.maskvalid)
            if self.L1:
                lossv = jnp.sum(diff)
            else:
                lossv = jnp.sum(diff**2)
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
    
    # Things needed to compile class
    def tree_flatten(self):
        children = (self.X, self.target, self.field2, self.masktrain, self.maskvalid, self.starmap)
        aux_data = (self.pm, self.Nstep, self.kind, self.n, self.index, self.L1)
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
            eps (float): const for stability
            eps_root (float): const for stability
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
        """
            LR (float, array-like or callable): learning rate
            b1 (float): momentum
            b2 (float): exponential decay rate
            eps (float): const for stability
            eps_root (float): const for stability
        """
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

    def get_state(self):
        """
        Prints current value of internal parameters of Adam and current learning rate
        """
        if callable(self.LR):
            LR = self.LR(self.itr)
        else:
            LR = self.LR
        return self.mu, self.nu, self.itr, LR

    def import_state(self, filename, paramsreq=False):
        """
        Imports Adam state from file "filename" and optionally some pre-saved parameters (if paramsreq)
        """
        paramsfound = False
        params = []
        with open(filename) as f:
            lines = f.readlines()
            state = 0
            mu = []
            nu = []
            for line in lines:
                line = line.rstrip()
                if state == 0:
                    if line == "mu":
                        state = 1
                elif state == 1:
                    if line == "nu":
                        state = 2
                    else:
                        mu.append(float(line))
                elif state == 2:
                    if line == "itr":
                        state = 3
                    else:
                        nu.append(float(line))
                elif state == 3:
                    itr = int(line)
                    state = 4
                elif state == 4:
                    if line == "params":
                        state = 5
                        paramsfound = True
                elif state == 5:
                    params.append(float(line))

            self.mu = jnp.array(mu)
            self.nu = jnp.array(nu)
            self.itr = itr

        if paramsreq:
            if paramsfound:
                return True, jnp.array(params)
            else:
                warnings.warn(f"Parameters not found in {filename}")
                return False, 0

        def save_state(self, filename, params=None):
        """
        Saves to "filename" the current state of the optimizer,
        optionally adding parameters being optimized, if passed
        """
        if callable(self.LR):
            LR = self.LR(self.itr)
        else:
            LR = self.LR
        with open(filename, "w") as f:
            f.write("mu\n")
            np.savetxt(f, self.mu)
            f.write("nu\n")
            np.savetxt(f, self.nu)
            f.write("itr\n")
            f.write(str(self.itr) + "\n")
            f.write("LR\n")
            if len(LR) > 0:
                np.savetxt(f, LR)
            else:
                f.write(str(LR) + "\n")
            if params:
                f.write("params\n")
                np.savetxt(f, params)

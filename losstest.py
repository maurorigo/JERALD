# Testing package
from model import LDLModel
from pmesh import PMesh
import numpy as np
import time
import jax.numpy as jnp
from jax import config, grad
config.update("jax_enable_x64", True)
from mpi4py import MPI
import mpi4jax
import tracemalloc
import jax
import sys

tracemalloc.start()

Ngrid=20

pm = PMesh(Ngrid, 205.)
comm = pm.comm
rank = comm.rank
size = comm.size
if rank==0:
    print("Mesh initialized")
    sys.stdout.flush() # In case it's run with sbatch this may be needed

np.random.seed(83)
pos = jnp.array(np.random.rand(Ngrid**3, 3)) * 205.
randomvels = 10*jnp.cos(jnp.linalg.norm(pos, axis=1)/19).reshape((len(pos), 1))
pos = pos.at[:].add(randomvels)
# Local positions (just easy division)
start = int(Ngrid**3/size*rank)
end = int(Ngrid**3/size*(rank+1))
pos = pos[start:end]
# Local positions (smarter division)
"""starts = (Ngrid**3/size*np.arange(size)).astype(int)
ends = (Ngrid**3/size*(np.arange(size)+1)).astype(int)
Nout = ends - starts
pos = pm.get_local_positions(pos, Nout)"""
print(f"Process {rank} reading {len(pos)} particles from {start} to {end}")
sys.stdout.flush()

# Parameters for LDL
Nstep = 2
param = jnp.array([.001, .5, 1., 8., 0.]*Nstep + [1., 1., 0.])
target = jnp.array(np.random.rand(Ngrid, Ngrid, Ngrid))
# Local target
targetl = target[pm.localS:pm.localS+pm.localL, :, :]
maskl = jnp.ones_like(targetl)
maskl2 = jnp.array(np.random.randint(0, 2, target.shape)[pm.localS:pm.localS+pm.localL, :, :])
model = LDLModel(pos, targetl, pm, Nstep=Nstep, baryon=True, masktrain=maskl, maskvalid=maskl2)

# Compute a first time for JIT compilation
loss, lossv = model.lossv(param)

tim = time.time()
loss, lossv = model.lossv(param)
if rank==0:
    print(f"Loss alone computed in {((time.time()-tim)*1000):.3f}ms")
    print(f"Loss: {loss}, validation loss: {lossv}")
    sys.stdout.flush()

loss, lossv, grad = model.lossv_and_grad(param)
tim = time.time()
loss, lossv, grad = model.lossv_and_grad(param)
if rank==0:
    print(f"Loss with gradient computed in {((time.time()-tim)*1000):.3f}ms")
    print("Gradient:")
    print(grad)
    sys.stdout.flush()

snapshot = tracemalloc.take_snapshot()
stats = snapshot.statistics('lineno')
total = comm.allreduce(sum(stat.size for stat in stats))
if rank == 0:
    print("Allocated size: %.1f KiB" % (total / 1024))
    sys.stdout.flush()


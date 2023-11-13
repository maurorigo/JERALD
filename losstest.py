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

tracemalloc.start()

Ngrid=50

pm = PMesh(Ngrid, 205.)
comm = pm.comm
rank = comm.rank
size = comm.size

np.random.seed(19)
pos = jnp.array(np.random.rand(Ngrid**3, 3)) * 205.
randomvels = 20*jnp.cos(jnp.linalg.norm(pos, axis=1)/19).reshape((len(pos), 1))
pos = pos.at[:].add(randomvels)
# Local positions
pos = pos[int(Ngrid**3/size*rank):int(Ngrid**3/size*(rank+1))]
print(f"Process {rank} from {int(Ngrid**3/size*rank)} to {int(Ngrid**3/size*(rank+1))} for a total of {Ngrid**3} particles")

# Parameters for LDL
Nstep = 2
param = jnp.array([100000., 0.5, 1., 8., 0.]*Nstep + [1., 1., 0.])
target = jnp.array(np.random.rand(Ngrid, Ngrid, Ngrid))
# Local target
target = target[pm.localS:pm.localS+pm.localL, :, :]
model = LDLModel(pm=pm)

model.set_loss_params(Nstep, baryon=True)
loss = model.loss(param, pos, target)

tim = time.time()
loss = model.loss(param, pos, target)
if rank==0:
    print(f"Loss computed in {((time.time()-tim)*1000):.3f}ms")
    print(f"Loss: {loss}")

grad = model.loss_gradient(param, pos, target)
tim = time.time()
grad = model.loss_gradient(param, pos, target)
if rank==0:
    print(f"Gradient computed in {((time.time()-tim)*1000):.3f}ms")
    print(grad)

snapshot = tracemalloc.take_snapshot()
stats = snapshot.statistics('lineno')
total = sum(stat.size for stat in stats)
if rank == 0:
    print("Allocated size: %.1f KiB" % (total / 1024))

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

size=100

pm = PMesh(size, 205.)
comm = pm.comm
rank = comm.rank
sz = comm.size

np.random.seed(19)
parts = np.array(np.random.rand(size**3, 3)) * 205.
randomvels = 20*np.cos(np.linalg.norm(parts, axis=1)/19).reshape((len(parts), 1))
shparts = parts + randomvels
shparts = jnp.array(shparts)
shparts = shparts[int(size**3/sz*rank):int(size**3/sz*(rank+1))]
print(f"Process {rank} from {int(size**3/sz*rank)} to {int(size**3/sz*(rank+1))} for a total of {size**3}")

Nstep = 2
param = jnp.array([100000., 0.5, 1., 8., 0.]*Nstep + [1., 1., 0.])
target = jnp.array(np.random.rand(size, size, size))
# LOCAL TARGET
target = target[pm.localS:pm.localS+pm.localL, :, :]
model = LDLModel(pm=pm)

model.set_loss_params(Nstep, baryon=True)
loss = model.loss(param, shparts, target)

tim = time.time()
loss = model.loss(param, shparts, target)
if rank==0:
    print(f"Loss computed in {((time.time()-tim)*1000):.3f}ms")
    print(f"Loss: {loss}")

grad = model.loss_gradient(param, shparts, target)
tim = time.time()
grad = model.loss_gradient(param, shparts, target)
if rank==0:
    print(f"Gradient computed in {((time.time()-tim)*1000):.3f}ms")
    print(grad)

snapshot = tracemalloc.take_snapshot()
stats = snapshot.statistics('lineno')
total = sum(stat.size for stat in stats)
if rank == 0:
    print("Allocated size: %.1f KiB" % (total / 1024))

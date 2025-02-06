import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
import numpy as np
from model import JERALDModel, Adam
from pmesh import PMesh
from maputils import loadfield, savefield
import time
import argparse
from bigfile import File
from mpi4py import MPI
import jax.numpy as jnp
import inspect
from utils import MpiExceptionHandler

from jax import config
config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser()

parser.add_argument('--target', type=str, default='dm', choices=['dm', 'Mstar', 'HI'], help='target field kind (only JERALD, no LDL)')
parser.add_argument('--Nbodypath', type=str, help='path to load N-body (FastPM) sim particles')
parser.add_argument('--improvepath', type=str, help='path to load parameters for improving N-body matter distribution')
parser.add_argument('--refpath', type=str, help='path to read reference map')
parser.add_argument('--Mstarpath', type=str, help='path to read stellar mass map (required for HI)')
parser.add_argument('--Nmesh', type=int, help='mesh resolution')
parser.add_argument('--BoxSize', type=float, help='simulation box size')
parser.add_argument('--Nstep', type=int, help='number of displacement layers')
parser.add_argument('--n', type=float, help='hyperparameter n in the smoothing kernel (only for Mstar, kept this way for consistency with LDL)')
parser.add_argument('--outpath', type=str, help='output path for saving maps/parameters/optimizer state')
parser.add_argument('--optpath', type=str, help='when specified, loads optimizer state from specified path (to resume training)')
parser.add_argument('--evalpath', type=str, help='when specified, does not train but only evaluates using parameters saved in specified path')
parser.add_argument('--redshift', type=float, help='the redshift')

args = parser.parse_args()


# Particle mesh
pm = PMesh(args.Nmesh, args.BoxSize)
comm = pm.comm
pm.printr("Mesh initialized.")
pm.printr([f"JERALD for {args.target} at redshift {args.redshift}",
    f"Mesh size: {args.Nmesh}",
    f"Using {args.Nstep} steps and n = {args.n}"])

# Load input data
with MpiExceptionHandler(comm):
    X = File(args.Nbodypath)['Position']
# Local positions
# fancy: import to this rank mostly particles belonging to the part of the CIC map to paint that
# is processed in this rank, to minimize communications when painting
# Takes a bit more to import but can bring 10%-20% gain in time
fancy = True
if fancy:
    starts = np.arange(comm.size) * X.size // comm.size
    ends = (np.arange(comm.size) + 1) * X.size // comm.size
    Nout = ends - starts
    X = jnp.array(pm.get_local_positions(X[:], Nout))
else:
    start = comm.rank * X.size // comm.size
    end = (comm.rank+1) * X.size // comm.size
    X = jnp.array(X[start:end])
NpartTot = comm.allreduce(len(X))

# Improve N-body matter distribution using JERALD with given parameters
# Can also save improved positions but this way I can keep track of time for full pipeline
if args.improvepath:
    pm.printr("Improving small scales in FastPM.")
    with MpiExceptionHandler(comm):
        params = np.loadtxt(args.improvepath) # Compatible with LDL
    if len(params) % 5 != 0:
        raise Exception("Number of parameters for dm improve should be a multiple of 5")
    Nstep = len(params) // 5 # Infer number of steps from the number of parameters
    model = JERALDModel(None, None, pm) # dm model by default
    times = time.time()
    X = model.displace(params, X, Nstep)
    comm.Barrier()
    pm.printr(f"Done in {(time.time()-times):.2f}s.")
    del model

# Import stellar mass map for HI
if args.target == 'HI':
    pm.printr("Loading stellar mass map...")
    starmap, _, __ = loadfield(args.Mstarpath, local=True, pm=pm)
    # Basically ReLU, just so that dividing by this and raising it to powers doesn't give nan's
    starmap = jnp.where(starmap < 1e-12, 1e-12, starmap)
    starmapmean = comm.allreduce(starmap.sum()) / comm.allreduce(np.prod(starmap.shape))
    starmap /= starmapmean
else:
    starmap = None

# Train!
if not args.evalpath:
    # Target map
    pm.printr("Loading target map...")
    
    targetmap, _, __ = loadfield(args.refpath, local=True, pm=pm)
    if args.target == 'HI':
        # Depending on redshift HI is very different in range for Sherwood-Relics
        # Normalize maps around unity
        if args.redshift == 0:
            targetmap /= 4e-13
            zshift = 0
        elif args.redshift == 2:
            targetmap /= 4e-11
            zshift = -1e-3
        elif args.redshift == 5:
            targetmap /= 1e-8
            zshift = -0.04

    pm.printr("Done.")

    # Split among training, validation and test set
    lmi = np.array(pm.local_mesh_indices()) # Unflattened local mesh indices
    Nmesh_valid = int(0.5 * args.Nmesh)
    Nmesh_test = int(0.43 * args.Nmesh)
    select_validate = (lmi[:,:,:,2]<Nmesh_valid)
    select_test = (lmi[:,:,:,0]<Nmesh_test) & (lmi[:,:,:,1]<Nmesh_test) & (lmi[:,:,:,2]<Nmesh_test)
    mask_train = np.ones_like(targetmap, dtype='bool')
    mask_train[select_validate] = False
    mask_validate = np.zeros_like(targetmap, dtype='bool')
    mask_validate[select_validate] = True
    mask_validate[select_test] = False
    mask_test = np.zeros_like(targetmap, dtype='bool')
    mask_test[select_test] = True
    mask_train = jnp.array(mask_train, dtype=float)
    mask_validate = jnp.array(mask_validate, dtype=float)

    trsum = comm.allreduce(mask_train.sum())
    vasum = comm.allreduce(mask_validate.sum())
    tesum = comm.allreduce(mask_test.sum())
    fulln = comm.allreduce(targetmap.size)
    pm.printr([f"Ntrain: {trsum:.0f}, Nvalid: {vasum:.0f}, Ntest: {tesum:.0f}, total: {fulln:.0f}",
        f"Ratios: train = {trsum/fulln*100:.1f}%, valid = {vasum/fulln*100:.1f}%, test = {tesum/fulln*100:.1f}%"])
    
    L1 = False if args.target == "dm" else True
    # Build model
    model = JERALDModel(X, targetmap, pm, Nstep=args.Nstep, kind=args.target, n=args.n, L1=L1, masktrain=mask_train, maskvalid=mask_validate, starmap=starmap)
    pm.printr("Model created.")

    # Initial params
    avgfact = comm.allreduce(targetmap.sum()) / NpartTot
    if args.target == 'dm':
        params0 = [1, 0.5, 1, 8, 0] * args.Nstep
    elif args.target == 'Mstar':
        params0 = [1, 0.5, 1, 8, 0] * args.Nstep
        params0 += [1, avgfact, 0, 0]
    elif args.target == 'HI':
        params0 = [1, 0.5, 1, 8, 0, 1, 0.5, 1, 8, 0] * args.Nstep
        params0 += [1, avgfact, -.1, 0.6, 1, 8, .01, 0, .1, 1, 0., zshift]
    params0 = jnp.array(params0)
    
    # Bounds (optimizer doesn't really do anything with them, they're just used to clip parameters)
    if args.target == 'dm':
        #         alpha         gamma         kh          kl           nu
        bounds = [(None, None), (None, None), (0., None), (0.1, None), (None, None)] * args.Nstep
    elif args.target == 'Mstar':
        bounds = [(None, None), (None, None), (0., None), (0.1, None), (None, None)] * args.Nstep
        #          mu           w             b
        bounds += [(0.1, None), (1e-5, None), (None, None)]
    elif args.target == 'HI':
        bounds = [(None, None), (None, None), (0., None), (0.1, None), (None, None),
                  (None, None), (None, None), (0., None), (0.1, None), (None, None)] * args.Nstep
        #          mu           w             b             beta1         khi         kli
        bounds += [(0.1, None), (1e-5, None), (None, None), (None, None), (0., None), (0.1, None),
        #          nui           beta2         gamma3        xi            eta           zshift
                   (None, None), (None, None), (None, None), (None, None), (None, None), (None, None)]

    # Train
    Nopt = 10
    # Path to save optimizer state in case one wants to continue training
    optf = f"{args.outpath}/{args.target}_Nmesh{args.Nmesh}_Nstep{args.Nstep}_n{args.n:.2f}_optstate.txt"
    pm.printr(f"Nopt: {Nopt}")
    def LRschedule(i, LRi=0.002, LRf=0.02, tau=100):
        LRt = (LRf-LRi) * (2/(np.exp(-i/tau)+1)-1) + LRi
        LRs = LRt * jnp.ones_like(params0)
        return LRs
    pm.printr("LR schedule:")
    LRsrc = inspect.getsource(LRschedule)
    pm.printr(LRsrc) # Print the LR schedule for a complete record
    optimizer = Adam(LRschedule)
    optimizer.init(params0)
    if args.optpath:
        pm.printr("Importing optimizer state...")
        cond, par0 = optimizer.import_state(args.optpath, True)
        params0 = par0 if cond else params0
        pm.printr("Done.")
    params = params0
    bestvloss = np.inf
    beststep = 0
    paramsave = f"{args.outpath}/{args.target}_Nmesh{args.Nmesh}_Nstep{args.Nstep}_n{args.n:.2f}_bestparam.txt"
    pm.printr("Beginning of training.")
    tstime = time.time()
    for i in range(Nopt):
        for j in range(len(params)): # Apply bounds to parameters (just clipping)
            if bounds[j][0] is not None:
                if params[j] < bounds[j][0]:
                    params = params.at[j].set(bounds[j][0])
            if bounds[j][1] is not None:
                if params[j] > bounds[j][1]:
                    params = params.at[j].set(bounds[j][1])

        stime = time.time() # Step time
        tloss, vloss, tgrad = model.lossv_and_grad(params) # Train loss, validation loss and train grad
        if vloss < bestvloss:
            bestvloss = vloss
            bestparams = params
            beststep = i+1
            # Save best parameters up to now
            with MpiExceptionHandler(comm): 
                if comm.rank == 0:
                    np.savetxt(paramsave, bestparams)

        pm.printr([f"Step {i+1}:",
            f"Train loss = {tloss:.5f}, validation loss = {vloss:.5f}, max gradient = {np.max(np.abs(tgrad)):.5f}, completed in {(time.time()-stime):.3f}s",
            "Params:",
            f"{params}", # Also print params and gradient
            "Grad:",
            f"{tgrad}"])
        
        # Advance optimizer
        params = optimizer.step(params, tgrad)

    pm.printr([f"Finished optimization in {(time.time()-tstime):.0f}s.", 
        f"Best validation loss: {bestvloss}, at step {beststep}",
        "Best parameters:",
        f"{bestparams}"])
    with MpiExceptionHandler(comm):
        if comm.rank == 0:
            print("Saving optimizer state...")
            optimizer.save_state(optf, params)
            print("Done.")
    
    params = bestparams
    del model

else: # Only evaluate model
    params = np.loadtxt(args.evalpath)
    if args.target == 'dm':
        reqlen = 5 * args.Nstep
    elif args.target == 'sm':
        reqlen = 5 * args.Nstep + 3
    elif args.target == 'HI':
        reqlen = 10 * args.Nstep + 11
    if len(params) != reqlen:
        raise Exception(f"Required {reqlen} parameters for {args.target} target with {args.Nstep} but only read {len(params)}")
        

# Evaluate
stime = time.time()
pm.printr(f"Producing {args.target} map...")
model = JERALDModel(None, None, pm, kind=args.target, starmap=starmap)
outmap = model.evaluate(params, X, args.Nstep)

pm.printr(f"Done in {(time.time()-stime):.2f}s. Saving field...")
savef = f"{args.outpath}/{args.target}_Nmesh{args.Nmesh}_Nstep{args.Nstep}_n{args.n:.2f}_map"
savefield(outmap, savef, pm)
pm.printr(f"Done.")


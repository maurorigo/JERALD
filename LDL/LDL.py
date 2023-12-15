# LDL
# Code is very similar to https://github.com/biweidai/LDL/blob/master/LDL.py
# Everything is float32
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
import numpy as np
from model import LDLModel, RMSProp
from pmesh import PMesh
from fieldIO import psavefield
from loadTNGdata import loaddata
import time
import argparse
from bigfile import File
from mpi4py import MPI
import jax.numpy as jnp
import optax

parser = argparse.ArgumentParser()

parser.add_argument('--target', type=str, default='Mstar', choices=['dm', 'Mstar', 'nHI', 'kSZ', 'tSZ_ne', 'tSZ_T', 'Xray_ne', 'Xray_T'], help='target field')
parser.add_argument('--FastPMpath', type=str, help='path to load in FastPM particles (if not provided, TNGDark will be used)')
parser.add_argument('--improveParams', type=str, help='path to load in LDL parameters for improving FastPM matter distribution')
parser.add_argument('--TNGpath', type=str, help='path to read TNG data (for target)')
parser.add_argument('--TNGDarkpath', type=str, help='path to read TNG Dark data (if FasPMpath is not provided)')
parser.add_argument('--snapNum', type=int, default=99, help='snapshot number of TNG and TNG Dark (default is 99, at redshift 0 for TNG300)')
parser.add_argument('--nChunks', type=int, help='number of chunks for snapshot (files composing it)')
parser.add_argument('--Nmesh', type=int, default=625, help='mesh resolution of LDL')
parser.add_argument('--BoxSize', type=float, default=205., help='simulation box size')
parser.add_argument('--Nstep', type=int, default=2, help='number of displacement layers in LDL')
parser.add_argument('--n', type=float, default=1., help='hyperparameter n in the smoothing kernel')
parser.add_argument('--save', type=str, help='optimized parameters path')
parser.add_argument('--pretrainParams', type=str, help='path to load pre-trained LDL parameters')
parser.add_argument('--nePretrainParams', type=str, help='path to load in the LDL parameters for ne (only applicable when fitting the temperature of tSZ and Xray)')
parser.add_argument('--evaluateOnly', action='store_true', help='when true, does not train but only evaluates')

args = parser.parse_args()

#Particle Mesh
pm = PMesh(Nmesh=args.Nmesh, BoxSize=args.BoxSize)
comm = pm.comm
if comm.rank==0:
    print("Mesh initialized.")
    sys.stdout.flush()

#load input data
if args.FastPMpath:
    if comm.rank==0:
        print("FastPM input selected.")
        sys.stdout.flush()
    X = File(args.FastPMpath)['Position']
    # Local positions
    start = comm.rank * X.size // comm.size
    end = (comm.rank+1) * X.size // comm.size
    X = jnp.array(X[start:end])
    lenXfull = comm.allreduce(len(X))
    # Improve FastPM matter distribution using LDL with given parameters
    if args.improveParams:
        if comm.rank==0:
            print("Improving small scales in FastPM.")
            sys.stdout.flush()
        params = np.loadtxt(args.improveParams)
        assert len(params) % 5 == 0
        Nstep = len(params) // 5 # Infer number of steps from the number of parameters
        displmodel = LDLModel(None, None, pm)
        X = displmodel.displace(params, X, Nstep)
        del displmodel
else:
    if not args.TNGDarkpath:
        raise Exception("At least one between FastPMpath and TNGDarkpath need to be specified. Run the script with -h for help.")
    else:
        if comm.rank==0:
            print("TNG Dark input selected.")
            sys.stdout.flush()
    # Local positions
    X = loaddata(args.TNGDarkpath, args.snapNum, args.nChunks, 'dm', pm, CICmap=False)
    lenXfull = comm.allreduce(len(X))

if not args.evaluateOnly:
    # Target map (by default, all these are local maps; see loaddata in case the full map is needed)
    if comm.rank==0:
        print("Loading target map...")
        sys.stdout.flush()
    if args.target == 'dm':
        if not args.FastPMpath:
            raise Exception('"dm" mode allowed only to improve FastPM, FastPMpath not specified.')
        targetmap = loaddata(args.TNGpath, args.snapNum, args.nChunks, args.target, pm)
    elif args.target in ['Mstar', 'nHI']:
        # TODO: modify nHI for different range
        targetmap = loaddata(args.TNGpath, args.snapNum, args.nChunks, args.target, pm)
    elif args.target == 'kSZ':
        targetmap = loaddata(args.TNGpath, args.snapNum, args.nChunks, 'ne', pm)
        targetmap *= 1e5 # Values are too small. Multiply the field by 1e5.
    elif args.target in ['tSZ_ne', 'Xray_ne']:
        targetmap_ne = loaddata(args.TNGpath, args.snapNum, args.nChunks, 'ne', pm)
        targetmap = loaddata(args.TNGpath, args.snapNum, args.nChunks, 'nT', pm)
        select = targetmap_ne <= 0
        targetmap_T = targetmap / targetmap_ne
        targetmap_T[select] = 0
        targetmap_ne *= 1e5 # Values are too small. Multiply the field by 1e5.
        targetmap_T *= 1e-5 # Values are too large. Multiply the field by 1e-5.
        if args.target == 'Xray_ne':
            targetmap = targetmap_ne**2 * targetmap_T**0.5 
        bias = comm.allreduce(targetmap_ne.sum()) / lenXfull 
        del targetmap_ne
    elif args.target in ['tSZ_T', 'Xray_T']:
        targetmap_ne = loaddata(args.TNGpath, args.snapNum, args.nChunks, 'ne', pm)
        targetmap_ne *= 1e5 # Values are too small. Multiply the field by 1e5.
        targetmap = loaddata(args.TNGpath, args.snapNum, args.nChunks, 'nT', pm)
        select = targetmap_ne <= 0
        map_T = targetmap / targetmap_ne
        map_T[select] = 0
        bias = comm.allreduce(map_T.sum()) / lenXfull 
        if args.target == 'Xray_T':
            targetmap = targetmap_ne**1.5 * targetmap**0.5 
        del targetmap_ne, map_T

        params = np.loadtxt(args.nePretrainParams)
        assert len(params) % 5 == 3
        Nstep = len(params) // 5
        model = LDLModel(X, None, pm)
        map_ne = model.LDL(params, Nstep=Nstep, baryon=True)
        del model
    
    if comm.rank==0:
        print("Done.")
        sys.stdout.flush()

    # Split among training, validation and test set
    index = np.array(pm.local_mesh_indices()) # Unflattened local mesh indices
    Nmesh_test = int(0.44 * args.Nmesh)
    Nmesh_validate = args.Nmesh - Nmesh_test
    select_test = (index[:,:,:,0]<Nmesh_test) & (index[:,:,:,1]<Nmesh_test) & (index[:,:,:,2]<Nmesh_test)
    # Original validation mask was:
    #select_validate = (index[:,:,:,0]>=Nmesh_test) & (index[:,:,:,1]<Nmesh_validate) & (index[:,:,:,2]<Nmesh_validate)
    # But I believe it's wrong (in this case it's a 114.8**3 subbox, corresponding to 17.6% of whole box
    select_validate = (index[:,:,:,0]<Nmesh_test) & (index[:,:,:,1]>Nmesh_test) & (index[:,:,:,2]<Nmesh_validate)
    
    # Local mask
    mask_train = np.ones_like(targetmap, dtype='bool')
    mask_validate = np.zeros_like(targetmap, dtype='bool')
    mask_test = np.zeros_like(targetmap, dtype='bool')
    mask_train[select_test] = False
    mask_train[select_validate] = False
    mask_validate[select_validate] = True
    mask_test[select_test] = True
    mask_train = jnp.array(mask_train, dtype=float)
    mask_validate = jnp.array(mask_validate, dtype=float)
    
    #build LDL model
    if args.target == 'dm':
        baryon = False
        L1 = False  # L2 works better for dm
    else:
        baryon = True 
        L1 = True

    if args.target == 'tSZ_ne':
        model = LDLModel(X, targetmap, pm, Nstep=args.Nstep, n=args.n, index=1., baryon=baryon, masktrain=mask_train, maskvalid=mask_validate)
    elif args.target == 'tSZ_T':
        model = LDLModel(X, targetmap, pm, Nstep=args.Nstep, n=args.n, index=1., baryon=baryon, masktrain=mask_train, maskvalid=mask_validate, field2=map_ne)
    elif args.target == 'Xray_ne':
        model = LDLModel(X, targetmap, pm, Nstep=args.Nstep, n=args.n, index=2., baryon=baryon, masktrain=mask_train, maskvalid=mask_validate, field2=targetmap_T**0.5)
    elif args.target == 'Xray_T':
        model = LDLModel(X, targetmap, pm, Nstep=args.Nstep, n=args.n, index=.5, baryon=baryon, masktrain=mask_train, maskvalid=mask_validate, field2=map_ne**2)
    else:
        model = LDLModel(X, targetmap, pm, Nstep=args.Nstep, n=args.n, baryon=baryon, masktrain=mask_train, maskvalid=mask_validate)
    
    if comm.rank==0:
        print("Model created.")
        sys.stdout.flush()
    
    # Initial guess
    if args.pretrainParams: # Can also improve existing parameters
        params0 = np.loadtxt(args.pretrainParams)
        if args.target == 'dm':
            assert len(params0) == 5 * args.Nstep
        else:
            assert len(params0) == 5 * args.Nstep + 3
    else:
        params0 = [0.01, 0.5, 1., 8., 0.] * args.Nstep
        if baryon:
            if args.target in ['tSZ_ne', 'tSZ_T', 'Xray_ne', 'Xray_T']:
                params0 += [1., bias, 0.]
            else:
                params0 += [1., comm.allreduce(targetmap.sum()) / lenXfull, 0.]
        params0 = jnp.array(params0)
    
    bounds = [(None, None), (0.05,2), (0.03,2*np.pi*args.Nmesh/205.), (0.03,2*np.pi*args.Nmesh/205.), (-4.5,4.5)] * args.Nstep
    if baryon:
        bounds += [(0.1,None), (0., None), (None, None)]
    
    # Train
    #optimizer = RMSProp(Nparams=len(params0), LR=0.05)
    optimizer = optax.adam(0.001)
    optstate = optimizer.init(params0)
    params = params0
    bestvloss = 1000.
    if comm.rank==0:
        print("Beginning of training.")
        sys.stdout.flush()
    for i in range(200):
        for j in range(len(params)): # Apply bounds to parameters
            if bounds[j][0] is not None:
                if params[j] < bounds[j][0]:
                    params = params.at[j].set(bounds[j][0])
            if bounds[j][1] is not None:
                if params[j] > bounds[j][1]:
                    params = params.at[j].set(bounds[j][1])
        stime = time.time()
        tloss, vloss, tgrad = model.lossv_and_grad(params)
        if vloss < bestvloss:
            bestvloss = vloss
            bestparams = params
        #params = optimizer.step(params, tgrad)
        updates, optstate = optimizer.update(tgrad, optstate)
        params = optax.apply_updates(params, updates)
        if comm.rank==0:
            print(f"Step {i+1}:")
            print(f"Train loss = {tloss:.5f}, validation loss = {vloss:.5f}, max gradient = {np.max(np.abs(tgrad)):.5f}, completed in {(time.time()-stime):.3f}s")
            print(tgrad)
            sys.stdout.flush()

    if comm.rank == 0:
        print('Finished optimization.')
        sys.stdout.flush()
        paramsave = args.save + '/' + args.target + '_snap' + str(args.snapNum).zfill(3) + '_Nmesh%d_Nstep%d_n%.2f_bestparam.txt' % (args.Nmesh, args.Nstep, args.n)
        np.savetxt(paramsave, params)
        print('Best validation loss:', bestvloss)
        print('Best parameters:', bestparams)
    params = bestparams
    #del model

else:
    param = np.loadtxt(args.pretrainParams)
    if args.target == 'dm':
        assert len(param) == 5 * args.Nstep
        baryon = False
    else:
        assert len(param) == 5 * args.Nstep + 3
        baryon = True

# Evaluate
if comm.rank == 0:
    print('Evaluating output.')
    sys.stdout.flush()

model = LDLModel(None, None, pm)
outmap = model.LDL(params, X, args.Nstep, baryon)

if comm.rank == 0:
    print('Done. Saving field.')
    sys.stdout.flush()

if args.target in ['dm', 'Mstar', 'nHI']:
    save = args.save + '/' + args.target + '_snap' + str(args.snapNum).zfill(3) + '_Nmesh%d_Nstep%d_n%.2f_map' % (args.Nmesh, args.Nstep, args.n)
    psavefield(outmap, save, pm)

"""elif args.target == 'kSZ':
    LDLmap *= 1e-5 #The learned LDL map is actually 1e5*map_ne.
    save = args.save + '/ne_snap' + str(args.snapNum).zfill(3) + '_Nmesh%d_Nstep%d_n%.2f_map' % (args.Nmesh, args.Nstep, args.n)
    FieldMesh(LDLmap).save(save)
    if args.FastPMpath:
        X = File(args.FastPMpath)['Position']
        X = np.array(X[start:end]).astype(np.float32)
        Vz = File(args.FastPMpath)['Velocity']
        Vz = np.array(Vz[start:end])[:,2].astype(np.float32)
    else:
        from readTNG import scalefactor
        a = scalefactor(args.TNGDarkpath, args.snapNum)
        Vz = load_TNG_data(TNG_basepath=args.TNGDarkpath, snapNum=args.snapNum, partType='dm', field='Velocities', mdi=2) * a**0.5
    layout = pm.decompose(X)
    X = layout.exchange(X)
    Vz = layout.exchange(Vz)
    map_vz = pm.create(type="real")
    map_vz.paint(X, mass=Vz, layout=None, hold=False)
    map_delta = pm.create(type="real")
    map_delta.paint(X, mass=1., layout=None, hold=False)
    select = map_delta > 0
    map_vz[select] = map_vz[select] / map_delta[select]
    map_vz[~select] = 0
    
    LDLmap = LDLmap * map_vz
    save = args.save + '/' + args.target + '_snap' + str(args.snapNum).zfill(3) + '_Nmesh%d_Nstep%d_n%.2f_map' % (args.Nmesh, args.Nstep, args.n)
    FieldMesh(LDLmap).save(save)

elif args.target in ['tSZ_T', 'Xray_T']:
    param = np.loadtxt(args.restore_ne)
    assert len(param) % 5 == 3
    Nstep = len(param) // 5
    model = LDL.build(X=X, pm=pm, Nstep=Nstep, baryon=True)
    map_ne = model.compute('F', init=dict(param=param))
    del model

    if args.target == 'tSZ_T':
        LDLmap = LDLmap * map_ne #the 1e5 factor in ne map and 1e-5 factor in T map cancels.
        save = args.save + '/tSZ_snap' + str(args.snapNum).zfill(3) + '_Nmesh%d_Nstep%d_n%.2f_map' % (args.Nmesh, args.Nstep, args.n)
    elif args.target == 'Xray_T':
        LDLmap = (LDLmap*1e5)**0.5 * (map_ne*1e-5)**2
        save = args.save + '/Xray_snap' + str(args.snapNum).zfill(3) + '_Nmesh%d_Nstep%d_n%.2f_map' % (args.Nmesh, args.Nstep, args.n)
    FieldMesh(LDLmap).save(save)"""


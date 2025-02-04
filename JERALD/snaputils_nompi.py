# ADAPTED FROM PYLIANS: https://github.com/franciscovillaescusa/Pylians/tree/master

# Utils to import fields and paint maps from snapshots saved in hdf5 format, not parallelized

import numpy as np
import MAS_library as MASL
import sys,os,h5py
import readgadget, readsnap
from maputils import savefield

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from pmesh import PMesh


# Name of the 1st chunk in the snapshot (adapted from Pylians)
def fname(path):
    if os.path.exists(path):
        if path[-4:] == 'hdf5': # Full path given
            filename = path
        else: # Likely binary, but unsupported anyway
            raise Exception('Unsupported file format')
    elif os.path.exists(path + '.0'): # Binary file given
        raise Exception('Binary type not supported')
    elif os.path.exists(path + '.hdf5'): # Chunk number given
        if path[-2:] != '.0': # Chunk number provided but not the 1st
            parts = path.split('.')[:-1]
            parts.append('.0')
            path = ''.join(parts)
        filename = path + '.hdf5'
    elif os.path.exists(path + '.0.hdf5'): # Only snap number given
        filename = path + '.0.hdf5'
    else:
        raise Exception('File not found')
    return filename


# Adapted from illustris_python (only for JERALD targets)
def partTypeNum(kind):
    # 0: gas, 1: dm, 3: tracers, 4: stars/wind, 5: bhs
    if kind == 'dm':
        ptName = "PartType1" # To be consistent with original TNG import lib
        ptNum = 1
    elif kind in ['Mstar', 'MstarVz']:
        ptName = "PartType4"
        ptNum = 4
    elif kind in ['nHI', 'XHI']:
        ptName = "PartType0"
        ptNum = 0
    else:
        raise Exception(f"Unkown required map type {kind}")

    return ptName, ptNum


# Adapted from readgadget, to also include neutral hydrogen abundance
def read_field_custom(snapshot, block, ptype):

    filename, fformat = readgadget.fname_format(snapshot)
    head              = readgadget.header(filename)
    Masses            = head.massarr*1e10 #Msun/h
    Npart             = head.npart        #number of particles in the subfile
    Nall              = head.nall         #total number of particles in the snapshot

    if fformat=="binary":
        return readsnap.read_block(filename, block, parttype=ptype)
    else:
        prefix = 'PartType%d/'%ptype
        f = h5py.File(filename, 'r')
        if   block=="POS ":  suffix = "Coordinates"
        elif block=="MASS":  suffix = "Masses"
        elif block=="ID  ":  suffix = "ParticleIDs"
        elif block=="VEL ":  suffix = "Velocities"
        elif block=="XHI ":  suffix = "NeutralHydrogenAbundance"
        else: raise Exception('Block not implemented in readgadget!')

        if '%s%s'%(prefix,suffix) not in f:
            print(f"{prefix}{suffix} not found in snapshot, using unit mass from header")
            if Masses[ptype] != 0.0:
                array = np.ones(Npart[ptype], np.float32)*Masses[ptype]
            else:
                raise Exception(f'Problem reading block {block}, maybe ptype {ptype} does not have it?')
        else:
            array = f[prefix+suffix][:]
        f.close()

        if block=="VEL ":  array *= np.sqrt(head.time)
        if block=="POS " and array.dtype==np.float64:
            array = array.astype(np.float32)

        return array


def makemap(path, kind, pm, f32=True, save=True, outpath=None, verbose=2, **kwargs):
    """
    Loads GADGET data, paints map of kind "kind", optionally saves it and returns it

    Parameters:
        path (str): path to snapshot files (Pylians like, only prefix is needed)
            e.g. "some/path/snapdir_004/snap_004"
        kind (str): what to load (see below for possibilities)
        pm (PMesh object): Particle mesh (used for mesh specs only)
        f32 (bool): whether to use float32 or float64
        save (bool): whether to save the map
        outpath (str): optional path to save map
        verbose (int): different levels of verbose (0: quiet, 1: kinda verbose, 2: very verbose)
        **kwargs (dict): key-worded arguments:
            'h' (float): Hubble constant (reduced)
            'XH' (float): hydrogen mass fraction

    Returns:
        "kind" map
    """
    
    filename, _ = readgadget.fname_format(path)
    head = readgadget.header(path)
    Nall = head.nall # Total number of particles in the snapshot per ptype
    redshift = head.redshift
    MAS = 'CIC'
    BoxSize = np.array(pm.BoxSize)
    Nmesh = np.array(pm.Nmesh)

    if verbose > 0:
        print(f"Snapshot box size: {head.boxsize}")
        print(f"Redshift: {redshift}")

    h = kwargs.get('h', 0.678) # Get h from kwargs if passed, otherwise use Sherwood-Relics one
    XH = kwargs.get('XH', 0.76) # As above with hydrogen mass fraction

    mp = 1.6726219e-27
    Msun10 = 1.989e40 # (*1e10)
    Mpc_cm = 3.085678e24
    kb = 1.38064852e-23

    a = 1 / (1 + head.redshift)

    # NOTE: the factors above are just there for understandability.
    # I define specific coefficients below to work with numbers around unit.
    # Msun10/h / (a*BoxSize/Nmesh*Mpc_cm/h)**3 * XH/mp --- for number densities
    coeff1 = 4.0475e-07 * h**2 / a**3 * XH / BoxSize.prod() * Nmesh.prod()

    if verbose > 0:
        print("Importing required data and paiting...")

    # Info needed to read subset of field (offset table)
    Nchunks = head.filenum
    # Simply loop over all subfiles in the snapshot
    outmap = np.zeros(Nmesh, dtype=np.float32)
    for i in range(Nchunks):

        # Find the name of the subfile
        if verbose == 2:
            print(f"Reading chunk {i}.")
            sys.stdout.flush()
        chunkfname = f'{path}.{i}'

        # Read required fields
        if kind == 'dm':    
            pos = readgadget.read_field(chunkfname, "POS ", 1)/1e3 # 1 is CDM ptype
            # Add to 3D density field
            MASL.MA(pos, outmap, BoxSize[0], MAS, verbose=False)
        elif kind == 'Mstar':
            pos = readgadget.read_field(chunkfname, "POS ", 4)/1e3 # 4 is star ptype
            mass = read_field_custom(chunkfname, "MASS", 4)
            MASL.MA(pos, outmap, BoxSize[0], MAS, W=mass, verbose=False)
        elif kind == 'nHI':
            pos = readgadget.read_field(chunkfname, "POS ", 0)/1e3 # 0 is gas ptype
            mass = readgadget.read_field(chunkfname, "MASS", 0)
            XHI = read_field_custom(chunkfname, "XHI ", 0)
            mass = coeff1 * mass * XHI
            MASL.MA(pos, outmap, BoxSize[0], MAS, W=mass, verbose=False)
        elif kind == 'XHI':
            pos = readgadget.read_field(chunkfname, "POS ", 0)/1e3 # 0 is gas ptype
            XHI = read_field_custom(chunkfname, "XHI ", 0)
            MASL.MA(pos, outmap, BoxSize[0], MAS, W=XHI, verbose=False)
        else:
            raise Exception(f"Kind {kind} unknown or not implemented")

    if kind == 'dm':
        outmap *= (Nmesh**3 / Nall[1])

    if save:
        if verbose > 0:
            pm.printr("Done. Saving map...")
        if not outpath:
            outpath = '/'.join(filename.split('/')[:-1]) # Snapshot folder
        outfile = f'{outpath}/{kind}map_Nmesh{Nmesh[0]}'
        savefield(outmap, outfile, pm)
        if verbose > 0:
            pm.printr("Saved at {outfile}.")
    else:
        if verbose > 0:
            pm.printr("Done.")

    return np.array(outmap)


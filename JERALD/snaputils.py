"""
Copyright (c) 2017, illustris & illustris_python developers All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those of the authors and should not be interpreted as representing official policies, either expressed or implied, of the FreeBSD Project.
"""
# ADAPTED FROM PYLIANS: https://github.com/franciscovillaescusa/Pylians/tree/master
# AND https://github.com/illustristng/illustris_python
# Does not deal with binary, only hdf5 files

# Utils to import fields and paint maps from snapshots saved in hdf5 format, parallelized

import numpy as np
import sys,os,h5py
from mpi4py import MPI
from maputils import savefield

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from pmesh import PMesh


# Just for readability if run in parallel
def printr(string, comm=MPI.COMM_WORLD):
    if comm.rank==0:
        print(string)
        sys.stdout.flush()


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


def mapPartNum(kind):
    # Returns particle type number for different kinds of maps (also LDL ones)
    # 0: gas, 1: dm, 3: tracers, 4: stars/wind, 5: bhs
    if kind == 'dm':
        ptNum = 1
    elif kind in ['Mstar', 'MstarVz']:
        ptNum = 4
    elif kind in ['ne', 'nT', 'nHI', 'neVz', 'XHI']:
        ptNum = 0
    else:
        raise Exception(f"Unkown required map type {kind}")

    return ptNum


# This class reads the header of the gadget file
# Only hdf5 (adapted from Pylians)
class Header:
    def __init__(self, snapshot):

        filename = fname(snapshot)

        f             = h5py.File(filename, 'r')
        self.time     = f['Header'].attrs[u'Time']
        self.redshift = f['Header'].attrs[u'Redshift']
        self.boxsize  = f['Header'].attrs[u'BoxSize']
        self.filenum  = f['Header'].attrs[u'NumFilesPerSnapshot']
        self.omega_m  = f['Header'].attrs[u'Omega0']
        self.omega_l  = f['Header'].attrs[u'OmegaLambda']
        self.hubble   = f['Header'].attrs[u'HubbleParam']
        self.massarr  = f['Header'].attrs[u'MassTable']
        self.npart    = f['Header'].attrs[u'NumPart_ThisFile']
        self.nall     = f['Header'].attrs[u'NumPart_Total']
        self.cooling  = f['Header'].attrs[u'Flag_Cooling']
        f.close()

        # km/s/(Mpc/h)
        self.Hubble = 100.0*np.sqrt(self.omega_m*(1.0+self.redshift)**3+self.omega_l)


def buildOffsetTable(path, nChunks):
    """ Make the offset table (by type) for the snapshot files,
        to be able to quickly determine within which chunk(s) a given offset+length will exist.
        Contains, for each ptype and in each chunk, the starting index of the particles of that type
        Could also read nChunks here instead of having it as an arugment but doesn't change much
    """
    path = fname(path) # Format correctly
    chunkoffsets = np.zeros((6, nChunks), dtype='int64') # 6: All particle types
    for i in range(1, nChunks):
        chunkfname = path[:-6] + str(i-1) + '.hdf5'
        f = h5py.File(chunkfname, 'r')
        for j in range(6):
            # Index in prev chunk + num parts in prev chunk
            chunkoffsets[j, i] = chunkoffsets[j, i-1] + f['Header'].attrs['NumPart_ThisFile'][j]
        f.close()

    return chunkoffsets


# Read subset (if parallelized) of input fields
# Mix between Pylians and illustris code
def readfields(path, ptNum, fields, mdi, comm=MPI.COMM_WORLD, f32=True, verbose=2):
    """
    Reads fields in a snapshot for particle type "ptNum"

    Parameters:
        path (str): path to snapshot files (Pylians like, only prefix is needed)
            e.g. "some/path/snapdir_004/snap_004"
        ptNum (str): particle type number (0: gas, 1: dm, 3: tracers, 4: stars/wind, 5: bhs)
        fields (list of str): fields to read for particles (e.g. Coordinates, Velocities, ...)
        mdi: (None or list of None or int): None for 1D fields or dimension index for 3D fields if only
            specific dimension is needed (e.g. vz -> field = ["Velocities"], mdi = [2])
        comm: MPI communicator
        f32 (bool): whether to use float32 or float64
        verbose (int): different levels of verbose (0: quiet, 1: kinda verbose, 2: very verbose)

    Returns:
        dict containing different local fields as np arrays, named with the name of the field
    """

    filename = fname(path)
    head = Header(filename)
    Nall = head.nall # Total number of particles in the snapshot per ptype
    redshift = head.redshift

    if verbose > 0:
        printr(f"Snapshot box size: {head.boxsize}", comm)
        printr(f"Redshift: {redshift}", comm)
    
    # Info needed to read subset of field (offset table)
    Nchunks = head.filenum
    snapoffsets = buildOffsetTable(filename, Nchunks)
    ptName = f"PartType{ptNum}"

    # Check if particle type exists
    if Nall[ptNum]==0:
        raise Exception(f"No particles of type {ptype} in {filename}")
    if verbose > 0:
        printr(f"To load {Nall[ptNum]} particles in total", comm)

    # !!! LOCAL !!!
    # Particles indices to load into this rank
    start = (comm.rank * Nall[ptNum]) // comm.size
    end = ((comm.rank+1) * Nall[ptNum]) // comm.size
    offsets = start - snapoffsets[ptNum, :]
    filen = np.max(np.where(offsets >= 0)) # Chunk that contains particles from index "start"
    fileo = offsets[filen] # Offset of "start" relative to the first index in said chunk
    Ntoread = end - start # Total number of particles this rank reads
    if verbose == 2:
        print(f"Rank {comm.rank} reading {Ntoread} particles starting from index {start}.")
        sys.stdout.flush()
    comm.Barrier()

    out = {}
    
    # Prepare out arrays
    localmem = 0 # Memory requested for this rank
    startfname = filename[:-6] + str(filen) + '.hdf5'
    with h5py.File(startfname, 'r') as f:
        
        for i, field in enumerate(fields):
            # Verify existence
            if field not in f[ptName].keys():
                raise Exception(f"Particle type {ptNum} does not have field {field}")

            # Fields are (nparts, dimfield), where dimfield is like 3 for positions e.g.
            # Replace single file length with the one requested for this rank
            shape = list(f[ptName][field].shape)
            shape[0] = Ntoread

            # Multi-dimensional index slice load
            if mdi is not None and mdi[i] is not None:
                if len(shape) != 2:
                    raise Exception(f"Read error: mdi requested on 1D field {field}")
                shape = [shape[0]] # Because it's a n-dimensional field but we're requesting specific dim

            # Allocate
            dtype = f[ptName][field].dtype
            if dtype == np.float64 and f32: dtype = np.float32
            fielddata = np.zeros(shape, dtype=dtype)
            localmem += fielddata.nbytes / 1024 / 1024
            out[field] = fielddata

    if verbose == 2:
        totalmem = comm.allreduce(localmem, op=MPI.SUM)
        printr(f"Rank {comm.rank} allocating {localmem:.2f}MB for fields, total {totalmem:.2f}MB.", comm)

    # Loop over chunks
    writeoffset = 0
    Nleft = Ntoread

    while Nleft > 0:
        if verbose == 2:
            print(f"Rank {comm.rank} reading chunk {filen}.")
            sys.stdout.flush()
        chunkfname = filename[:-6] + str(filen) + '.hdf5'
        f = h5py.File(chunkfname, 'r') # Start from file that contains start index

        # No particles of requested type in this file chunk?
        if ptName not in f:
            f.close()
            filen += 1
            fileo  = 0
            continue

        # Set local read length for this file chunk, truncate to be within the local size
        Nlocal = f['Header'].attrs['NumPart_ThisFile'][ptNum]
        Ntoreadlocal = Nlocal - fileo # For first file, only read particles after start index
        if Ntoreadlocal > Nleft:
            Ntoreadlocal = Nleft

        # Loop over each requested field for this particle type
        for i, field in enumerate(fields):
            # Read required data from this chunk
            if mdi is None or mdi[i] is None:
                out[field][writeoffset:writeoffset+Ntoreadlocal] =\
                        f[ptName][field][fileo:fileo+Ntoreadlocal]
            else:
                out[field][writeoffset:writeoffset+Ntoreadlocal] =\
                        f[ptName][field][fileo:fileo+Ntoreadlocal, mdi[i]]

        writeoffset += Ntoreadlocal
        Nleft -= Ntoreadlocal
        filen += 1
        fileo = 0  # Start at beginning of all file chunks other than the first

        f.close()

    # Verify we read the correct number
    if Ntoread != writeoffset:
        raise Exception(f"Read {writeoffset} particles, but was expecting {Ntoread}")

    return out


# NOTE: could also make it so that the map is accumulated and only fields local to each file
# are stored when a chunk is checked, this would avoid saving the whole field (not map) for each rank
def makemap(path, kind, pm, f32=True, local=True, save=True, outpath=None, verbose=2, **kwargs):
    """
    Loads snapshot data in parallel, paints map of kind "kind", optionally saves it and returns it

    Parameters:
        path (str): path to snapshot files (Pylians like, only prefix is needed)
            e.g. "some/path/snapdir_004/snap_004"
        kind (str): what to load (see below for possibilities)
        pm (PMesh object): Particle mesh (used for painting and mesh specs)
        f32 (bool): whether to use float32 or float64
        local (bool): whether to return only the local part of the field given the particle mesh pm
        save (bool): whether to save the map
        outpath (str): optional path to save map
        verbose (int): different levels of verbose (0: quiet, 1: kinda verbose, 2: very verbose)
        **kwargs (dict): key-worded arguments:
            'h' (float): Hubble constant (reduced)
            'XH' (float): hydrogen mass fraction

    Returns:
        "kind" map, optionally local
    """
    Nmesh = pm.Nmesh
    BoxSize = pm.BoxSize
    comm = pm.comm

    # Require fields to load (can add more)
    fields = ['Coordinates'] # Loads coordinates for all
    mdi = [None]

    # Including LDL quantities for completeness
    if kind == 'Mstar':
        fields.append('Masses')
        mdi.append(None)
    elif kind == 'ne':
        fields.append('Masses')
        mdi.append(None)
        fields.append('ElectronAbundance')
        mdi.append(None)
    elif kind == 'nT':
        fields.append('ElectronAbundance')
        mdi.append(None)
        fields.append('InternalEnergy')
        mdi.append(None)
        fields.append('Masses')
        mdi.append(None)
    elif kind == 'nHI':
        fields.append('Masses')
        mdi.append(None)
        fields.append('NeutralHydrogenAbundance')
        mdi.append(None)
    elif kind == 'neVz':
        fields.append('Masses')
        mdi.append(None)
        fields.append('ElectronAbundance')
        mdi.append(None)
        fields.append('Velocities')
        mdi.append(2) # Vz
    elif kind == 'MstarVz':
        fields.append('Masses')
        mdi.append(None)
        fields.append('Velocities')
        mdi.append(2)
    elif kind == 'XHI':
        fields.append('NeutralHydrogenAbundance')
        mdi.append(None)
    elif kind != 'dm':
        raise Exception(f"Unknown kind to import {kind}")

    if verbose > 0:
        printr("Importing required data...", comm)
    comm.Barrier()

    ptNum = mapPartNum(kind)
    outs = readfields(path, ptNum, fields, mdi, comm, f32, verbose)

    comm.Barrier()
    if verbose > 0:
        printr("Import finished. Computing quantities...", comm)

    h = kwargs.get('h', 0.678) # Get h from kwargs if passed, otherwise use Sherwood-Relics one
    XH = kwargs.get('XH', 0.76) # As above with hydrogen mass fraction

    mp = 1.6726219e-27
    Msun10 = 1.989e40 # (*1e10)
    Mpc_cm = 3.085678e24
    kb = 1.38064852e-23

    filename = fname(path)
    head = Header(filename)
    a = 1 / (1 + head.redshift)

    # NOTE: the factors above are just there for understandability.
    # I define specific coefficients below to work with numbers around unit.
    # Msun10/h / (a*BoxSize/Nmesh*Mpc_cm/h)**3 * XH/mp --- for number densities
    coeff1 = 4.0475e-07 * h**2 / a**3 * XH / BoxSize.prod() * Nmesh.prod()
    # 2./3. * 1e6 / kb * mp * 4 --- 1e6 comes from (km/s)^2 -> (m/s)^2 for u --- for temperature
    coeff2 = 323.06014

    # Define quantities to do CIC on
    if kind == 'ne':
        # Msun10/h / (a*BoxSize/Nmesh*Mpc_cm/h)**3 * mass/mp*XH*Xe
        mass = coeff1 * outs['Masses'] * outs['ElectronAbundance']
    elif kind == 'nT':
        # Msun10/h / (a*BoxSize/Nmesh*Mpc_cm/h)**3 * mass/mp*XH*Xe * T
        # T = 2./3. * 1e6 * u / kb * 4./(1.+3.*XH+4.*XH*Xe) * mp
        mass = coeff1 * outs['Masses'] * outs['ElectronAbundance'] *\
            coeff2 * outs['InternalEnergy'] / (1 + 3*XH + 4*XH*outs['ElectronAbundance']) 
    elif kind == 'nHI':
        # Msun10/h / (a*BoxSize/Nmesh*Mpc_cm/h)**3 * mass/mp*XH*XHI
        mass = coeff1 * outs['Masses'] * outs['NeutralHydrogenAbundance']
    elif kind == 'neVz':
        # Msun10/h / (a*BoxSize/Nmesh*Mpc_cm/h)**3 * mass/mp*XH/Xe * vz * a**0.5
        mass = coeff1 * outs['Masses'] * outs['ElectronAbundance'] * outs['Velocities'] * a**0.5
    elif kind == 'MstarVz':
        mass = outs['Masses'] * outs['Velocities'] * a**0.5
    elif kind == 'XHI':
        mass = outs['NeutralHydrogenAbundance']
    
    pos = outs['Coordinates']
    pos /= 1000. # Positions to Mpc
    pos %= BoxSize # PBC
    del outs

    comm.Barrier()
    if verbose > 0:
        printr("Done. Painting map...", comm)

    lenpos = len(pos)
    # Normalization for DM overdensity+1
    if kind == 'dm':
        mass = 1.0 * Nmesh.prod() / comm.allreduce(lenpos, op=MPI.SUM)

    outmap = pm.paint(pos, mass)
    del pos, mass
    comm.Barrier()

    if save:
        if verbose > 0:
            printr("Done. Saving map...", comm)
        if not outpath:
            outpath = '/'.join(filename.split('/')[:-1]) # Snapshot folder 
        outfile = f'{outpath}/{kind}map_Nmesh{Nmesh[0]}'
        savefield(outmap, outfile, pm)
        if verbose > 0:
            printr("Saved at {outfile}.", comm)
    else:
        if verbose > 0:
            printr("Done.", comm)

    if local:
        return np.array(outmap)
    else:
        return pm.preview(outmap)


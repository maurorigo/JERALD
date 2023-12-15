""" Part of this code is adapted from https://github.com/illustristng/illustris_python

Copyright (c) 2017, illustris & illustris_python developers All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those of the authors and should not be interpreted as representing official policies, either expressed or implied, of the FreeBSD Project.
"""

# Imports TNG data and optionally paints it (if no paint, just imports positions)
import numpy as np
import h5py
from mpi4py import MPI
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from pmesh import PMesh
from fieldIO import psavefield, preadfield
import argparse

# Adapted from illustris_python
def partTypeNum(kind):
    # 0: gas, 1: dm, 3: tracers, 4: stars/wind, 5: bhs
    if kind == 'dm':
        gName = "PartType1" # To be consistent with original TNG import lib
        ptNum = 1
    elif kind in ['Mstar', 'MstarVz']:
        gName = "PartType4"
        ptNum = 4
    elif kind in ['ne', 'nT', 'nHI', 'neVz']:
        gName = "PartType0"
        ptNum = 0
    else:
        raise Exception("Unkown required map type " + kind)

    return gName, ptNum

# From illustris_python
def snapPath(basepath, snapNum, chunkNum=0):
    """ Return absolute path to a snapshot HDF5 file (modify as needed). """
    snapPath = basepath + '/snapdir_' + str(snapNum).zfill(3) + '/'
    filePath = snapPath + 'snap_' + str(snapNum).zfill(3)
    filePath += '.' + str(chunkNum) + '.hdf5'
    return filePath

# Make the offset table (by type) for the snapshot files, to be able to quickly determine within which file(s) a given offset+length will exist. I guess this contains, for each ptype and in each chunk, the starting index of the particles of that type 
def buildOffsetTable(TNGpath, snapNum, nChunks):
    snapOffsets = np.zeros((6, nChunks), dtype='int64') # All particle types, not really needed
    for i in range(1, nChunks):
        f = h5py.File(snapPath(TNGpath, snapNum, i-1), 'r')
        for j in range(6):
            # Index in prev chunk + num parts in prev chunk
            snapOffsets[j, i] = snapOffsets[j, i-1] + f['Header'].attrs['NumPart_ThisFile'][j]
        f.close()

    return snapOffsets

def scalefactor(TNGpath, snapNum):
    # Returns the scale factor of TNG data
    with h5py.File(snapPath(TNGpath, snapNum), 'r') as f:
        header = dict(f['Header'].attrs.items())
        z = header['Redshift']
        a = 1. / (1. + z)
    return a


def loaddata(TNGpath, snapNum, nChunks, kind, pm, f32=True, CICmap=True, local=True):
    """
    Loads TNG data (optionally in parallel): if CICmap is False, just return positions,
    otherwise check if map already exists, if yes return it otherwise paint, save and return it

    Parameters:
        TNGpath (str): path to TNG file
        snapNum (int): Snapshot number
        nChunks (int): Number of chunks, aka files composing the snapshot
        kind (str): What to load, see argparse below for possibilities
        f32 (bool): Whether to use float32 or float64
        pm (PMesh object): Particle mesh (used for painting and LDL specs)
        local (bool): whether to return local or full field

    Returns:
        TNG dm particle positions if CICmap is false, "kind" map if CICmap is true
    """
    Nmesh = pm.Nmesh[0]
    BoxSize = pm.BoxSize[0]
    comm = pm.comm
    if CICmap:
        # Try to import map directly, otherwise just import all data and paint it
        address = TNGpath + '/snapdir_' + str(snapNum).zfill(3) + '/' + kind + 'map_Nmesh' + str(Nmesh)
        try:
            field, Nmesh, BoxSize = preadfield(address, comm)
            if comm.rank==0:
                print(f"{kind} map found at {address}.")
                sys.stdout.flush()
            if local:
                return field[pm.fftss[0]:pm.fftss[1], :, :]
            else:
                return field
        except:
            if comm.rank==0:
                print(f"{kind} map not found at {address}, building map and saving it...")
                sys.stdout.flush()
    else:
        # If not paint, just import positions
        kind = 'dm'
        if comm.rank==0:
            print("Requested TNG positions.")
            sys.stdout.flush()

    # Offset table
    snapOffsets = buildOffsetTable(TNGpath, snapNum, nChunks)
    # Name and number of particle to read
    gName, ptNum = partTypeNum(kind)

    # Require fields to load (can add more)
    fields = ['Coordinates'] # Loads all coordinates at once
    mdi = [None]

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

    if comm.rank == 0:
        print("Importing required data...")
        sys.stdout.flush()
    comm.Barrier()
    
    # The next lines are adapted from loadSubset in illustris_python
    with h5py.File(snapPath(TNGpath, snapNum), 'r') as f:

        header = dict(f['Header'].attrs.items())
        # Calculate number of particles of required type for snapshot header
        nPart = header['NumPart_Total'][ptNum] | (header['NumPart_Total_HighWord'][ptNum] << 32)
        if nPart==0:
            raise Exception(f"No particles of type {kind}")
        if comm.rank==0:
            print(f"To load {nPart} particles")
            sys.stdout.flush()

        # Split data into different ranks
        start = comm.rank * nPart // comm.size
        end = (comm.rank+1) * nPart // comm.size

        offsetsThisType = start - snapOffsets[ptNum, :]
        fileNum = np.max(np.where(offsetsThisType >= 0))
        fileOff = offsetsThisType[fileNum]
        numToRead = nPart // comm.size
        print(f"Rank {comm.rank} reading {numToRead} particles starting from index {start}.")
        sys.stdout.flush()
        comm.Barrier()

        i = 1
        while gName not in f: # Remember gName is str of ptype
            f = h5py.File(snapPath(TNGpath, snapNum, i), 'r')
            i += 1

        for i, field in enumerate(fields):
            # verify existence
            if field not in f[gName].keys():
                raise Exception(f"Particle type {ptNum} does not have field {field}")

            # replace local length with global
            shape = list(f[gName][field].shape)
            shape[0] = numToRead

            # multi-dimensional index slice load
            if mdi is not None and mdi[i] is not None:
                if len(shape) != 2:
                    raise Exception(f"Read error: mdi requested on non-2D field {field}")
                shape = [shape[0]]

            # allocate within return dict
            dtype = f[gName][field].dtype
            if dtype == np.float64 and f32: dtype = np.float32
            if field=='Coordinates':
                pos = np.zeros(shape, dtype=dtype)
                nMB = round(pos.nbytes / 1024 / 1024, 2)
                totalnMB = comm.allreduce(nMB, op=MPI.SUM)
                if comm.rank==0:
                    # Just some info
                    print(f"Allocating {nMB}MB for positions, for a total of {totalnMB}MB requested.")
                    sys.stdout.flush()
            elif field=='Masses':
                mass = np.zeros(shape, dtype=dtype)
            elif field=='ElectronAbundance':
                Xe = np.zeros(shape, dtype=dtype)
            elif field=='InternalEnergy':
                u = np.zeros(shape, dtype=dtype)
            elif field=='NeutralHydrogenAbundance':
                XHI = np.zeros(shape, dtype=dtype)
            elif field=='Velocities':
                vz = np.zeros(shape, dtype=dtype)

    # loop over chunks
    wOffset = 0
    origNumToRead = numToRead

    while numToRead:
        f = h5py.File(snapPath(TNGpath, snapNum, fileNum), 'r')

        # no particles of requested type in this file chunk?
        if gName not in f:
            f.close()
            fileNum += 1
            fileOff  = 0
            continue

        # set local read length for this file chunk, truncate to be within the local size
        numTypeLocal = f['Header'].attrs['NumPart_ThisFile'][ptNum]

        numToReadLocal = numToRead

        if fileOff + numToReadLocal > numTypeLocal:
            numToReadLocal = numTypeLocal - fileOff

        # loop over each requested field for this particle type
        for i, field in enumerate(fields):
            # read data local to the current file
            if field=='Coordinates': # Use fields aready defined
                pos[wOffset:wOffset+numToReadLocal] = f[gName][field][fileOff:fileOff+numToReadLocal]
            elif field=='Masses':
                mass[wOffset:wOffset+numToReadLocal] = f[gName][field][fileOff:fileOff+numToReadLocal]
            elif field=='ElectronAbundance':
                Xe[wOffset:wOffset+numToReadLocal] = f[gName][field][fileOff:fileOff+numToReadLocal]
            elif field=='InternalEnergy':
                u[wOffset:wOffset+numToReadLocal] = f[gName][field][fileOff:fileOff+numToReadLocal]
            elif field=='NeutralHydrogenAbundance':
                XHI[wOffset:wOffset+numToReadLocal] = f[gName][field][fileOff:fileOff+numToReadLocal]
            elif field=='Velocities':
                vz[wOffset:wOffset+numToReadLocal] = f[gName][field][fileOff:fileOff+numToReadLocal, mdi[i]]

        wOffset   += numToReadLocal
        numToRead -= numToReadLocal
        fileNum   += 1
        fileOff    = 0  # start at beginning of all file chunks other than the first

        f.close()

    # verify we read the correct number
    if origNumToRead != wOffset:
        raise Exception(f"Read {wOffset} particles, but was expecting {origNumToRead}")

    pos /= 1000. # To Mpc
    pos %= BoxSize # PBC
    # FINISHED LOCAL LOADSUBSET

    comm.Barrier()
    if comm.rank == 0:
        print("Import finished.")
        sys.stdout.flush()
    
    if not CICmap:
        # If not paint, just return positions (DIVIDED INTO RANKS)
        return pos
    
    if comm.rank == 0:
        print("Computing quantities...")
        sys.stdout.flush()

    h = 0.6774 # From TNG website
    XH = 0.76 # What is this?
    mp = 1.6726219e-27
    Msun10 = 1.989e40 # (*1e10)
    Mpc_cm = 3.085678e24
    kb = 1.38064852e-23

    if kind == 'Mstar':
        foo = 1
        # Useless
    elif kind == 'ne':
        a = scalefactor(TNGpath, snapNum)
        mass = Msun10/h / (a*BoxSize/Nmesh*Mpc_cm/h)**3 * mass*XH/mp*Xe # From n_cm3
        del Xe
    elif kind == 'nT':
        ufac = 1e6 #u: (km/s)^2 -> (m/s)^2
        u = 2./3. * ufac * u / kb * 4./(1.+3.*XH+4.*XH*Xe) * mp # Actually this is temperature, but to avoid introducing multiple variables
        a = scalefactor(TNGpath, snapNum) 
        mass = Msun10/h / (a*BoxSize/Nmesh*Mpc_cm/h)**3 * mass*XH/mp*Xe * u # This is n_cm3(mss, Xe) * T = ne*T
        del Xe, u
    elif kind == 'nHI':
        a = scalefactor(TNGpath, snapNum)
        mass = Msun10/h / (a*BoxSize/Nmesh*Mpc_cm/h)**3 * mass*XH/mp*XHI # From n_cm3
        del XHI
    elif kind == 'neVz':
        a = scalefactor(TNGpath, snapNum) 
        mass = Msun10/h / (a*BoxSize/Nmesh*Mpc_cm/h)**3 * mass*XH/mp*Xe * vz * a**0.5 # This is n_cm3(mass, Xe)*vz = ne*vz
        del vz
    elif kind == 'MstarVz':
        mass *= vz * a**0.5 # Vz factor as above

    comm.Barrier()
    if comm.rank == 0:
        print("Done. Painting map...")
        sys.stdout.flush()
    lenpos = len(pos)
    # Normalization for DM overdensity+1
    if kind == 'dm':
        mass = 1.0 * Nmesh**3 / comm.allreduce(lenpos, op=MPI.SUM)

    TNGmap = pm.paint(pos, mass)
    del pos, mass
    if comm.rank == 0:
        print("Done. Saving map...")
        sys.stdout.flush()

    address = TNGpath + '/snapdir_' + str(snapNum).zfill(3) + '/' + kind + 'map_Nmesh' + str(Nmesh)
    psavefield(TNGmap, address, pm)
    
    if comm.rank == 0:
        print("Done.")
        sys.stdout.flush()

    if local:
        return np.array(TNGmap)
    else:
        return pm.preview(TNGmap)


if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description="Run as main, this program paints and saves a map if it doesn't already exist.")

    parser.add_argument('TNGpath', type=str, help='path to load in TNG particles')
    parser.add_argument('snapNum', type=int, help='snapshot number')
    parser.add_argument('nChunks', type=int, help='number of chunks for snapshot (files composing it)')
    parser.add_argument('kind', type=str, help='what to paint ("dm": DM density, "Mstar" for stellar mass, "MstarVz", "ne", "nT", "nHI", "neVz" idk for the time being)')
    parser.add_argument('Nmesh', type=int, help='number of particles per side')
    parser.add_argument('--float64', dest='float32', action='store_false', help="Import data as float64 (default is float32, saves memory)")
    parser.set_defaults(float32=True)
    parser.add_argument('--BoxSize', type=float, default=205., help='size of simulation box (default is 205.)')

    args = parser.parse_args()
    
    pm = PMesh(args.Nmesh, args.BoxSize)
    a = loaddata(args.TNGpath, args.snapNum, args.nChunks, args.kind, pm, args.float32)
    print(a)


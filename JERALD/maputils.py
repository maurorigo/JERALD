# Utils for map generation, importing and saving
import numpy as np
import bigfile
import json
from mpi4py import MPI
import h5py

# These are here just to be able to import pmesh from a parent folder without using __init__.py files
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from pmesh import PMesh

__all__ = ["psavefield", "preadfield", "makefield"]

# Just for readability if run in parallel
def printr(string, comm=MPI.COMM_WORLD):
    if comm.rank==0:
        print(string)
        sys.stdout.flush()


# Saving field using bigfile, compatible with LDL
def savefield(data, filename, pm):
    """
    Saves field in "data" to file with name "filename"
    optionally in parallel (if run with mpirun)

    Uses format of bigfile, dividing data into blocks and
    saving extra attributes that can be used for a PMesh
    """
    with bigfile.FileMPI(pm.comm, filename, create=True) as ff:
        # Flatten in C order
        data = np.array(data).flatten()
        with ff.create_from_array('Field', data) as bb: # Create Field immediately
            bb.attrs['ndarray.shape'] = pm.Nmesh # For compatibility with LDL code
            bb.attrs['BoxSize'] = pm.BoxSize
            bb.attrs['Nmesh'] = pm.Nmesh


# Reading field saved using bigfile, compatible with LDL
def loadfield(path, comm=MPI.COMM_WORLD, local=False, pm=None):
    """
    Loads field in "path" (which should be constructed with bigfile.FileMPI)
    Returns a 3D numpy array with the readout field as well as its parameters
    Optionally returns local part of the field given a particle mesh "pm"
    If pm is specified, uses pm.comm as communicator
    """
    if pm:
        comm = pm.comm

    try:
        with bigfile.FileMPI(comm=comm, filename=path)['Field'] as ff: # Just expect a Field
            printr(f"Field found at {path}")
            attrs = {}
            for key in ff.attrs:
                v = ff.attrs[key]
                attrs[key] = np.squeeze(v)

            if ff.dtype.itemsize == 8:
                dtype = np.float64
            else:
                dtype = np.float32

            out = np.empty(np.prod(attrs['Nmesh']), dtype=dtype)
            out[...] = ff[...]
            out, Nmesh, BoxSize = out.reshape(attrs['Nmesh']), attrs['Nmesh'], attrs['BoxSize']

            if local:
                if not pm:
                    pm = PMesh(Nmesh=Nmesh, BoxSize=BoxSize, comm=comm)
                return out[pm.fftss[0]:pm.fftss[1], :, :]
            
            return out

    except:
        raise Exception(f"No map found at {path}")


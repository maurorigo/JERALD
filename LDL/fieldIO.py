# Utils for saving and reading files as in the old code, using bigfile
import bigfile
import numpy as np
import json
from mpi4py import MPI

__all__ = ["psavefield", "preadfield"]

def psavefield(data, filename, pm):
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
            bb.attrs['ndarray.shape'] = pm.Nmesh # For compatibility with old code
            bb.attrs['BoxSize'] = pm.BoxSize
            bb.attrs['Nmesh'] = pm.Nmesh


def preadfield(path, comm=MPI.COMM_WORLD):
    """
    Loads field in "path" (which should be constructed with bigfile.FileMPI)
    Returns a 3D numpy array with the readout field as well as its parameters
    """
    with bigfile.FileMPI(comm=comm, filename=path)['Field'] as ff: # Just expect a Field
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
        return out.reshape(attrs['Nmesh']), attrs['Nmesh'], attrs['BoxSize']


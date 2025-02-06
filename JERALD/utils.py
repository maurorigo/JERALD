from functools import wraps
from mpi4py import MPI
import numpy as np
import traceback

__all__ = ["MpiExceptionHandler"]

class MpiException(Exception):
    """
    Not really needed, but just for name
    """
    def __init__(self, msg):
        self.msg = msg
        super().__init__(self.msg)

    def __str__(self):
        return f"{self.msg}"

class MpiExceptionHandler(object):
    """
    Context manager for function that may raise an Exception.
    Sometimes a rank raising an error is not enough for the program to terminate, causing it to get
    stuck in case there are MPI communications happening later (and wasting cluster time).
    This also allows to print only one exception when multiple ranks raise the same one.
    """
    def __init__(self, comm):
        """
        Parameters:
            comm: MPI communicator
        """
        self.comm = comm
    def __enter__(self):
        pass
    def __exit__(self, etype, evalue, etraceback):
        err = False
        if evalue: # If any rank raises an exception set err to 1
            err = True
        err = self.comm.allgather(err)
        errranks = np.where(np.array(err))[0]
        if any(err):
            num = "Rank"
            if len(errranks) > 1:
                num = "Ranks"
            if self.comm.rank == 0: # Only raise exception with rank 0
                rlist = ", ".join(map(str, errranks))
                raise MpiException(f"{num} {rlist} failed for {etype.__name__} (see above)")
            else:
                exit()

""" Example (try without the context manager to see program getting stuck)
def somefunc(a, b):
    print("test")
    return a/b

comm = MPI.COMM_WORLD
with MpiExceptionHandler(comm):
    if comm.rank >= 0:
        a = somefunc(1, comm.rank)
        print(a)

foo = comm.allreduce(2)
"""



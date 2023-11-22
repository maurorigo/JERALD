# JAX BINDINGS FOR PARALLEL OPERATIONS
# Big thanks to https://github.com/dfm/extending-jax/tree/main

__all__ = ["initLayout", "clean", "ppaint", "preadout", "buildplan", "pfft", "pifft"]

from functools import partial

import numpy as np
from jax import core, dtypes, lax, custom_vjp, jit, config
from jax import numpy as jnp
from jax.core import ShapedArray, AbstractValue
from jax.interpreters import ad, batching, mlir, xla
from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call
import jaxlib.mlir.ir as ir
from jaxlib.mlir.dialects import mhlo

from mpi4py import MPI
import mpi4jax

# Register the CPU XLA custom calls
from jaxpops import cpu as cpu_ops
for _name, _value in cpu_ops.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="cpu")


# THE NEXT 4 FCNS ARE FROM MPI4JAX
def as_mhlo_constant(val, dtype):
    return mhlo.ConstantOp(
        ir.DenseElementsAttr.get(
            np.array([val], dtype=dtype), type=mlir.dtype_to_ir_type(np.dtype(dtype))
        )
    ).result

def to_mpi_ptr(mpi_obj):
    #Returns a pointer to the underlying C MPI object
    return np.uintp(MPI._addressof(mpi_obj))

def to_mpi_handle(mpi_obj):
    # Returns the handle of the underlying C mpi object.
    # Only defined for some MPI types (such as MPI_Comm), throws NotImplementedError otherwise.
    #Note: This is not a pointer, but the actual C integer representation of the object.
    return np.uintp(MPI._handleof(mpi_obj))

class HashableMPIType:
    def __init__(self, obj):
        self.wrapped = obj

    def __hash__(self):
        return int(to_mpi_ptr(self.wrapped))


# |############|
# | INITLAYOUT |
# |############|
# initLayout doesn't actually need a primitive, but for consistency
def initLayout(commsize, lyidx=0):
    # Initializes layout with index lyidx.
    
    return _initLayout_prim.bind(commsize=commsize, lyidx=lyidx)

def _initLayout_abstract(commsize, lyidx):
    return (ShapedArray((1, ), np.int32))

def _initLayout_lowering(ctx, commsize, lyidx):

    comm = as_mhlo_constant(commsize, np.int32)
    lyidx = as_mhlo_constant(lyidx, np.int32)

    out_type = mlir.aval_to_ir_type(ctx.avals_out[0])

    return custom_call(
        "initLayout",
        # Output types
        result_types=[out_type],
        # Inputs
        operands=[comm, lyidx],
    ).results


# Register the op
_initLayout_prim = core.Primitive("initLayout")
_initLayout_prim.def_impl(partial(xla.apply_primitive, _initLayout_prim))
_initLayout_prim.def_abstract_eval(_initLayout_abstract)

# Connect the XLA translation rules for JIT compilation
mlir.register_lowering(_initLayout_prim, _initLayout_lowering, platform="cpu")
# No need for vjp


# |#######|
# | CLEAN |
# |#######|
# clean doesn't actually need a primitive, but for consistency
def clean(lyidx=0):
    # Cleans layout and exchanged quantities with index lyidx.

    return _clean_prim.bind(lyidx=lyidx)

def _clean_abstract(lyidx):
    return (ShapedArray((1, ), np.int32))

def _clean_lowering(ctx, lyidx):

    lyidx = as_mhlo_constant(lyidx, np.int32)
    out_type = mlir.aval_to_ir_type(ctx.avals_out[0])

    # Different call based on dtype
    if config.x64_enabled:
        op_name = "clean_f64"
    else:
        op_name = "clean_f32"

    return custom_call(
        op_name,
        # Output types
        result_types=[out_type],
        # Inputs
        operands=[lyidx],
    ).results


# Register the op
_clean_prim = core.Primitive("clean")
_clean_prim.def_impl(partial(xla.apply_primitive, _clean_prim))
_clean_prim.def_abstract_eval(_clean_abstract)

# Connect the XLA translation rules for JIT compilation
mlir.register_lowering(_clean_prim, _clean_lowering, platform="cpu")
# No need for vjp


# |#######|
# | PAINT |
# |#######|
@partial(custom_vjp, nondiff_argnums=(2, 3, 4, 5, 6, 7, 8))
def ppaint(pos, mass, Nmesh, BoxSize, edges, dims, comm, lyidx=0, bwd=False):
    """
    JAX binding of parallel Cloud-In-Cell (CIC) painter in C++.
    Could do the type hinting at some point

    Parameters:
        pos (rrayLike): Local positions (Nparts/Nranks, 3) to paint
        mass (ArrayLike): Local masses (Nparts/Nranks, ) associated to the particles
        Nmesh (ArrayLike): Mesh sizes along x, y and z
        BoxSize (ArrayLike): Simulation box size along x, y and z
        edges (Array-like): (3, Nranks, 2) array specifying lower and upper limit along
            each direction for each rank
        dims (ArrayLike): Local dimensions of the output field
        comm (MPI_Comm): MPI communicator
        lyidx (int): Layout decomposition index (in case there are multiple, up to 4)
        bwd (bool): whether it's part of a readout derivative or not (if yes, use pre-existing layout)
    
    Returns:
        out (ArrayLike): Local painted field
    """
    if lyidx > 4:
        raise Exception("lyidx needs to be smaller than 5")

    # To make it possible to broadcast for constants
    mass = jnp.ones((pos.shape[0], )) * mass
    
    pos = pos % BoxSize; # PBC
    pos /= (BoxSize / Nmesh) # Mesh coordinates
    # Flatten everything for simplicity
    pos = pos.flatten()
    # Edges is (3, nranks, 2)
    edgesx = edges[0, :, :].flatten()
    edgesy = edges[1, :, :].flatten()
    edgesz = edges[2, :, :].flatten()

    # From https://github.com/google/jax/issues/3221
    # Actually check mpi4jax, for instance in collective_ops (in _src), barrier
    comm = HashableMPIType(comm)

    out = _ppaint_prim.bind(pos, mass, Nmesh, edgesx, edgesy, edgesz, dims=dims, comm=comm, lyidx=lyidx, bwd=bwd)
    
    return out.reshape((dims[0], dims[1], dims[2])) # Make it 3D

# Abstract evaluation rule
def _ppaint_abstract(pos, mass, Nmesh, edgesx, edgesy, edgesz, dims, comm, lyidx, bwd):
    return (ShapedArray((dims[0]*dims[1]*dims[2],), mass.dtype))

# Lowering rule
def _ppaint_lowering(ctx, pos, mass, Nmesh, edgesx, edgesy, edgesz, dims, comm, lyidx, bwd):

    # Get value of hashable
    comm = comm.wrapped

    # Extract the numpy type of the inputs
    pos_aval = ctx.avals_in[0]
    np_dtype = np.dtype(pos_aval.dtype)
    
    # Type of the output
    out_type = mlir.aval_to_ir_type(ctx.avals_out[0])

    # Number of particles is length of pos / 3
    Nparts = (np.prod(pos_aval.shape) / 3).astype(np.int64)
    lyidx = as_mhlo_constant(lyidx, np.int32)
    # This dtype is int64 only for Nparts and int32 only for lyidx, all the other ints are variable
    # int32_t or int64_t in C++ based on np_dtype (could also make more general)
    bwd = as_mhlo_constant(bwd, bool)

    # Dealing with comm as in mpi4jax, see for instance barrier.py in collective ops
    comm = as_mhlo_constant(to_mpi_handle(comm), np.uintp)

    # Different call based on dtype
    if np_dtype == np.float32:
        op_name = "ppaint_f32"
    elif np_dtype == np.float64:
        op_name = "ppaint_f64"
    else:
        raise NotImplementedError(f"Unsupported dtype {np_dtype}")

    return custom_call(
        op_name,
        # Output types
        result_types=[out_type],
        # Inputs
        operands=[pos, mlir.ir_constant(Nparts), mass, Nmesh, edgesx, edgesy, edgesz, comm, lyidx, bwd],
    ).results


# Register the op
_ppaint_prim = core.Primitive("ppaint")
_ppaint_prim.def_impl(partial(xla.apply_primitive, _ppaint_prim))
_ppaint_prim.def_abstract_eval(_ppaint_abstract)

# Connect the XLA translation rules for JIT compilation
mlir.register_lowering(_ppaint_prim, _ppaint_lowering, platform="cpu")

# VJP definition for ppaint (see notes for a derivation)
def ppaint_fwd(pos, mass, Nmesh, BoxSize, edges, dims, comm, lyidx=0, bwd=False):
    return ppaint(pos, mass, Nmesh, BoxSize, edges, dims, comm, lyidx, bwd), (pos, mass)

def ppaint_bwd(Nmesh, BoxSize, edges, dims, comm, lyidx, bwd, res, vimg):
    pos, mass = res
    localedges = edges[:, comm.Get_rank(), :]
    out1 = jnp.zeros(pos.shape)
    out1 = out1.at[:, 0].set(preadout(pos, vimg, Nmesh, BoxSize, edges, dims, comm, lyidx, 0))
    out1 = out1.at[:, 1].set(preadout(pos, vimg, Nmesh, BoxSize, edges, dims, comm, lyidx, 1))
    out1 = out1.at[:, 2].set(preadout(pos, vimg, Nmesh, BoxSize, edges, dims, comm, lyidx, 2))

    out2 = preadout(pos, vimg, Nmesh, BoxSize, edges, dims, comm, lyidx)

    return ((out1.T * mass).T, out2)

ppaint.defvjp(ppaint_fwd, ppaint_bwd)


# |#########|
# | READOUT |
# |#########|
@partial(custom_vjp, nondiff_argnums=(2, 3, 4, 5, 6, 7, 8))
def preadout(pos, field, Nmesh, BoxSize, edges, dims, comm, lyidx=0, vjpdim=-1):
    """
    JAX binding of parallel Cloud-In-Cell (CIC) readout in C++.

    Parameters:
        pos (rrayLike): Local positions (Nparts/Nranks, 3) to readout at
        field (ArrayLike): Local field to readout from
        Nmesh (ArrayLike): Mesh sizes along x, y and z
        BoxSize (ArrayLike): Simulation box size along x, y and z
        edges (Array-like): (3, Nranks, 2) array specifying lower and upper limit along
            each direction for each rank
        dims (ArrayLike): Local dimensions of the field (mostly for consistency with ppaint)
        comm (MPI_Comm): MPI communicator
        lyidx (int): Layout decomposition index (in case there are multiple, up to 4)
        vjpdim (int): Dimension of the gradient, see notes for an explanation
    
    Returns:
        out (ArrayLike): (Nparts/Nranks, ) array representing the local read-out values
    """
    if lyidx > 4:
        raise Exception("lyidx needs to be smaller than 5")
    if vjpdim > 2:
        raise Exception("vjpdim must be between 0 and 2")

    pos = pos % BoxSize; # PBC
    pos /= (BoxSize / Nmesh) # Mesh coordinates
    # Flatten everything for simplicity
    pos = pos.flatten()
    field = field.flatten()
    # Edges is (3, nranks, 2)
    edgesx = edges[0, :, :].flatten()
    edgesy = edges[1, :, :].flatten()
    edgesz = edges[2, :, :].flatten()

    # From https://github.com/google/jax/issues/3221
    # Actually check mpi4jax, for instance in collective_ops (in _src), barrier
    comm = HashableMPIType(comm)

    return _preadout_prim.bind(pos, field, Nmesh, BoxSize, edgesx, edgesy, edgesz, comm=comm, lyidx=lyidx, vjpdim=vjpdim)

# Abstract evaluation rules
def _preadout_abstract(pos, field, Nmesh, BoxSize, edgesx, edgesy, edgesz, comm, lyidx, vjpdim):
    return (ShapedArray((int(pos.shape[0]/3), ), pos.dtype))

# Lowering rules
def _preadout_lowering(ctx, pos, field, Nmesh, BoxSize, edgesx, edgesy, edgesz, comm, lyidx, vjpdim):

    # Get value of hashable
    comm = comm.wrapped

    # Extract the numpy type of the inputs
    pos_aval = ctx.avals_in[0]
    np_dtype = np.dtype(pos_aval.dtype)

    out_type = mlir.aval_to_ir_type(ctx.avals_out[0])

    # Number of particles is length of pos / 3
    Nparts = (np.prod(pos_aval.shape) / 3).astype(np.int64)
    lyidx = as_mhlo_constant(lyidx, np.int32)
    vjpdim = as_mhlo_constant(vjpdim, np.int32)
    # This dtype is int64 only for Nparts and int32 only for lyidx and vjpdim, all the
    # other ints are int32_t or int64_t in C++ based on np_dtype (could also make more general)

    # Dealing with comm as in mpi4jax, see for instance barrier.py in collective ops
    comm = as_mhlo_constant(to_mpi_handle(comm), np.uintp)

    # Different call based on dtype
    if np_dtype == np.float32:
        op_name = "preadout_f32"
    elif np_dtype == np.float64:
        op_name = "preadout_f64"
    else:
        raise NotImplementedError(f"Unsupported dtype {np_dtype}")

    return custom_call(
        op_name,
        # Output types
        result_types=[out_type],
        # Inputs
        operands=[pos, mlir.ir_constant(Nparts), field, Nmesh, BoxSize, edgesx, edgesy, edgesz, comm, lyidx, vjpdim],
    ).results


# Register the op
_preadout_prim = core.Primitive("preadout")
_preadout_prim.def_impl(partial(xla.apply_primitive, _preadout_prim))
_preadout_prim.def_abstract_eval(_preadout_abstract)

# Connect the XLA translation rules for JIT compilation
mlir.register_lowering(_preadout_prim, _preadout_lowering, platform="cpu")

# VJP definition for preadout (see notes for a derivation)
def preadout_fwd(pos, field, Nmesh, BoxSize, edges, dims, comm, lyidx=0, vjpdim=-1):
    return preadout(pos, field, Nmesh, BoxSize, edges, dims, comm, lyidx, vjpdim), (pos, field)

def preadout_bwd(Nmesh, BoxSize, edges, dims, comm, lyidx, vjpdim, res, vmass):
    pos, field = res
    out1 = jnp.zeros(pos.shape)
    out1 = out1.at[:, 0].set(preadout(pos, field, Nmesh, BoxSize, edges, dims, comm, lyidx, 0))
    out1 = out1.at[:, 1].set(preadout(pos, field, Nmesh, BoxSize, edges, dims, comm, lyidx, 1))
    out1 = out1.at[:, 2].set(preadout(pos, field, Nmesh, BoxSize, edges, dims, comm, lyidx, 2))

    out2 = ppaint(pos, vmass, Nmesh, BoxSize, edges, dims, comm, lyidx, True)

    return ((out1.T * vmass).T, out2)

preadout.defvjp(preadout_fwd, preadout_bwd)


# |###########|
# | BUILDPLAN |
# |###########|
# Buildplan doesn't actually need a primitive, but for consistency
def buildplan(Nmesh, comm):
    """
        Creates plans for forward and backward ffts, returning partition specs
        Binds FFTW, which only splits 1st axis, so returns local x size and offset.
    """
    # From https://github.com/google/jax/issues/3221
    # Actually check mpi4jax, for instance in collective_ops (in _src), barrier
    comm = HashableMPIType(comm)

    return _buildplan_prim.bind(Nmesh.astype(np.int32), comm=comm)

def _buildplan_abstract(Nmesh, comm):
    return (ShapedArray((2, ), np.int32))

def _buildplan_lowering(ctx, Nmesh, comm):

    comm = comm.wrapped

    out_type = mlir.aval_to_ir_type(ctx.avals_out[0])

    comm = as_mhlo_constant(to_mpi_handle(comm), np.uintp)

    # Different call based on dtype
    if config.x64_enabled:
        op_name = "buildplan_f64"
    else:
        op_name = "buildplan_f32"

    return custom_call(
        op_name,
        # Output types
        result_types=[out_type],
        # Inputs
        operands=[Nmesh, comm],
    ).results


# Register the op
_buildplan_prim = core.Primitive("buildplan")
_buildplan_prim.def_impl(partial(xla.apply_primitive, _buildplan_prim))
_buildplan_prim.def_abstract_eval(_buildplan_abstract)

# Connect the XLA translation rules for JIT compilation
mlir.register_lowering(_buildplan_prim, _buildplan_lowering, platform="cpu")
# No need for vjp


# |######|
# | PFFT |
# |######|
@partial(custom_vjp, nondiff_argnums=(1,))
def pfft(data, dims):
    """
        Computes parallel forward FFT of data (localL, M, N), returning
        the FT as a complex array of size (localL, M, int(N/2)+1)
    """
    real, imag = _pfft_prim.bind(data.flatten(), dims=dims)

    return real.reshape((dims[0], dims[1], int(dims[2]/2)+1)) +\
            1j*imag.reshape((dims[0], dims[1], int(dims[2]/2)+1))

def _pfft_abstract(data, dims):
    return (ShapedArray((dims[0] * dims[1] * (int(dims[2]/2)+1), ), data.dtype),
            ShapedArray((dims[0] * dims[1] * (int(dims[2]/2)+1), ), data.dtype))

def _pfft_lowering(ctx, data, dims):
    
    data_aval = ctx.avals_in[0]
    np_dtype = np.dtype(data_aval.dtype)

    out_type = mlir.aval_to_ir_type(ctx.avals_out[0])

    # Different call based on dtype
    if np_dtype == np.float32:
        op_name = "pfft_f32"
    elif np_dtype == np.float64:
        op_name = "pfft_f64"
    else:
        raise NotImplementedError(f"Unsupported dtype {np_dtype}")
    
    return custom_call(
        op_name,
        # Output types
        result_types=[out_type, out_type],
        # Inputs
        operands=[data],
    ).results


# Register the op
_pfft_prim = core.Primitive("pfft")
_pfft_prim.multiple_results = True
_pfft_prim.def_impl(partial(xla.apply_primitive, _pfft_prim))
_pfft_prim.def_abstract_eval(_pfft_abstract)

# Connect the XLA translation rules for JIT compilation
mlir.register_lowering(_pfft_prim, _pfft_lowering, platform="cpu")

# VJP definition for pfft (see notes for details)
def pfft_fwd(data, dims):
    return pfft(data, dims), ()

def pfft_bwd(dims, res, v):
    # Should be conj(ifft(conj(v))).real, but outer conj is useless
    return (pifft(jnp.conj(v), dims).real,)

pfft.defvjp(pfft_fwd, pfft_bwd)


# |#######|
# | PIFFT |
# |#######|
@partial(custom_vjp, nondiff_argnums=(1,))
def pifft(data, dims):
    """
        Computes parallel backward FFT of complex data (localL, M, int(N/2)+1),
        returning the inverse FT with size (localL, M, N)
    """
    out = _pifft_prim.bind(data.real.flatten(), data.imag.flatten(), dims=dims)

    return out.reshape((dims[0], dims[1], dims[2]))

def _pifft_abstract(real, imag, dims):
    return (ShapedArray((dims[0]*dims[1]*dims[2], ), real.dtype))

def _pifft_lowering(ctx, real, imag, dims):

    data_aval = ctx.avals_in[0]
    np_dtype = np.dtype(data_aval.dtype)

    out_type = mlir.aval_to_ir_type(ctx.avals_out[0])

    # Different call based on dtype
    if np_dtype == np.float32:
        op_name = "pifft_f32"
    elif np_dtype == np.float64:
        op_name = "pifft_f64"
    else:
        raise NotImplementedError(f"Unsupported dtype {np_dtype}")

    return custom_call(
        op_name,
        # Output types
        result_types=[out_type],
        # Inputs
        operands=[real, imag],
    ).results


# Register the op
_pifft_prim = core.Primitive("pifft")
_pifft_prim.def_impl(partial(xla.apply_primitive, _pifft_prim))
_pifft_prim.def_abstract_eval(_pifft_abstract)

# Connect the XLA translation rules for JIT compilation
mlir.register_lowering(_pifft_prim, _pifft_lowering, platform="cpu")

# VJP definition for pifft (also see notes)
def pifft_fwd(data, dims):
    return pifft(data, dims), ()

def pifft_bwd(dims, res, v):
    return (jnp.conj(pfft(v, dims)),)

pifft.defvjp(pifft_fwd, pifft_bwd)


// Also adapted from https://github.com/dfm/extending-jax
// This file defines the Python interface to the XLA custom call implemented on the CPU.
// For handling the MPI communicator, I'm using:
// https://github.com/mpi4jax/mpi4jax/blob/master/mpi4jax/_src/xla_bridge/mpi_xla_bridge_cpu.pyx

#include "parops.h"
#include "parfft.h"
#include <pybind11/pybind11.h>
#include <cstdint> // For MPI comm handle and int types

using namespace parops_jax;
using namespace parfft_jax;

// These global variables are not passed to python because they may be hard to handle by JAX
Layout layout[5]; // General decomposition layouts (at most 5 can be saved)
template <typename T>Exchanged<T> exd[5]; // Exchanged positions and masses (at most 5 can be saved)

Planner planner; // fft double planner
Plannerf plannerf; // fft float planner

namespace {

void cinitLayout(void* out, void** in){
    int32_t commsize = *reinterpret_cast<int32_t*>(in[0]);
    int32_t lyidx = *reinterpret_cast<int32_t*>(in[1]);

    initLayout(&(layout[lyidx]), commsize);

    int32_t* chk = reinterpret_cast<int32_t*>(out);
    int32_t foo = 0;
    *chk = foo;
}

template <typename T>
void cclean(void* out, void** in){
    int32_t lyidx = *reinterpret_cast<int32_t*>(in[1]);

    cleanpops(&(layout[lyidx]), &(exd<T>[lyidx]));

    int32_t* chk = reinterpret_cast<int32_t*>(out);
    int32_t foo = 0;
    *chk = foo;
}

template <typename T, typename intT>
void cppaint(void* out, void** in){
    // Parallel painter, also build decomposition layout
    // Parse inputs (flattened for simplicity, row-major order)
    T* pos = reinterpret_cast<T*>(in[0]);
    int64_t Nparts = *reinterpret_cast<int64_t*>(in[1]);
    T* mass = reinterpret_cast<T*>(in[2]);
    intT* Nmesh = reinterpret_cast<intT*>(in[3]);
    intT* edgesx = reinterpret_cast<intT*>(in[4]);
    intT* edgesy = reinterpret_cast<intT*>(in[5]);
    intT* edgesz = reinterpret_cast<intT*>(in[6]);
    MPI_Comm comm = reinterpret_cast<MPI_Comm>(*reinterpret_cast<uintptr_t*>(in[7]));
    int32_t lyidx = *reinterpret_cast<int32_t*>(in[8]);
    bool useLayout = *reinterpret_cast<bool*>(in[9]);

    // Output (flattened, row-major)
    T* outp = reinterpret_cast<T*>(out);
    ppaint<T, intT>(pos, Nparts, mass, Nmesh, edgesx, edgesy, edgesz, &outp, &(layout[lyidx]), &(exd<T>[lyidx]), useLayout, comm);
}

template <typename T, typename intT>
void cpreadout(void* out, void** in){
    // Parallel readout
    // NOTE: uses decomposition layout already defined
    // Very similar to above
    T* pos = reinterpret_cast<T*>(in[0]);
    int64_t Nparts = *reinterpret_cast<int64_t*>(in[1]);
    T* localfield = reinterpret_cast<T*>(in[2]);
    intT* Nmesh = reinterpret_cast<intT*>(in[3]);
    T* BoxSize = reinterpret_cast<T*>(in[4]);
    intT* edgesx = reinterpret_cast<intT*>(in[5]);
    intT* edgesy = reinterpret_cast<intT*>(in[6]);
    intT* edgesz = reinterpret_cast<intT*>(in[7]);
    MPI_Comm comm = reinterpret_cast<MPI_Comm>(*reinterpret_cast<uintptr_t*>(in[8]));
    int32_t lyidx = *reinterpret_cast<int32_t*>(in[9]);
    int32_t vjpdim = *reinterpret_cast<int32_t*>(in[10]);

    // Output
    T* outp = reinterpret_cast<T*>(out);
    preadout<T, intT>(layout[lyidx], exd<T>[lyidx], Nparts, localfield, Nmesh, BoxSize, edgesx, edgesy, edgesz, &outp, comm, vjpdim);
}

void cbuildplan(void* out, void** in){
    // Build plan for FFTW, saves it locally and returns useful variables for decomposition
    // Inputs
    int32_t* Nmesh = reinterpret_cast<int32_t*>(in[0]);
    MPI_Comm comm = reinterpret_cast<MPI_Comm>(*reinterpret_cast<uintptr_t*>(in[1]));
    // Outputs
    int32_t* localvars = reinterpret_cast<int32_t*>(out);

    buildplan(Nmesh[0], Nmesh[1], Nmesh[2], &planner, comm);

    // Return local size and local start index
    localvars[0] = planner.localL;
    localvars[1] = planner.localstart;
}

void cbuildplanf(void* out, void** in){
    // Float version of above
    // Inputs
    int32_t* Nmesh = reinterpret_cast<int32_t*>(in[0]);
    MPI_Comm comm = reinterpret_cast<MPI_Comm>(*reinterpret_cast<uintptr_t*>(in[1]));
    // Outputs
    int32_t* localvars = reinterpret_cast<int32_t*>(out);

    buildplanf(Nmesh[0], Nmesh[1], Nmesh[2], &plannerf, comm);

    localvars[0] = plannerf.localL;
    localvars[1] = plannerf.localstart;
}

void cpfft(void* out, void** in){
    // Parallel forward FFT (real to complex)
    // NOTE: uses pre-build plan
    // Inputs
    double* data = reinterpret_cast<double*>(in[0]);
    // Outputs
    void** outs = reinterpret_cast<void**>(out);
    double* realout = reinterpret_cast<double*>(outs[0]);
    double* imagout = reinterpret_cast<double*>(outs[1]);

    pfft(data, planner);
    for (int i=0; i<planner.localL*planner.M*(planner.N/2+1); i++){
    	realout[i] = planner.cplx[i][0];
	imagout[i] = planner.cplx[i][1];
    }
}

void cpfftf(void* out, void** in){
    // Float version of the above
    // Inputs
    float* data = reinterpret_cast<float*>(in[0]);
    // Outputs
    void** outs = reinterpret_cast<void**>(out);
    float* realout = reinterpret_cast<float*>(outs[0]);
    float* imagout = reinterpret_cast<float*>(outs[1]);

    pfftf(data, plannerf);
    for (int i=0; i<plannerf.localL*plannerf.M*(plannerf.N/2+1); i++){
        realout[i] = plannerf.cplx[i][0];
        imagout[i] = plannerf.cplx[i][1];
    }
}

void cpifft(void* out, void** in){
    // Parallel backwards FFT (complex to real)
    // NOTE: Uses pre-built plan
    // Inputs
    double* realdata = reinterpret_cast<double*>(in[0]);
    double* imagdata = reinterpret_cast<double*>(in[1]);
    // Output
    double* outp = reinterpret_cast<double*>(out);

    pifft(realdata, imagdata, planner);
	
    int L = planner.localL;
    int M = planner.M;
    int N = planner.N;
    for (int i = 0; i < L; i++){
        for (int j = 0; j < M; j++){
            for (int k = 0; k < N; k++){
                outp[i*M*N + N*j + k] = planner.real[(i*M + j) * 2*(N/2+1) + k];
            }
        }
    }
}

void cpifftf(void* out, void** in){
    // Float version of the above
    // Inputs
    float* realdata = reinterpret_cast<float*>(in[0]);
    float* imagdata = reinterpret_cast<float*>(in[1]);
    // Output
    float* outp = reinterpret_cast<float*>(out);

    pifftf(realdata, imagdata, plannerf);

    int L = plannerf.localL;
    int M = plannerf.M;
    int N = plannerf.N;
    for (int i = 0; i < L; i++){
        for (int j = 0; j < M; j++){
            for (int k = 0; k < N; k++){
                outp[i*M*N + N*j + k] = plannerf.real[(i*M + j) * 2*(N/2+1) + k];
            }
        }
    }
}

template <typename T>
pybind11::capsule EncapsulateFunction(T* fn) {
    return pybind11::capsule((void*)fn, "xla._CUSTOM_CALL_TARGET");
}

// Pybind registrations for XLA
pybind11::dict Registrations(){
    pybind11::dict dict;
    dict["initLayout"] = EncapsulateFunction(cinitLayout);
    dict["clean_f32"] = EncapsulateFunction(cclean<float>);
    dict["clean_f64"] = EncapsulateFunction(cclean<double>);
    dict["ppaint_f32"] = EncapsulateFunction(cppaint<float, int32_t>);
    dict["ppaint_f64"] = EncapsulateFunction(cppaint<double, int64_t>);
    dict["preadout_f32"] = EncapsulateFunction(cpreadout<float, int32_t>);
    dict["preadout_f64"] = EncapsulateFunction(cpreadout<double, int64_t>);
    dict["buildplan_f64"] = EncapsulateFunction(cbuildplan);
    dict["buildplan_f32"] = EncapsulateFunction(cbuildplanf);
    dict["pfft_f64"] = EncapsulateFunction(cpfft);
    dict["pfft_f32"] = EncapsulateFunction(cpfftf);
    dict["pifft_f64"] = EncapsulateFunction(cpifft);
    dict["pifft_f32"] = EncapsulateFunction(cpifftf);
    return dict;
}

PYBIND11_MODULE(cpu, m) {
    m.def("registrations", &Registrations);
}

} // namespace


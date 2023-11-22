#ifndef _JAXPARFFT_H_
#define _JAXPARFFT_H_

#include <fftw3-mpi.h>
#include <iostream>
#include <cstdint>
#include <mpi.h>

using namespace std;

namespace parfft_jax {

struct Planner{
    fftw_plan planfwd; // Plan for forward fft
    fftw_plan planbwd; // Plan for backward fft
    int L, M, N; // Dimensions
    ptrdiff_t localL; // Local 0 dimension
    ptrdiff_t localstart; // Starting dimension 0 index
    double* real; // Real field pointer
    fftw_complex* cplx; // Complex field pointer
};

struct Plannerf{ // Same as above, for float
    fftwf_plan planfwd;
    fftwf_plan planbwd;
    int L, M, N;
    ptrdiff_t localL;
    ptrdiff_t localstart;
    float* real;
    fftwf_complex* cplx;
};

void buildplan(int32_t& L, int32_t& M, int32_t& N, Planner* planner, MPI_Comm& comm){
    // Build fft and ifft plans and find partitioning
    fftw_mpi_init();	
    ptrdiff_t alloclocal, localL, localstart;

    // Get local data size and allocate
    alloclocal = fftw_mpi_local_size_3d(L, M, N/2+1, comm,
                                         &localL, &localstart);
    double* real = fftw_alloc_real(2 * alloclocal);
    fftw_complex* cplx = fftw_alloc_complex(alloclocal);

    // Create plans
    fftw_plan planfwd = fftw_mpi_plan_dft_r2c_3d(L, M, N, real, cplx, comm, FFTW_ESTIMATE);
    fftw_plan planbwd = fftw_mpi_plan_dft_c2r_3d(L, M, N, cplx, real, comm, FFTW_ESTIMATE);
    // Save things in planner
    (*planner).planfwd = planfwd;
    (*planner).planbwd = planbwd;
    (*planner).localL = localL;
    (*planner).localstart = localstart;
    (*planner).real = real;
    (*planner).cplx = cplx;
    (*planner).L = L;
    (*planner).M = M;
    (*planner).N = N;
}

void buildplanf(int32_t& L, int32_t& M, int32_t& N, Plannerf* planner, MPI_Comm& comm){
    // As above, for float 
    fftwf_mpi_init();
    ptrdiff_t alloclocal, localL, localstart;

    alloclocal = fftwf_mpi_local_size_3d(L, M, N/2+1, comm,
                                         &localL, &localstart);
    float* real = fftwf_alloc_real(2 * alloclocal);
    fftwf_complex* cplx = fftwf_alloc_complex(alloclocal);

    fftwf_plan planfwd = fftwf_mpi_plan_dft_r2c_3d(L, M, N, real, cplx, comm, FFTW_ESTIMATE);
    fftwf_plan planbwd = fftwf_mpi_plan_dft_c2r_3d(L, M, N, cplx, real, comm, FFTW_ESTIMATE);
    (*planner).planfwd = planfwd;
    (*planner).planbwd = planbwd;
    (*planner).localL = localL;
    (*planner).localstart = localstart;
    (*planner).real = real;
    (*planner).cplx = cplx;
    (*planner).L = L;
    (*planner).M = M;
    (*planner).N = N;
}

void pfft(double*& data, Planner& planner){
    // Forward double fft
    int L = planner.localL;
    int M = planner.M;
    int N = planner.N;
    // Copy input to planner pointer
    for (int i = 0; i < L; i++){
        for (int j = 0; j < M; j++){
            for (int k = 0; k < N; k++){
                planner.real[(i*M + j) * 2*(N/2+1) + k] = data[i*M*N + N*j + k];
            }
        }
    }
    
    // Run fft
    fftw_execute(planner.planfwd);

    // Old way to copy output (requires inputs double** realout, double** imagout), now just copy outside
    /*(*realout) = new double[L*M*(N/2+1)];
    (*imagout) = new double[L*M*(N/2+1)];
    for (int i = 0; i < L*M*(N/2+1); i++){
	(*realout)[i] = planner.cplx[i][0];
	(*imagout)[i] = planner.cplx[i][1];
    }*/
}

void pifft(double*& realdata, double*& imagdata, Planner& planner){
    // Backward double fft
    int L = planner.localL;
    int M = planner.M;
    int N = planner.N;
    // Copy input to planner pointer
    for (int i = 0; i < L*M*(N/2+1); i++){
        planner.cplx[i][0] = realdata[i];
        planner.cplx[i][1] = imagdata[i];
    }

    // Run ifft
    fftw_execute(planner.planbwd);

    // Output is saved outside (to use this again, need a double** out input))
    /*(*out) = new double[L*M*N];
    for (int i = 0; i < L; i++){
        for (int j = 0; j < M; j++){
            for (int k = 0; k < N; k++){
                (*out)[i*M*N + N*j + k] = planner.real[(i*M + j) * 2*(N/2+1) + k];
            }
        }
    }*/
}

void pfftf(float*& data, Plannerf& planner){
    // Forward float fft
    int L = planner.localL;
    int M = planner.M;
    int N = planner.N;
    for (int i = 0; i < L; i++){
        for (int j = 0; j < M; j++){
            for (int k = 0; k < N; k++){
                planner.real[(i*M + j) * 2*(N/2+1) + k] = data[i*M*N + N*j + k];
            }
        }
    }

    fftwf_execute(planner.planfwd);

    // Old way to copy output (requires inputs float** realout, float** imagout), now just copy outside
    /*(*realout) = new float[L*M*(N/2+1)];
    (*imagout) = new float[L*M*(N/2+1)];
    for (int i = 0; i < L*M*(N/2+1); i++){
        (*realout)[i] = planner.cplx[i][0];
        (*imagout)[i] = planner.cplx[i][1];
    }*/
}

void pifftf(float*& realdata, float*& imagdata, Plannerf& planner){
    // Backward float fft
    int L = planner.localL;
    int M = planner.M;
    int N = planner.N;
    for (int i = 0; i < L*M*(N/2+1); i++){
        planner.cplx[i][0] = realdata[i];
        planner.cplx[i][1] = imagdata[i];
    }

    fftwf_execute(planner.planbwd);

    // Output is saved outside (to use this again, need a float** out input)
    /*(*out) = new float[L*M*N];
    for (int i = 0; i < L; i++){
        for (int j = 0; j < M; j++){
            for (int k = 0; k < N; k++){
                (*out)[i*M*N + N*j + k] = planner.real[(i*M + j) * 2*(N/2+1) + k];
            }
        }
    }*/
}

} // Namespace parfft_jax

#endif

#ifndef _JAXPAROPS_H_
#define _JAXPAROPS_H_

#include <iostream>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <omp.h>

using namespace std;

namespace parops_jax {

struct Layout{
    int* indices; // Particles indices per rank, stacked vertically
    // Parameters for sending data using alltoallv
    int sendsize;
    int* sendcounts;
    int* senddispl;
    // Parameters for receiving data using alltoallv
    int recvsize;
    int* recvcounts;
    int* recvdispl;
};

template <typename T, typename intT>
void decompose(T** pos, // (Nparts, 3) Positions (in mesh space)
	       int64_t Nparts, // Number of particles
	       intT* Nmesh, // Mesh grid specs (3, )
	       intT* edgesx, // Limit grid indices along x for each rank, (comm_size, 2) flattened
	       intT* edgesy, // Same, for y
	       intT* edgesz, // Same, for z
	       Layout* layout, // Layout for decomposition
	       MPI_Comm comm=MPI_COMM_WORLD){ // MPI communicator
    // Creates a layout decomposition based on the given positions and
    // edges of the domains for CIC.
    // Particles can be repeated (due to ghosts)
    
    // Shifts for different contributions of CIC
    int ds[8][3] = {{0, 0, 0},
    		    {1, 0, 0},
		    {0, 1, 0},
		    {0, 0, 1},
		    {1, 1, 0},
		    {1, 0, 1},
		    {0, 1, 1},
		    {1, 1, 1}};

    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    // Array of vectors with particle indices for each rank (vector because size is unknown a priori)
    vector<int>* ppr = new vector<int>[comm_size];
    int* sendcounts = new int[comm_size];
    for (int rank=0; rank<comm_size; rank++){
        ppr[rank].reserve(Nparts / comm_size); // Initialize vec to a sensible size to avoid realloc
	sendcounts[rank] = 0;
    }

    int i, j, k, i1, j1, k1;
    for (int64_t p=0; p<Nparts; p++){
	i = int(pos[p][0]) % Nmesh[0]; // Closest grid point to position towards axes origin
	j = int(pos[p][1]) % Nmesh[1];
	k = int(pos[p][2]) % Nmesh[2];
	for (int rank=0; rank<comm_size; rank++){ // For each rank, check every CIC direction
	    for (int idx=0; idx<8; idx++){
	    	i1 = (i + ds[idx][0]) % Nmesh[0];
            	j1 = (j + ds[idx][1]) % Nmesh[1];
            	k1 = (k + ds[idx][2]) % Nmesh[2];
		if ((unsigned)(i1 - edgesx[2*rank]) < (edgesx[2*rank+1] - edgesx[2*rank]) &&
                    (unsigned)(j1 - edgesy[2*rank]) < (edgesy[2*rank+1] - edgesy[2*rank]) &&
                    (unsigned)(k1 - edgesz[2*rank]) < (edgesz[2*rank+1] - edgesz[2*rank])){    
		    ppr[rank].push_back(p); // Add particle index to this rank
		    sendcounts[rank]++; // Increment this rank counter
		    break; // If this particle contributes to the rank, just exit (no doubles)
		}
	    }
	}
    }

    // Define the layout decomposition
    (*layout).sendcounts = sendcounts;

    // Send to each rank the size of the data coming from this
    int* recvcounts = new int[comm_size];
    MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, comm);
    (*layout).recvcounts = recvcounts;

    // Define the send and receive sizes
    int sendsize = 0, recvsize = 0;
    for (int rank=0; rank<comm_size; rank++){
	sendsize += sendcounts[rank];
	recvsize += recvcounts[rank];
    }
    (*layout).sendsize = sendsize;
    (*layout).recvsize = recvsize;

    // Define layout indices
    int *indices = new int[sendsize];
    int ctr = 0;
    for (int rank=0; rank<comm_size; rank++){
        for (int p : ppr[rank]){
  	    indices[ctr] = p;
	    ctr++;
	}
    }
    (*layout).indices = indices;

    // Compute send and receive displacements for alltoallv
    int* senddispl = new int[comm_size];
    int* recvdispl = new int[comm_size];
    senddispl[0] = 0;
    recvdispl[0] = 0;
    for (int rank=1; rank<comm_size; rank++){
	senddispl[rank] = senddispl[rank-1] + sendcounts[rank-1];
	recvdispl[rank] = recvdispl[rank-1] + recvcounts[rank-1];
    }
    (*layout).senddispl = senddispl;
    (*layout).recvdispl = recvdispl;

}

template <typename T>
void exchange(T** data, int dims, Layout layout, T*** outdata, MPI_Comm comm=MPI_COMM_WORLD){
    // Sends what's in "data" to the correct ranks to perform
    // CIC operations.
    // dims is the dimensionality of data
    // layout is the decomposition layout, computed using "decompose".
    // Sets the value in outdata to the "data" each
    // rank have assigned to this (including self).

    // Construct the array that contains the data to send
    // Note: data is flattened to easily make it contiguous (needed for alltoallv)
    T* sendbuf = new T[layout.sendsize * dims];
    for (int i=0; i<layout.sendsize; i++){
    	for (int d=0; d<dims; d++) sendbuf[dims*i + d] = data[layout.indices[i]][d];
    }

    // Receive buffer (made this way to make it contiguous)
    // see https://stackoverflow.com/questions/21943621/how-to-create-a-contiguous-2d-array-in-c
    T** recvbuf = new T*[layout.recvsize];
    T* recvpool = new T[layout.recvsize * dims];
    for (int i=0; i<layout.recvsize; i++, recvpool += dims){
        recvbuf[i] = recvpool;
    }

    // Create datatype to send
    MPI_Datatype dt;
    if (is_same<T, double>::value){
	MPI_Type_contiguous(dims, MPI_DOUBLE, &dt);
    } else if (is_same<T, float>::value){
        MPI_Type_contiguous(dims, MPI_FLOAT, &dt);
    } else {
        throw invalid_argument("Data type not supported");
    }
    MPI_Type_commit(&dt);

    // FIRE
    MPI_Alltoallv(sendbuf,
		  layout.sendcounts,
		  layout.senddispl,
		  dt,
		  &(recvbuf[0][0]),
		  layout.recvcounts,
		  layout.recvdispl,
		  dt,
		  comm);

    *outdata = recvbuf; // Store it in the input pointer
}

template <typename T>
void exchange1D(T* data, Layout layout, T** outdata, MPI_Comm comm=MPI_COMM_WORLD){
    // 1D version of above for convenience
    
    // Note: 1D array is flat and contiguous so can pass easily to alltoallv
    T* sendbuf = new T[layout.sendsize];
    for (int i=0; i<layout.sendsize; i++){
        sendbuf[i] = data[layout.indices[i]];
    }

    // Receive buffer
    T* recvbuf = new T[layout.recvsize];

    // Choose datatype to send based on T
    MPI_Datatype dt;
    if (is_same<T, double>::value){
        dt = MPI_DOUBLE;
    } else if (is_same<T, float>::value){
        dt = MPI_FLOAT;
    } else {
        throw invalid_argument("Data type not supported");
    }

    // FIRE
    MPI_Alltoallv(sendbuf,
                  layout.sendcounts,
                  layout.senddispl,
                  dt,
                  recvbuf,
                  layout.recvcounts,
                  layout.recvdispl,
                  dt,
                  comm);

    *outdata = recvbuf;
}

template <typename T>
void gather1D(T* data, Layout layout, T** outdata, MPI_Comm comm=MPI_COMM_WORLD){
    // Gathers decomposed data, sending it back to the original rank,
    // and summing contributions of ghosts, based on decomposition layout
    
    // This time we're receiving back original data, so receiving and sending parameters are flipped
    T* recvbuf = new T[layout.sendsize];

    // Choose datatype to send based on T
    MPI_Datatype dt;
    if (is_same<T, double>::value){
        dt = MPI_DOUBLE;
    } else if (is_same<T, float>::value){
        dt = MPI_FLOAT;
    } else {
        throw invalid_argument("Data type not supported");
    }

    // Fire back
    MPI_Alltoallv(data,
		  layout.recvcounts,
		  layout.recvdispl,
		  dt,
		  recvbuf,
		  layout.sendcounts,
		  layout.senddispl,
		  dt,
		  comm);

    // Sum contributions of different ranks to each particle (with index indices[i])
    int i = 0;
    for_each(layout.indices, layout.indices+layout.sendsize, [&](int & index){
    	(*outdata)[index] += recvbuf[i];
     	i++;
    });
}

template <typename T, typename intT>
void lpaint(T** pos, // Positions in mesh space (Nparts, 3)
	    intT Nparts, // Number of particles
	    T* mass, // Mass of each particle
	    intT ex[2], // Limits along x of grid for local part of the field
	    intT ey[2], // Along y
	    intT ez[2], // Along z
	    intT* Nmesh, // Mesh sizes, needed for ghosts
	    T** field){ // Output field, flattened (row-major order)
    // Computes the local part of the field using CIC, assuming the
    // positions in "pos" are the result of an "exchange" operation.

    intT szx = ex[1] - ex[0];
    intT szy = ey[1] - ey[0];
    intT szz = ez[1] - ez[0];
    T dx, dy, dz, tx, ty, tz;
    int i, j, k, ip1, jp1, kp1;
    bool inx, iny, inz, inpx, inpy, inpz;
    T pmass;
    int p = 0;
    for_each(pos, pos+Nparts, [&](T* & part){
	i = int(part[0]); // Should already be ex[0] <= part[0] < ex[1] unless ghost
	j = int(part[1]);
	k = int(part[2]); 
	dx = part[0] - i;
	dy = part[1] - j;
	dz = part[2] - k;
	tx = 1 - dx;
	ty = 1 - dy;
	tz = 1 - dz;

	// Neighbors for CIC
	ip1 = (i+1) % Nmesh[0];
	jp1 = (j+1) % Nmesh[1];
	kp1 = (k+1) % Nmesh[2];
	// The following bools are for ghosts
	inx = (unsigned)(i - ex[0]) < szx; // If false, ghost from below
        iny = (unsigned)(j - ey[0]) < szy; // Just checking if it's in range in an efficient way
        inz = (unsigned)(k - ez[0]) < szz;
        inpx = (unsigned)(ip1 - ex[0]) < szx; // If false, ghost from above
        inpy = (unsigned)(jp1 - ey[0]) < szy;
        inpz = (unsigned)(kp1 - ez[0]) < szz;
	i = abs((i - ex[0]) % szx); // Make sure value is always in limits of out field
	j = abs((j - ey[0]) % szy); // If it's negative it's a ghost, so don't care
	k = abs((k - ez[0]) % szz); // Bools deal with setting the wrong contribution of ghosts to 0
	ip1 = abs((ip1 - ex[0]) % szx);
	jp1 = abs((jp1 - ey[0]) % szy);
	kp1 = abs((kp1 - ez[0]) % szz);
	pmass = mass[p];
	// CIC update
	(*field)[i*szy*szz + j*szz + k] += tx*ty*tz*pmass * inx*iny*inz;
	(*field)[ip1*szy*szz + j*szz + k] += dx*ty*tz*pmass * inpx*iny*inz;
	(*field)[i*szy*szz + jp1*szz + k] += tx*dy*tz*pmass * inx*inpy*inz;
	(*field)[i*szy*szz + j*szz + kp1] += tx*ty*dz*pmass * inx*iny*inpz;
	(*field)[ip1*szy*szz + jp1*szz + k] += dx*dy*tz*pmass * inpx*inpy*inz;
	(*field)[ip1*szy*szz + j*szz + kp1] += dx*ty*dz*pmass * inpx*iny*inpz;
	(*field)[i*szy*szz + jp1*szz + kp1] += tx*dy*dz*pmass * inx*inpy*inpz;
	(*field)[ip1*szy*szz + jp1*szz + kp1] += dx*dy*dz*pmass * inpx*inpy*inpz;
    	p++;
    });
}

template <typename T, typename intT>
void lreadout(T** pos, // Positions in mesh space (Nparts, 3)
	      intT Nparts, // Number of particles
              T* field, // Field to read from (flattened, row-major order)
	      intT ex[2], // Limits along x of grid for local part of the field
	      intT ey[2], // Along y
	      intT ez[2], // Along z
	      intT* Nmesh, // Mesh sizes, needed for ghosts
	      T* BoxSize, // Size of simulation box in each direction, needed for derivative
	      T** outmass, // Output mass ptr
	      int32_t vjpdim=-1){ // Dimension along which derivative is computed
    // Computes the CIC interpolation in the local part of the field, assuming
    // the positions in "pos" are the result of an "exchange" operation.
    // Very similar to lpaint
    // BoxSize and dim are used to define vjps, see notes for how

    intT szx = ex[1] - ex[0];
    intT szy = ey[1] - ey[0];
    intT szz = ez[1] - ez[0];
    int i, j, k, ip1, jp1, kp1;
    bool inx, iny, inz, inpx, inpy, inpz;
    T mass;
    T ds[3], ts[3];
    #pragma omp parallel for
    for(int p=0; p<Nparts; p++){
        i = int(pos[p][0]); // Should already be ex[0] <= part[0] < ex[1] unless ghost
        j = int(pos[p][1]);
        k = int(pos[p][2]);
	// Defining distances like this here for convenience
        ds[0] = pos[p][0] - i;
	ds[1] = pos[p][1] - j;
       	ds[2] = pos[p][2] - k;
	ts[0] = 1 - ds[0];
	ts[1] = 1 - ds[1];
	ts[2] = 1 - ds[2];
	if (vjpdim >= 0){ // For vjp, see notes
	    ds[vjpdim] = Nmesh[vjpdim] / BoxSize[vjpdim]; // 1/cellsize
	    ts[vjpdim] = -ds[vjpdim];
	}

        // Neighbors for CIC
	ip1 = (i+1) % Nmesh[0];
	jp1 = (j+1) % Nmesh[1];
        kp1 = (k+1) % Nmesh[2];
        // The following bools are for ghosts
        inx = (unsigned)(i - ex[0]) < szx; // If false, ghost from below
        iny = (unsigned)(j - ey[0]) < szy; // Just checking if it's in range
        inz = (unsigned)(k - ez[0]) < szz;
        inpx = (unsigned)(ip1 - ex[0]) < szx; // If false, ghost from above
        inpy = (unsigned)(jp1 - ey[0]) < szy;
        inpz = (unsigned)(kp1 - ez[0]) < szz;
        i = abs((i - ex[0]) % szx); // Make sure value is always in limits of out field
        j = abs((j - ey[0]) % szy); // If it's negative it's a ghost, so don't care
        k = abs((k - ez[0]) % szz); // Bools deal with setting the wrong contribution of ghosts to 0
        ip1 = abs((ip1 - ex[0]) % szx);
        jp1 = abs((jp1 - ey[0]) % szy);
        kp1 = abs((kp1 - ez[0]) % szz);
	// CIC update
        mass = ts[0]*ts[1]*ts[2]*field[i*szy*szz + j*szz + k] * inx*iny*inz +
               ds[0]*ts[1]*ts[2]*field[ip1*szy*szz + j*szz + k] * inpx*iny*inz +
	       ts[0]*ds[1]*ts[2]*field[i*szy*szz + jp1*szz + k] * inx*inpy*inz +
               ts[0]*ts[1]*ds[2]*field[i*szy*szz + j*szz + kp1] * inx*iny*inpz +
               ds[0]*ds[1]*ts[2]*field[ip1*szy*szz + jp1*szz + k] * inpx*inpy*inz +
               ds[0]*ts[1]*ds[2]*field[ip1*szy*szz + j*szz + kp1] * inpx*iny*inpz +
               ts[0]*ds[1]*ds[2]*field[i*szy*szz + jp1*szz + kp1] * inx*inpy*inpz +
               ds[0]*ds[1]*ds[2]*field[ip1*szy*szz + jp1*szz + kp1] * inpx*inpy*inpz;
	(*outmass)[p] = mass;
    }
}

template <typename T, typename intT>
void ppaint(T**& pos, // (Nparts, 3) Positions (in mesh space)
            int64_t& Nparts, // Number of particles
	    T*& mass, // Particle masses
            intT*& Nmesh, // Mesh grid specs (3, )
            intT*& edgesx, // Limiting grid indices along x for each rank, (comm_size, 2) flattened
            intT*& edgesy, // Same, for y
            intT*& edgesz, // Same, for z
            T** outptr, // Output pointer (flattened field, row-major order)
	    intT* sizeptr, // Output size pointer, for convenience
	    Layout* layoutptr, // Output layout (store it so that it's already defined)
	    T*** epos, // Positions exchanged with layout, to pass them straight to preadout
	    MPI_Comm& comm){ // MPI communicator
    // Parallel paint
    // Moves particles with relative masses from all ranks around ranks to bring those
    // contributing to local part of the field (as specified in edges) in this rank,
    // then computes local part of field using CIC. Also deals with ghosts, which are
    // particles at the edges of the field limits that only partially contribute here
    // Stores resulting local part of density field in outptr, as well as
    // decomposition layout and exchanged positions to make preadout faster
    
    int comm_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

    // Find layout decomposition
    Layout layout;
    decompose<T, intT>(pos, Nparts, Nmesh, edgesx, edgesy, edgesz, &layout, comm);
    (*layoutptr) = layout; // Store layout
    
    // Exchange positions and mass
    T** outpos;
    exchange<T>(pos, 3, layout, &outpos, comm);
    (*epos) = outpos; // Store exchanged positions

    T* outmass;
    exchange1D<T>(mass, layout, &outmass, comm);

    // Now paint local field
    intT* ex = &(edgesx[2*comm_rank]); // Edges for this rank
    intT* ey = &(edgesy[2*comm_rank]);
    intT* ez = &(edgesz[2*comm_rank]);
    int outsize = (ex[1] - ex[0])*(ey[1] - ey[0])*(ez[1] - ez[0]);
    (*sizeptr) = outsize;
    (*outptr) = new T[outsize];
    for (int i=0; i<outsize; i++) (*outptr)[i] = 0;
    // Paint locally
    lpaint<T, intT>(outpos, layout.recvsize, outmass, ex, ey, ez, Nmesh, outptr);
}

template <typename T, typename intT>
void preadout(T**& epos, // (?, 3) Exchanged positions (in mesh space) PPAINT CALL NEEDED
              int64_t& Nparts, // Number of particles
              Layout& layout, // Decomposition layout PPAINT CALL NEEDED
	      T*& localfield, // Field to readout from (flattened, row-major order)
              intT*& Nmesh, // Mesh grid specs (3, )
	      T*& BoxSize, // Needed for vjp
              intT*& edgesx, // Limiting grid indices along x for each rank, (comm_size, 2) flattened
              intT*& edgesy, // Same, for y
              intT*& edgesz, // Same, for z
              T** outptr, // Output pointer
              MPI_Comm& comm, // MPI communicator
	      int32_t& vjpdim){ // Also needed for vjp
    // Parallel readout
    // NOTE: THIS EXPLOITS AVAILABLE DECOMPOSITION LAYOUT, SO CALL PPAINT FIRST
    // Interpolates local field using CIC. Particle positions are assumed to be already
    // the ones which the local field contributes to. Sums ghosts at the end
    // Stores resulting local part of density field in outptr
        
    int comm_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

    // Readout at exchanged positions
    intT* ex = &(edgesx[2*comm_rank]); // Edges for this rank
    intT* ey = &(edgesy[2*comm_rank]);
    intT* ez = &(edgesz[2*comm_rank]);
    // Readout locally
    T* outmass = new T[layout.recvsize];
    lreadout<T, intT>(epos, layout.recvsize, localfield, ex, ey, ez, Nmesh, BoxSize, &outmass, vjpdim);

    // Now gather the data back into its original rank
    T* mass = new T[Nparts];
    for (int i=0; i<Nparts; i++) mass[i] = 0;
    gather1D(outmass, layout, &mass, comm);
    (*outptr) = mass;
}

// NOT NEEDED, BUT KEEPING FOR SAFETY
template <typename T, typename intT>
void preadout_nolayout(T**& pos, // (Nparts, 3) Particle positions (in mesh space)
              int64_t& Nparts, // Number of particles
              T*& localfield, // Field to readout from (flattened, z fastest dimension)
              intT*& Nmesh, // Mesh grid specs (3, )
              T*& BoxSize, // Needed for vjp
              intT*& edgesx, // Limiting grid indices along x for each rank, (comm_size, 2) flattened
              intT*& edgesy, // Same, for y
              intT*& edgesz, // Same, for z
              T** outptr, // Output pointer
              MPI_Comm& comm, // MPI communicator
              int64_t& vjpdim){ // Also needed for vjp
    // Parallel readout
    // Moves particles with relative masses from all ranks around ranks to bring them
    // to the rank(s) hosting the part of the field contributing to their mass, interpolates
    // using CIC the mass of the point. Sums ghosts at the end
    // Stores resulting local part of density field in outptr

    int comm_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

    // Find decomposition layout
    Layout layout;
    decompose<T, intT>(pos, Nparts, Nmesh, edgesx, edgesy, edgesz, &layout, comm);

    // Exchange positions
    T** outpos;
    exchange<T>(pos, 3, layout, &outpos, comm);

    // Now readout at exchanged positions
    intT* ex = &(edgesx[2*comm_rank]); // Edges for this rank
    intT* ey = &(edgesy[2*comm_rank]);
    intT* ez = &(edgesz[2*comm_rank]);
    // Readout locally
    T* outmass = new T[layout.recvsize];
    lreadout<T, intT>(outpos, layout.recvsize, localfield, ex, ey, ez, Nmesh, BoxSize, &outmass, vjpdim);

    // Now gather the data back into its original rank
    T* mass = new T[Nparts];
    for (int i=0; i<Nparts; i++) mass[i] = 0;
    gather1D(outmass, layout, &mass, comm);
    (*outptr) = mass;
}

} // namespace parops_jax

#endif

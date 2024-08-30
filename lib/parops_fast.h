#ifndef _JAXPAROPS_H_
#define _JAXPAROPS_H_

// NOTE: ALL FLATTENINGS ARE ASSUMED TO BE ROW-MAJOR

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
    int* indices = new int[10]; // Particles indices per rank, stacked vertically (except self)
    int* selfindices = new int[10]; // Particles in this rank belonging to itself (no need to move)
    int selfsize;
    // Parameters for sending data using alltoallv
    int sendsize;
    int* sendcounts;
    int* senddispl;
    // Parameters for receiving data using alltoallv
    int recvsize;
    int* recvcounts;
    int* recvdispl;
    // Total sizes
    int totsend;
    int totrecv;
};

void initLayout(Layout* layout, int comm_size){
    // Initializes layout parameters of input layout
    layout->sendcounts = new int[comm_size];
    layout->senddispl = new int[comm_size];
    layout->recvcounts = new int[comm_size];
    layout->recvdispl = new int[comm_size];
}

template <typename T>
void cleanpops(Layout* layout, T** pos){
    // Cleans stored data to free memory of layout and exchanged positions
    delete[] layout->indices;
    delete[] (*pos);
}

// Passing big inputs by reference to save memory
template <typename T, typename intT>
void decompose(T*& pos, // (Nparts * 3) Flattened positions (in mesh space)
	       int64_t Nparts, // Number of particles
	       intT* Nmesh, // (3, ) Mesh grid specs
	       intT* edgesx, // (comm_size, 2) Flattened limit grid indices along x for each rank
	       intT* edgesy, // Same, for y
	       intT* edgesz, // Same, for z
	       Layout* layout, // Layout variable to store decomposition
	       MPI_Comm comm=MPI_COMM_WORLD){ // MPI communicator
    // Creates a decomposition layout based on the given positions and
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

    int comm_size, comm_rank;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);

    // Array of vectors with particle indices for each rank (vector because size is unknown a priori)
    vector<int>* ppr = new vector<int>[comm_size];
    for (int rank=0; rank<comm_size; rank++){
        ppr[rank].reserve(Nparts / comm_size); // Initialize vec to a sensible size to avoid realloc
	layout->sendcounts[rank] = 0;
    }
    // Indices of particles in this rank
    vector<int> pself;
    pself.reserve(Nparts);
    layout->selfsize = 0;

    int i, j, k, i1, j1, k1;
    for (int64_t p=0; p<Nparts; p++){
	i = int(pos[3*p]) % Nmesh[0]; // Closest grid point to position towards axes origin
	j = int(pos[3*p+1]) % Nmesh[1];
	k = int(pos[3*p+2]) % Nmesh[2];
	for (int rank=0; rank<comm_size; rank++){ // For each rank, check every CIC direction
	    for (int idx=0; idx<8; idx++){
	    	i1 = (i + ds[idx][0]) % Nmesh[0];
            	j1 = (j + ds[idx][1]) % Nmesh[1];
            	k1 = (k + ds[idx][2]) % Nmesh[2];
		if ((unsigned)(i1 - edgesx[2*rank]) < (edgesx[2*rank+1] - edgesx[2*rank]) &&
                    (unsigned)(j1 - edgesy[2*rank]) < (edgesy[2*rank+1] - edgesy[2*rank]) &&
                    (unsigned)(k1 - edgesz[2*rank]) < (edgesz[2*rank+1] - edgesz[2*rank])){    
		    if (rank == comm_rank){
			pself.push_back(p); // Particles in this rank that belong to this rank
			layout->selfsize++;
			break;
		    } else {
		    	ppr[rank].push_back(p); // Add particle index to this rank
		    	layout->sendcounts[rank]++; // Increment this rank counter
		   	break; // If this particle contributes to the rank, just exit (no doubles)
		    }
		}
	    }
	}
    }

    // Update the self indices in the layout
    //pself.shrink_to_fit();
    delete[] layout->selfindices;
    layout->selfindices = new int[layout->selfsize];
    for (int i=0; i<layout->selfsize; i++) layout->selfindices[i] = pself[i];
    vector<int>().swap(pself); // Free vector memory
    
    // Layout counts and displacements should be already initialized
    // Send to each rank the size of the data coming from this
    MPI_Alltoall(layout->sendcounts, 1, MPI_INT, layout->recvcounts, 1, MPI_INT, comm);

    // Define the send and receive sizes
    layout->sendsize = 0;
    layout->recvsize = 0;
    for (int rank=0; rank<comm_size; rank++){
	layout->sendsize += layout->sendcounts[rank];
	layout->recvsize += layout->recvcounts[rank];
	//ppr[rank].shrink_to_fit(); // Just to avoid memory spike when creating also indices
    }
    //cout << layout->sendsize << " " << layout-> recvsize << endl; // For testing memeory

    // Indices of particles for each rank, stacked vertically
    delete[] layout->indices; // Delete the old one
    layout->indices = new int[layout->sendsize];
    int ctr = 0;
    for (int rank=0; rank<comm_size; rank++){
        for (int p : ppr[rank]){
  	    layout->indices[ctr] = p;
	    ctr++;
	}
	vector<int>().swap(ppr[rank]); // Free vector memory
    }
    delete[] ppr;

    // Compute send and receive displacements for alltoallv
    layout->senddispl[0] = 0;
    layout->recvdispl[0] = 0;
    for (int rank=1; rank<comm_size; rank++){
	layout->senddispl[rank] = layout->senddispl[rank-1] + layout->sendcounts[rank-1];
	layout->recvdispl[rank] = layout->recvdispl[rank-1] + layout->recvcounts[rank-1];
    }
	
    // Particles from own rank + from other ranks
    layout->totsend=layout->selfsize + layout->sendsize;
    layout->totrecv=layout->selfsize + layout->recvsize;
}

template <typename T>
void exchange(T*& data, int dims, Layout& layout, T** outdata, MPI_Comm comm=MPI_COMM_WORLD){
    // Sends what's in "data" (flattened) to the correct ranks to perform CIC operations.
    // dims is the dimensionality of data
    // layout is the decomposition layout.
    // Sets the value in outdata to the "data" each rank have assigned to this (including self).

    // Construct the array that contains the data to send
    T* sendbuf = new T[layout.sendsize * dims];
    for (int i=0; i<layout.sendsize; i++){
	 for (int d=0; d<dims; d++) sendbuf[dims*i + d] = data[dims*layout.indices[i] + d];
    }

    // Construct the receive buffer (because this is only particles from other ranks)
    T* recvbuf = new T[layout.recvsize * dims];

    // Create datatype to send (contiguous T of dim length)
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
		  recvbuf,
		  layout.recvcounts,
		  layout.recvdispl,
		  dt,
		  comm);

    // Now build the output (particles from this rank + from other ranks)
    for (int i=0; i<layout.selfsize; i++){
         for (int d=0; d<dims; d++) (*outdata)[dims*i + d] = data[dims*layout.selfindices[i] + d];
    }
    for (int i=0; i<dims*layout.recvsize; i++){
         (*outdata)[dims*layout.selfsize + i] = recvbuf[i];
    }

    delete[] sendbuf;
    delete[] recvbuf;
    MPI_Type_free(&dt);
}

template <typename T>
void exchange1D(T*& data, Layout& layout, T** outdata, MPI_Comm comm=MPI_COMM_WORLD){
    // 1D version of above for convenience
    
    T* sendbuf = new T[layout.sendsize];
    for (int i=0; i<layout.sendsize; i++) sendbuf[i] = data[layout.indices[i]];

    T* recvbuf = new T[layout.recvsize];

    MPI_Datatype dt;
    if (is_same<T, double>::value){
        dt = MPI_DOUBLE;
    } else if (is_same<T, float>::value){
        dt = MPI_FLOAT;
    } else {
        throw invalid_argument("Data type not supported");
    }
    MPI_Type_commit(&dt);

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

    for (int i=0; i<layout.selfsize; i++){
         (*outdata)[i] = data[layout.selfindices[i]];
    }
    for (int i=0; i<layout.recvsize; i++){
         (*outdata)[layout.selfsize + i] = recvbuf[i];
    }

    delete[] sendbuf;
    delete[] recvbuf;
    MPI_Type_free(&dt);
}

template <typename T>
void gather1D(T*& data, Layout& layout, T** outdata, MPI_Comm comm=MPI_COMM_WORLD){
    // Gathers decomposed data, sending it back to the original rank,
    // and summing contributions of ghosts, based on decomposition layout
    
    // This time we're receiving back original data, so receiving and sending parameters are flipped
    T* recvbuf = new T[layout.sendsize];

    // Build send buffer to only contain things to send to other ranks
    T* sendbuf = new T[layout.recvsize];
    for (int i=0; i<layout.recvsize; i++){
         sendbuf[i] = data[layout.selfsize + i];
    }

    // Choose datatype to send based on T
    MPI_Datatype dt;
    if (is_same<T, double>::value){
        dt = MPI_DOUBLE;
    } else if (is_same<T, float>::value){
        dt = MPI_FLOAT;
    } else {
        throw invalid_argument("Data type not supported");
    }
    MPI_Type_commit(&dt);

    // Fire back
    MPI_Alltoallv(sendbuf,
		  layout.recvcounts,
		  layout.recvdispl,
		  dt,
		  recvbuf,
		  layout.sendcounts,
		  layout.senddispl,
		  dt,
		  comm);

    // Sum contributions of different ranks to each particle (with index indices[i])
    for (int i=0; i<layout.selfsize; i++){
	(*outdata)[layout.selfindices[i]] += data[i];
    }
    for (int i=0; i<layout.sendsize; i++){
        (*outdata)[layout.indices[i]] += recvbuf[i];
    }
    delete[] sendbuf;
    delete[] recvbuf;
    MPI_Type_free(&dt);
}

template <typename T>
void gather(T*& data, int dims, Layout& layout, T** outdata, MPI_Comm comm=MPI_COMM_WORLD){
    // Gathers decomposed data, sending it back to the original rank,
    // and summing contributions of ghosts, based on decomposition layout
    // This is a ND version of above, in case one uses the 3D readout

    // This time we're receiving back original data, so receiving and sending parameters are flipped
    T* recvbuf = new T[layout.sendsize * dims];

    // Build send buffer because we don't gather all data, only the one
    T* sendbuf = new T[layout.recvsize * dims];
    for (int i=0; i<layout.recvsize; i++){
         for (int d=0; d<dims; d++) sendbuf[dims*i + d] = data[dims*(layout.selfsize + i) + d];
    }

    // Choose datatype to send based on T
    MPI_Datatype dt;
    if (is_same<T, double>::value){
        MPI_Type_contiguous(dims, MPI_DOUBLE, &dt);
    } else if (is_same<T, float>::value){
        MPI_Type_contiguous(dims, MPI_FLOAT, &dt);
    } else {
        throw invalid_argument("Data type not supported");
    }
    MPI_Type_commit(&dt);

    // Fire back
    MPI_Alltoallv(sendbuf,
                  layout.recvcounts,
                  layout.recvdispl,
                  dt,
                  recvbuf,
                  layout.sendcounts,
                  layout.senddispl,
                  dt,
                  comm);

    // Sum contributions of different ranks to each particle (particles are in rows and row-major)
    for(int i=0; i<layout.selfsize; i++){
        for (int d=0; d<dims; d++) (*outdata)[dims*layout.selfindices[i] + d] += data[dims*i + d];
    }
    for (int i=0; i<layout.sendsize; i++){
         for (int d=0; d<dims; d++) (*outdata)[dims*layout.indices[i] + d] += recvbuf[dims*i + d];
    }
    delete[] sendbuf;
    delete[] recvbuf;
    MPI_Type_free(&dt);
}

template <typename T, typename intT>
void lpaint(T*& pos, // (Nparts * 3) Flattened positions in mesh space
	    intT Nparts, // Number of particles
	    T*& mass, // Mass of each particle
	    intT ex[2], // Limits along x of grid for local part of the field
	    intT ey[2], // Along y
	    intT ez[2], // Along z
	    intT* Nmesh, // Mesh sizes, needed for ghosts
	    T** field){ // Flattened output field
    // Computes the local part of the field using CIC, assuming the
    // positions in "pos" are the result of an "exchange" operation.

    intT szx = ex[1] - ex[0];
    intT szy = ey[1] - ey[0];
    intT szz = ez[1] - ez[0];
    T dx, dy, dz, tx, ty, tz;
    int i, j, k, ip1, jp1, kp1;
    bool inx, iny, inz, inpx, inpy, inpz;
    T pmass;
    for (int p=0; p<Nparts; p++){
	i = int(pos[3*p]); // Should already be ex[0] <= part[0] < ex[1] unless ghost
	j = int(pos[3*p+1]);
	k = int(pos[3*p+2]); 
	dx = pos[3*p] - i;
	dy = pos[3*p+1] - j;
	dz = pos[3*p+2] - k;
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
    }
}

template <typename T, typename intT>
void lreadout(T*& pos, // (Nparts * 3) Flattened positions in mesh space
	      intT Nparts, // Number of particles
              T*& field, // Flattened local field to read from
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
    //#pragma omp parallel for // Doesn't make any difference and intel doesn't like it anyway
    for(int p=0; p<Nparts; p++){
        i = int(pos[3*p]); // Should already be ex[0] <= part[0] < ex[1] unless ghost
        j = int(pos[3*p+1]);
        k = int(pos[3*p+2]);
	// Defining distances like this here for convenience
        ds[0] = pos[3*p] - i;
	ds[1] = pos[3*p+1] - j;
       	ds[2] = pos[3*p+2] - k;
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
void ppaint(T*& pos, // (Nparts * 3) Flattened positions (in mesh space)
            int64_t& Nparts, // Number of particles
	    T*& mass, // Particle masses
            intT*& Nmesh, // (3, ) Mesh grid specs
            intT*& edgesx, // (comm_size, 2) Flattened limiting grid indices along x for each rank
            intT*& edgesy, // Same, for y
            intT*& edgesz, // Same, for z
            T** outptr, // Flattened output pointer
	    Layout* layout, // Output layout (to store it)
	    T** expos, // Exchanged particle positions (to store it)
	    bool& useLayout, // Use pre-defined decomposition layout or no
	    MPI_Comm& comm){ // MPI communicator
    // Parallel paint
    // Moves particles with relative masses from all ranks around ranks to bring those
    // contributing to local part of the field (as specified in edges) in this rank,
    // then computes local part of field using CIC. Also deals with ghosts, which are
    // particles at the edges of the field limits that only partially contribute here
    // Stores resulting local part of density field in outptr, as well as
    // decomposition layout and exchanged positions to make preadout faster
    // Optionally does CIC using pre-existing decomposition layout and exchanged positions
    
    int comm_rank;
    MPI_Comm_rank(comm, &comm_rank);

    if (!useLayout){
    	// Find layout decomposition
    	decompose<T, intT>(pos, Nparts, Nmesh, edgesx, edgesy, edgesz, layout, comm);
    
    	// Exchange positions
	delete[] (*expos); // Delete the previous one
    	(*expos) = new T[layout->totrecv * 3];
    	exchange<T>(pos, 3, *layout, expos, comm);
    }
	
    // Exchange mass
    T* exdmass = new T[layout->totrecv];
    exchange1D<T>(mass, *layout, &exdmass, comm);

    // Now paint local field
    intT* ex = &(edgesx[2*comm_rank]); // Edges for this rank
    intT* ey = &(edgesy[2*comm_rank]);
    intT* ez = &(edgesz[2*comm_rank]);
    int outsize = (ex[1] - ex[0])*(ey[1] - ey[0])*(ez[1] - ez[0]);
    for (int i=0; i<outsize; i++) (*outptr)[i] = 0;
    // Paint locally
    lpaint<T, intT>(*expos, layout->totrecv, exdmass, ex, ey, ez, Nmesh, outptr);
    delete[] exdmass;
}

template <typename T, typename intT>
void preadout(Layout& layout, // Decomposition layout PPAINT CALL NEEDED
	      T*& expos, // Exchanged positions PPAINT CALL NEEDED
	      int64_t& Nparts, // Number of particles
	      T*& localfield, // Flattened local field to readout from
              intT*& Nmesh, // (3, ) Mesh grid specs
	      T*& BoxSize, // Needed for vjp
              intT*& edgesx, // Flattened (comm_size, 2) limiting grid indices along x for each rank
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
    MPI_Comm_rank(comm, &comm_rank);

    // Readout at exchanged positions
    intT* ex = &(edgesx[2*comm_rank]); // Edges for this rank
    intT* ey = &(edgesy[2*comm_rank]);
    intT* ez = &(edgesz[2*comm_rank]);
    // Readout locally
    T* outmass = new T[layout.totrecv];
    lreadout<T, intT>(expos, layout.totrecv, localfield, ex, ey, ez, Nmesh, BoxSize, &outmass, vjpdim);

    // Now gather the data back into its original rank
    for (int i=0; i<Nparts; i++) (*outptr)[i] = 0;
    gather1D(outmass, layout, outptr, comm);
    delete[] outmass;
}

// NOT NEEDED, BUT KEEPING FOR SAFETY (ALSO NOT UPDATED)
/*template <typename T, typename intT>
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
    MPI_Comm_rank(comm, &comm_rank);

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
    delete[] outmass;
    (*outptr) = mass;
}*/

template <typename T, typename intT>
void lpaint3D(T*& pos, // (Nparts * 3) Flattened positions in mesh space
            intT Nparts, // Number of particles
            T*& mass, // Mass of each particle * 3 (flattened from (Nparts, 3))
            intT ex[2], // Limits along x of grid for local part of the field
            intT ey[2], // Along y
            intT ez[2], // Along z
            intT* Nmesh, // Mesh sizes, needed for ghosts
            T** field){ // Flattened output field, thrice
    // Computes the local part of the field using CIC, assuming the
    // positions in "pos" are the result of an "exchange" operation.

    intT szx = ex[1] - ex[0];
    intT szy = ey[1] - ey[0];
    intT szz = ez[1] - ez[0];
    intT size = szx * szy * szz;
    T dx, dy, dz, tx, ty, tz;
    int i, j, k, ip1, jp1, kp1;
    bool inx, iny, inz, inpx, inpy, inpz;
    T pmass;
    for (int p=0; p<Nparts; p++){
        i = int(pos[3*p]); // Should already be ex[0] <= part[0] < ex[1] unless ghost
        j = int(pos[3*p+1]);
        k = int(pos[3*p+2]);
        dx = pos[3*p] - i;
        dy = pos[3*p+1] - j;
        dz = pos[3*p+2] - k;
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
	for (int d=0; d<3; d++){
            pmass = mass[3*p + d];
            // CIC update
            (*field)[d*size + i*szy*szz + j*szz + k] += tx*ty*tz*pmass * inx*iny*inz;
            (*field)[d*size + ip1*szy*szz + j*szz + k] += dx*ty*tz*pmass * inpx*iny*inz;
            (*field)[d*size + i*szy*szz + jp1*szz + k] += tx*dy*tz*pmass * inx*inpy*inz;
            (*field)[d*size + i*szy*szz + j*szz + kp1] += tx*ty*dz*pmass * inx*iny*inpz;
            (*field)[d*size + ip1*szy*szz + jp1*szz + k] += dx*dy*tz*pmass * inpx*inpy*inz;
            (*field)[d*size + ip1*szy*szz + j*szz + kp1] += dx*ty*dz*pmass * inpx*iny*inpz;
            (*field)[d*size + i*szy*szz + jp1*szz + kp1] += tx*dy*dz*pmass * inx*inpy*inpz;
            (*field)[d*size + ip1*szy*szz + jp1*szz + kp1] += dx*dy*dz*pmass * inpx*inpy*inpz;
	}
    }
}

template <typename T, typename intT>
void ppaint3D(T*& pos, // (Nparts * 3) Flattened positions (in mesh space)
            int64_t& Nparts, // Number of particles
            T*& mass, // Particle masses (flattened from (Nparts, 3))
            intT*& Nmesh, // (3, ) Mesh grid specs
            intT*& edgesx, // (comm_size, 2) Flattened limiting grid indices along x for each rank
            intT*& edgesy, // Same, for y
            intT*& edgesz, // Same, for z
            T** outptr, // Flattened output pointer (3 * local_size)
            Layout* layout, // Output layout (to store it)
            T** expos, // Exchanged particle positions (to store it)
            MPI_Comm& comm){ // MPI communicator
    // 3D VERSION
    // Parallel paint
    // Moves particles with relative masses from all ranks around ranks to bring those
    // contributing to local part of the field (as specified in edges) in this rank,
    // then computes local part of field using CIC. Also deals with ghosts, which are
    // particles at the edges of the field limits that only partially contribute here
    // Stores resulting local part of density field in outptr, as well as
    // decomposition layout and exchanged positions to make preadout faster
    // Optionally does CIC using pre-existing decomposition layout and exchanged positions

    int comm_rank;
    MPI_Comm_rank(comm, &comm_rank);

    // Find layout decomposition
    decompose<T, intT>(pos, Nparts, Nmesh, edgesx, edgesy, edgesz, layout, comm);

    // Exchange positions
    delete[] (*expos); // Delete the previous one
    *expos = new T[layout->totrecv * 3];
    exchange<T>(pos, 3, *layout, expos, comm);

    // Exchange mass
    T* exdmass = new T[3 * layout->totrecv];
    exchange<T>(mass, 3, *layout, &exdmass, comm);

    // Now paint local field
    intT* ex = &(edgesx[2*comm_rank]); // Edges for this rank
    intT* ey = &(edgesy[2*comm_rank]);
    intT* ez = &(edgesz[2*comm_rank]);
    // Field is 3 times as big now
    int outsize = 3*(ex[1] - ex[0])*(ey[1] - ey[0])*(ez[1] - ez[0]);
    for (int i=0; i<outsize; i++) (*outptr)[i] = 0;
    // Paint locally
    lpaint3D<T, intT>(*expos, layout->totrecv, exdmass, ex, ey, ez, Nmesh, outptr);
    delete[] exdmass;
}

// TODO
// COULD ALSO USE SINGLE FUNCTION BOTH FOR READOUT OF 1 FIELD AND 3 FIELDS + VJP
template <typename T, typename intT>
void lreadout3D(T*& pos, // (Nparts * 3) Flattened positions in mesh space
                intT Nparts, // Number of particles
                T*& field, // Flattened local field(s) to read from (can be Nmesh ** 3 or 3 * Nmesh**3)
                intT ex[2], // Limits along x of grid for local part of the field
                intT ey[2], // Along y
                intT ez[2], // Along z
                intT* Nmesh, // Mesh sizes, needed for ghosts
                T* BoxSize, // Size of simulation box in each direction, needed for derivative
                T** outmass, // Output mass ptr
                bool vjp){ // Whether this is needed to compute 3D vjp or not
    // Computes either CIC interpolation for 3 local fields or 3 jvp components
    // for 1 local field, assuming the positions in "pos" are the result of an
    // "exchange" operation. BoxSize is only used in case of vjp

    intT szx = ex[1] - ex[0];
    intT szy = ey[1] - ey[0];
    intT szz = ez[1] - ez[0];
    intT size = szx * szy * szz;
    int i, j, k, ip1, jp1, kp1;
    bool inx, iny, inz, inpx, inpy, inpz;
    T mass;
    T ds[2][3], ts[2][3];
    // Stuff for vjp
    int sel[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}; // This becomes the identity for vjp
    ds[1][0] = Nmesh[0] / BoxSize[0]; // 1/cellsize[0]
    ds[1][1] = Nmesh[1] / BoxSize[1]; // 1/cellsize[1]
    ds[1][2] = Nmesh[2] / BoxSize[2]; // 1/cellsize[2]
    ts[1][0] = -ds[1][0];
    ts[1][1] = -ds[1][1];
    ts[1][2] = -ds[1][2];
    int offmtp = 1; // Multiplier in case no vjp (because field is 3 times as big)
    if (vjp){
        sel[0][0] = 1; // This way we'll select ds[1][..] for dim 0, 1 and 2 when looping over d
        sel[1][1] = 1;
        sel[2][2] = 1;
        offmtp = 0;
    }
    //#pragma omp parallel for // Doesn't make any difference and intel doesn't like it anyway
    for(int p=0; p<Nparts; p++){
        i = int(pos[3*p]); // Should already be ex[0] <= part[0] < ex[1] unless ghost
        j = int(pos[3*p+1]);
        k = int(pos[3*p+2]);
        // Defining distances like this here for convenience
        ds[0][0] = pos[3*p] - i;
        ds[0][1] = pos[3*p+1] - j;
        ds[0][2] = pos[3*p+2] - k;
        ts[0][0] = 1 - ds[0][0];
        ts[0][1] = 1 - ds[0][1];
        ts[0][2] = 1 - ds[0][2];

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
	for (int d=0; d<3; d++){
	    // Fields are stacked as (3, size), row major, so [field1, field2, field3]
	    mass = ts[sel[d][0]][0]*ts[sel[d][1]][1]*ts[sel[d][2]][2] *
		   field[d*size*offmtp + i*szy*szz + j*szz + k] * inx*iny*inz +
                   ds[sel[d][0]][0]*ts[sel[d][1]][1]*ts[sel[d][2]][2] *
		   field[d*size*offmtp + ip1*szy*szz + j*szz + k] * inpx*iny*inz +
                   ts[sel[d][0]][0]*ds[sel[d][1]][1]*ts[sel[d][2]][2] *
		   field[d*size*offmtp + i*szy*szz + jp1*szz + k] * inx*inpy*inz +
                   ts[sel[d][0]][0]*ts[sel[d][1]][1]*ds[sel[d][2]][2] *
		   field[d*size*offmtp + i*szy*szz + j*szz + kp1] * inx*iny*inpz +
                   ds[sel[d][0]][0]*ds[sel[d][1]][1]*ts[sel[d][2]][2] *
		   field[d*size*offmtp + ip1*szy*szz + jp1*szz + k] * inpx*inpy*inz +
                   ds[sel[d][0]][0]*ts[sel[d][1]][1]*ds[sel[d][2]][2] *
		   field[d*size*offmtp + ip1*szy*szz + j*szz + kp1] * inpx*iny*inpz +
                   ts[sel[d][0]][0]*ds[sel[d][1]][1]*ds[sel[d][2]][2] *
		   field[d*size*offmtp + i*szy*szz + jp1*szz + kp1] * inx*inpy*inpz +
                   ds[sel[d][0]][0]*ds[sel[d][1]][1]*ds[sel[d][2]][2] *
		   field[d*size*offmtp + ip1*szy*szz + jp1*szz + kp1] * inpx*inpy*inpz;
	    // Row major and particles are along rows (because it's easier to gather with alltoallv)
	    // That is, [Npart, 3] flattened row-major (so [x0, y0, z0, x1, y1, z1, ...]
            (*outmass)[3*p + d] = mass;
	}
    }
}

template <typename T, typename intT>
void preadout3D(Layout& layout, // Decomposition layout PPAINT CALL NEEDED
                T*& expos, // Exchanged positions PPAINT CALL NEEDED
                int64_t& Nparts, // Number of particles
                T*& localfield, // Flattened local field (or fields) to readout from
                intT*& Nmesh, // (3, ) Mesh grid specs
                T*& BoxSize, // Needed for vjp
                intT*& edgesx, // Flattened (comm_size, 2) limiting grid indices along x for each rank
                intT*& edgesy, // Same, for y
                intT*& edgesz, // Same, for z
                T** outptr, // Output pointer
                MPI_Comm& comm, // MPI communicator
                bool& vjp){ // Whether the call is used to compute
    // Parallel readout in 3D
    // NOTE: THIS EXPLOITS AVAILABLE DECOMPOSITION LAYOUT, SO CALL PPAINT FIRST
    // Interpolates local field using CIC. Particle positions are assumed to be already
    // the ones which the local field contributes to. Sums ghosts at the end
    // Stores resulting local part of density field in outptr

    int comm_rank;
    MPI_Comm_rank(comm, &comm_rank);

    // Readout at exchanged positions
    intT* ex = &(edgesx[2*comm_rank]); // Edges for this rank
    intT* ey = &(edgesy[2*comm_rank]);
    intT* ez = &(edgesz[2*comm_rank]);
    // Readout locally
    T* outmasses = new T[3 * layout.totrecv];
    lreadout3D<T, intT>(expos, layout.totrecv, localfield, ex, ey, ez, Nmesh, BoxSize, &outmasses, vjp);

    // Now gather the data back into its original rank
    for (int i=0; i<3*Nparts; i++) (*outptr)[i] = 0;
    gather(outmasses, 3, layout, outptr, comm);
    delete[] outmasses;
}

template <typename T, typename intT>
void localpositions(T*& pos, // (Nparts, 3) Positions to shuffle (flattened, in mesh space)
		    int64_t& Nparts, // Number of particles
		    intT*& Nmesh, // (3, ) Mesh grid specs
		    intT*& edgesx, // Flattened (comm_size, 2) limit grid indices along x for each rank
                    intT*& edgesy, // Same, for y
                    intT*& edgesz, // Same, for z
		    int comm_size, // MPI communicator size
		    int comm_rank, // MPI rank of this process
		    T** outpos, // (Npartsout[comm_rank], 3) Output pointer (flattened)
		    int64_t*& Npartsout){ // (comm_size, ) Number of output particles for each rank
    // Selects positions that are within limits of local field, to speed up the painting.
    // Does so until Npartsout, usually Nparts / comm_size, particles have been selected,
    // otherwise moves them to other ranks that haven't reached Npartsout.
    // Every rank calls this, even though it's not parallel, because it's convenient and
    // it doesn't really make sense that only one calls it while the others wait anyway.
    
    int* rankcounts = new int[comm_size]; // How many particles are assigned to this rank (max Npartsout)
    bool* assigned = new bool[Nparts]; // Whether the particle was successfully assigned or not
    for (int rank=0; rank<comm_size; rank++) rankcounts[rank] = 0;
    int i, j, k;

    // First loop to assign as many particles as possible to this rank
    for (int p=0; p<Nparts; p++){
	i = int(pos[3*p+0]) % Nmesh[0];
        j = int(pos[3*p+1]) % Nmesh[1];
        k = int(pos[3*p+2]) % Nmesh[2];
	for (int rank=0; rank<comm_size; rank++){
	    if ((unsigned)(i - edgesx[2*rank]) < (edgesx[2*rank+1] - edgesx[2*rank]) &&
                (unsigned)(j - edgesy[2*rank]) < (edgesy[2*rank+1] - edgesy[2*rank]) &&
                (unsigned)(k - edgesz[2*rank]) < (edgesz[2*rank+1] - edgesz[2*rank])){
		if (rankcounts[rank] < Npartsout[rank]){ // If can assign do
		    if (rank == comm_rank){ // Only store particles if it's this rank
		    	(*outpos)[3*rankcounts[rank]+0] = pos[3*p+0];
		    	(*outpos)[3*rankcounts[rank]+1] = pos[3*p+1];
		    	(*outpos)[3*rankcounts[rank]+2] = pos[3*p+2];
		    }
		    assigned[p] = true; // p was stored either in this rank or in another
                    rankcounts[rank]++; // Increment this rank counter
		} else { // Else set it to not assigned
		    assigned[p] = false;
		}
		break; // Rank of correct local part of the field is found, just stop looping over ranks
            }
	}
    }

    // Loop to assign remaining particles in random rank
    for (int p=0; p<Nparts; p++){
	if (!assigned[p]) {
	    for (int rank=0; rank<comm_size; rank++){ // Look up which rank can host more particles
		if (rankcounts[rank] < Npartsout[rank]){ // Assign it to this rank
		    if (rank == comm_rank){ // Only store particles if it's this rank
			(*outpos)[3*rankcounts[rank]+0] = pos[3*p+0];
                    	(*outpos)[3*rankcounts[rank]+1] = pos[3*p+1];
                    	(*outpos)[3*rankcounts[rank]+2] = pos[3*p+2];
		    }
		    rankcounts[rank]++;
		    break; // Particle was assigned in this or in another rank, just break rank loop
		}
	    }
	}
    }

    delete[] rankcounts;
    delete[] assigned;
}

} // namespace parops_jax

#endif

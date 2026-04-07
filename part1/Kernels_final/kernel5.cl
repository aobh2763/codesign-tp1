#define TRANSPOSEX 32
#define TRANSPOSEY 32

// Simple transpose kernel for a P * Q matrix
__kernel void transpose(const int P, const int Q,
                        const __global float* input,
                        __global float* output) {
    
    // Thread identifiers
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int ID0 = get_group_id(0)*TRANSPOSEX + tx; // 0..P
    const int ID1 = get_group_id(1)*TRANSPOSEY + ty; // 0..Q
 
    // Set-up the local memory for shuffling
    __local float buffer[TRANSPOSEX][TRANSPOSEY];
 
    // Swap the x and y coordinates to perform the rotation (coalesced)
    if (ID0 < P && ID1 < Q) {
        buffer[ty][tx] = input[ID1*P + ID0];
    }
 
    // Synchronise all threads
    barrier(CLK_LOCAL_MEM_FENCE);
 
    // We don't have to swap the x and y thread indices here,
    // because that's already done in the local memory
    const int newID0 = get_group_id(1)*TRANSPOSEY + tx;
    const int newID1 = get_group_id(0)*TRANSPOSEX + ty;
 
    // Store the transposed result (coalesced)
    if (newID0 < Q && newID1 < P) {
        output[newID1*Q + newID0] = buffer[tx][ty];
    }
}

#ifndef TSM
#define TSM 64 // Tile size for M dimension
#endif

#ifndef TSN
#define TSN 64 // Tile size for N dimension
#endif

#ifndef TSK
#define TSK 32 // Tile size for K dimension
#endif

#ifndef WPTN
#define WPTN 8 // Work per thread in N dimension
#endif

#ifndef WPTM
#define WPTM 8 // Work per thread in M dimension
#endif

#define RTSN (TSN/WPTN)        // The reduced tile-size in dimension N
#define RTSM (TSM/WPTM)        // The reduced tile-size in dimension M
#define LPT ((TSK*TSM)/(RTSM*RTSN)) // The loads-per-thread for a tile

// Pre-transpose the input matrix B and use rectangular tiles
__kernel void mmul(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C) {
    
    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TSM)
    const int col = get_local_id(1); // Local col ID (max: TSN/WPTN)
    const int globalRow = TSM*get_group_id(0) + row; // 0..M
    const int globalCol = TSN*get_group_id(1) + col; // 0..N
 
    // Local memory to fit a tile of A and B
    __local float Asub[TSK][TSM];
    __local float Bsub[TSN][TSK+2];
 
    // Initialise the accumulation registers
    float acc[WPTN];
    for (int w=0; w<WPTN; w++) {
        acc[w] = 0.0f;
    }
    
    // Loop over all tiles
    int numTiles = K/TSK;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        for (int l=0; l<LPT; l++) {
            int tiledIndex = TSK*t + col + l*RTSN;
            int indexA = tiledIndex*M + TSM*get_group_id(0) + row;
            int indexB = tiledIndex*N + TSN*get_group_id(1) + row;
            Asub[col + l*RTSN][row] = A[indexA];
            Bsub[row][col + l*RTSN] = B[indexB];
       }
        
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k=0; k<TSK; k++) {
            for (int w=0; w<WPTN; w++) {
                acc[w] += Asub[k][row] * Bsub[col + w*RTSN][k];
            }
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final results in C
    for (int w=0; w<WPTN; w++) {
        C[(globalCol + w*RTSN)*M + globalRow] = acc[w];
    }
}
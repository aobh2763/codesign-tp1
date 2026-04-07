#ifndef TS
#define TS 32
#endif

#ifndef WTP
#define WPT 8
#endif

#ifndef TSDK
#define TSDK 16
#endif

#define RTS (TS/WPT)                 // The reduced tile-size in one dimension
#define LPT ((TSDK*WPT)/(TS))        // The amount of loads-per-thread (assume TSN==TSM)

#define TRANSPOSEX 16
#define TRANSPOSEY 16

// Simple transpose kernel for a P * Q matrix
__kernel void transpose(const int P, const int Q,
                        const __global float* input,
                        __global float* output) {
    
    const int tx  = get_local_id(0);
    const int ty  = get_local_id(1);
    const int ID0 = get_group_id(0)*TRANSPOSEX + tx; // 0..P
    const int ID1 = get_group_id(1)*TRANSPOSEY + ty; // 0..Q

<<<<<<< HEAD
    __local float buffer[TRANSPOSEX][TRANSPOSEY];

=======
    // Set-up the local memory for shuffling
    __local float buffer[TRANSPOSEX][TRANSPOSEY];

    // Swap the x and y coordinates to perform the rotation (coalesced)
>>>>>>> a4b9a6ffe88eb61396f2132f16ad36d1100ecdfe
    if (ID0 < P && ID1 < Q) {
        buffer[ty][tx] = input[ID1*P + ID0];
    }

<<<<<<< HEAD
    barrier(CLK_LOCAL_MEM_FENCE);

    const int newID0 = get_group_id(1)*TRANSPOSEY + tx;
    const int newID1 = get_group_id(0)*TRANSPOSEX + ty;

=======
    // Synchronise all threads
    barrier(CLK_LOCAL_MEM_FENCE);

    // We don't have to swap the x and y thread indices here,
    // because that's already done in the local memory
    const int newID0 = get_group_id(1)*TRANSPOSEY + tx;
    const int newID1 = get_group_id(0)*TRANSPOSEX + ty;

    // Store the transposed result (coalesced)
>>>>>>> a4b9a6ffe88eb61396f2132f16ad36d1100ecdfe
    if (newID0 < Q && newID1 < P) {
        output[newID1*Q + newID0] = buffer[tx][ty];
    }
}

<<<<<<< HEAD
// ---------------------------------------------------------------
// Tunable defines — overridden by -D flags from the host
// ---------------------------------------------------------------
#ifndef TSM
#define TSM  64
#endif
#ifndef TSN
#define TSN  64
#endif
#ifndef TSK
#define TSK  32
#endif
#ifndef WPTM
#define WPTM 1          // work-per-thread in M  (this kernel keeps WPTM=1)
#endif
#ifndef WPTN
#define WPTN 8          // work-per-thread in N
#endif

#define RTSM (TSM/WPTM)
#define RTSN (TSN/WPTN)
#define LPT  ((TSK*TSM)/(RTSM*RTSN))   // loads-per-thread

// ---------------------------------------------------------------
// myGEMM5 — pre-transposed B, rectangular tiles, WPTN work per thread
// WPTM is fixed to 1 (one output row per thread)
// ---------------------------------------------------------------
__kernel void mmul(const int M, const int N, const int K,
                   const __global float* A,
                   const __global float* B,   // B already transposed (N x K)
                   __global float* C) {

    // Thread identifiers
    const int row       = get_local_id(0);              // 0..RTSM-1
    const int col       = get_local_id(1);              // 0..RTSN-1
    const int globalRow = TSM*get_group_id(0) + row;    // 0..M-1
    const int globalCol = TSN*get_group_id(1) + col;    // 0..N-1  (first of WPTN cols)

    // Local tiles
    // Asub[k][m] — column-major so the compute loop reads Asub[k][row] with no bank conflicts
    __local float Asub[TSK][TSM];
    // Bsub[n][k] — B is pre-transposed, so a row of Bsub is a row of B^T
    __local float Bsub[TSN][TSK];

    // Private accumulation registers — one per output column (WPTN outputs)
    float acc[WPTN];
    for (int w = 0; w < WPTN; w++) {
        acc[w] = 0.0f;
    }

    // -------  tile loop  -------
    const int numTiles = K / TSK;
    for (int t = 0; t < numTiles; t++) {

        // Load LPT elements of A and B into local memory.
        // The RTSM*RTSN threads collectively load the full TSM*TSK tile of A
        // and the full TSN*TSK tile of B.
        for (int l = 0; l < LPT; l++) {

            // tiledIndex: which K-index inside this tile are we loading?
            // col + l*RTSN walks 0..(TSK-1) when LPT = TSK*TSM/(RTSM*RTSN)
            // and TSM == RTSM (i.e. WPTM=1).  More precisely each (row,col,l)
            // triple maps to a unique (m_local, k_local) pair:
            int linearIdx = row + col*RTSM + l*RTSM*RTSN; // 0..TSM*TSK-1
            int m_local   = linearIdx % TSM;               // local M index
            int k_local   = linearIdx / TSM;               // local K index

            // --- Load A tile (A is M-major, i.e. column-major here) ---
            // A[k][m]  with k = TSK*t + k_local, m = TSM*group + m_local
            int kGlobal = TSK*t + k_local;
            int mGlobal = TSM*get_group_id(0) + m_local;
            Asub[k_local][m_local] = A[kGlobal*M + mGlobal];

            // --- Load B tile (B is already transposed: stored as B^T[n][k]) ---
            // We load a TSN*TSK block. Re-use linearIdx over TSN*TSK elements.
            int n_local  = linearIdx % TSN;
            int k_local2 = linearIdx / TSN;
            int nGlobal  = TSN*get_group_id(1) + n_local;
            int kGlobal2 = TSK*t + k_local2;
            if (linearIdx < TSN*TSK) {   // guard: only needed when TSN != TSM
                Bsub[n_local][k_local2] = B[kGlobal2*N + nGlobal];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute: each thread accumulates WPTN dot-products
        for (int k = 0; k < TSK; k++) {
            float a_val = Asub[k][row];               // one A value for this row
            for (int w = 0; w < WPTN; w++) {
                acc[w] += a_val * Bsub[col + w*RTSN][k];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write WPTN results to C (C is column-major: C[n*M + m])
    for (int w = 0; w < WPTN; w++) {
        if (globalRow < M && (globalCol + w*RTSN) < N) {
            C[(globalCol + w*RTSN)*M + globalRow] = acc[w];
        }
=======
// Pre-transpose the input matrix B and use rectangular tiles
__kernel void mmul(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C) {

    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS/WPT == RTS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)

    // Local memory to fit a tile of A and B
    __local float Asub[TSDK][TS];
    __local float Bsub[TS][TSDK+2];

    // Initialise the accumulation registers
    float acc[WPT];
    for (int w=0; w<WPT; w++) {
        acc[w] = 0.0f;
    }
    
    // Loop over all tiles
    const int numTiles = K/TSDK;
    for (int t=0; t<numTiles; t++) {

        // Load one tile of A and B into local memory
        for (int l=0; l<LPT; l++) {
            const int tiledIndex = TSDK*t + col + l*RTS;
            int indexA = (tiledIndex)*M + TS*get_group_id(0) + row;
            int indexB = (tiledIndex)*N + TS*get_group_id(1) + row;
            Asub[col + l*RTS][row] = A[indexA];
            Bsub[row][col + l*RTS] = B[indexB];
        }

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the computation for a single tile
        for (int k=0; k<TSDK; k++) {
            for (int w=0; w<WPT; w++) {
                acc[w] += Asub[k][row] * Bsub[col + w*RTS][k];
            }
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final results in C
    for (int w=0; w<WPT; w++) {
        C[(globalCol + w*RTS)*M + globalRow] = acc[w];
>>>>>>> a4b9a6ffe88eb61396f2132f16ad36d1100ecdfe
    }
}
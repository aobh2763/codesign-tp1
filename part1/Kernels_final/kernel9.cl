#ifndef TSM
#define TSM 128                      // The tile-size in dimension M
#endif

#ifndef TSN
#define TSN 128                      // The tile-size in dimension N
#endif

#ifndef TSK
#define TSK 16                       // The tile-size in dimension K
#endif

#ifndef WPTM
#define WPTM 8                       // The amount of work-per-thread in dimension M
#endif

#ifndef WPTN
#define WPTN 8                       // The amount of work-per-thread in dimension N
#endif

#define WIDTH 4

#define RTSM (TSM/WPTM)              // The reduced tile-size in dimension M (== number of threads)
#define RTSN (TSN/WPTN)              // The reduced tile-size in dimension N (== number of threads)
#define LPTA ((TSK*WPTM*WPTN)/(TSN)) // The amount of loads-per-thread for A
#define LPTB ((TSK*WPTM*WPTN)/(TSM)) // The amount of loads-per-thread for B

// Data-widths
#if WIDTH == 1
    typedef float floatX;
#elif WIDTH == 2
    typedef float2 floatX;
#elif WIDTH == 4
    typedef float4 floatX;
#elif WIDTH == 8
    typedef float8 floatX;
#endif

// With pre-fetching
__kernel void mmul(const int M, const int N, const int K,
                      const __global floatX* A,
                      const __global floatX* B,
                      __global float* C) {

    // Thread identifiers
    const int tidm = get_local_id(0); // Local row ID (max: TSM/WPTM == RTSM)
    const int tidn = get_local_id(1); // Local col ID (max: TSN/WPTN == RTSN)
    const int offsetM = TSM*get_group_id(0); // Work-group offset
    const int offsetN = TSN*get_group_id(1); // Work-group offset

    // Local memory to fit two tiles of A and B
    __local float Asub[2][TSK*TSM];
    __local float Bsub[2][TSK*TSN];

    // Allocate register space
    float Areg;
    float Breg[WPTN];
    float acc[WPTM][WPTN];

    // Initialise the accumulation registers
    #pragma unroll
    for (int wm=0; wm<WPTM; wm++) {
        #pragma unroll
        for (int wn=0; wn<WPTN; wn++) {
            acc[wm][wn] = 0.0f;
        }
    }

    // Load the first tile of A and B into local memory
    #pragma unroll
    for (int la=0; la<LPTA/WIDTH; la++) {
        int tid = tidn*RTSM + tidm;
        int id = la*RTSN*RTSM + tid;
        int row = id % (TSM/WIDTH);
        int col = id / (TSM/WIDTH);

        // Load the values (wide vector load)
        int tiledIndex = TSK*0 + col;
        int indexA = tiledIndex*(M/WIDTH) + offsetM/WIDTH + row;
        int indexB = tiledIndex*(N/WIDTH) + offsetN/WIDTH + row;
        #ifdef USE_LDG
            floatX vecA = __ldg(&A[indexA]);
            floatX vecB = __ldg(&B[indexB]);
        #else
            floatX vecA = A[indexA];
            floatX vecB = B[indexB];
        #endif

        // Store the loaded vectors into local memory
        #if WIDTH == 1
            Asub[0][col*TSM + row] = vecA;
        #elif WIDTH == 2
            Asub[0][col*TSM + WIDTH*row + 0] = vecA.x;
            Asub[0][col*TSM + WIDTH*row + 1] = vecA.y;
        #elif WIDTH == 4
            Asub[0][col*TSM + WIDTH*row + 0] = vecA.x;
            Asub[0][col*TSM + WIDTH*row + 1] = vecA.y;
            Asub[0][col*TSM + WIDTH*row + 2] = vecA.z;
            Asub[0][col*TSM + WIDTH*row + 3] = vecA.w;
        #endif
        #if WIDTH == 1
            Bsub[0][col*TSN + row] = vecB;
        #elif WIDTH == 2
            Bsub[0][col*TSN + WIDTH*row + 0] = vecB.x;
            Bsub[0][col*TSN + WIDTH*row + 1] = vecB.y;
        #elif WIDTH == 4
            Bsub[0][col*TSN + WIDTH*row + 0] = vecB.x;
            Bsub[0][col*TSN + WIDTH*row + 1] = vecB.y;
            Bsub[0][col*TSN + WIDTH*row + 2] = vecB.z;
            Bsub[0][col*TSN + WIDTH*row + 3] = vecB.w;
        #endif
    }

    // Synchronise
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Loop over all tiles
    const int numTiles = K/TSK;
    int t=0;
    do {

        // Load the next tile of A and B into local memory
        int tt = t + 1;
        if (tt < numTiles) {
            #pragma unroll
            for (int la=0; la<LPTA/WIDTH; la++) {
                int tid = tidn*RTSM + tidm;
                int id = la*RTSN*RTSM + tid;
                int row = id % (TSM/WIDTH);
                int col = id / (TSM/WIDTH);

                // Load the values (wide vector load)
                int tiledIndex = TSK*tt + col;
                int indexA = tiledIndex*(M/WIDTH) + offsetM/WIDTH + row;
                int indexB = tiledIndex*(N/WIDTH) + offsetN/WIDTH + row;
                #ifdef USE_LDG
                    floatX vecA = __ldg(&A[indexA]);
                    floatX vecB = __ldg(&B[indexB]);
                #else
                    floatX vecA = A[indexA];
                    floatX vecB = B[indexB];
                #endif

                // Store the loaded vectors into local memory
                #if WIDTH == 1
                    Asub[tt%2][col*TSM + row] = vecA;
                #elif WIDTH == 2
                    Asub[tt%2][col*TSM + WIDTH*row + 0] = vecA.x;
                    Asub[tt%2][col*TSM + WIDTH*row + 1] = vecA.y;
                #elif WIDTH == 4
                    Asub[tt%2][col*TSM + WIDTH*row + 0] = vecA.x;
                    Asub[tt%2][col*TSM + WIDTH*row + 1] = vecA.y;
                    Asub[tt%2][col*TSM + WIDTH*row + 2] = vecA.z;
                    Asub[tt%2][col*TSM + WIDTH*row + 3] = vecA.w;
                #endif
                #if WIDTH == 1
                    Bsub[tt%2][col*TSN + row] = vecB;
                #elif WIDTH == 2
                    Bsub[tt%2][col*TSN + WIDTH*row + 0] = vecB.x;
                    Bsub[tt%2][col*TSN + WIDTH*row + 1] = vecB.y;
                #elif WIDTH == 4
                    Bsub[tt%2][col*TSN + WIDTH*row + 0] = vecB.x;
                    Bsub[tt%2][col*TSN + WIDTH*row + 1] = vecB.y;
                    Bsub[tt%2][col*TSN + WIDTH*row + 2] = vecB.z;
                    Bsub[tt%2][col*TSN + WIDTH*row + 3] = vecB.w;
                #endif
            }
        }

        // Loop over the values of a single tile
        #pragma unroll
        for (int k=0; k<TSK; k++) {

            // Cache the values of Bsub in registers
            #pragma unroll
            for (int wn=0; wn<WPTN; wn++) {
                int col = tidn + wn*RTSN;
                Breg[wn] = Bsub[t%2][k*TSN + col];
            }

            // Perform the computation
            #pragma unroll
            for (int wm=0; wm<WPTM; wm++) {
                int row = tidm + wm*RTSM;
                Areg = Asub[t%2][k*TSM + row];
                #pragma unroll
                for (int wn=0; wn<WPTN; wn++) {
                    acc[wm][wn] += Areg * Breg[wn];
                }
            }
        }

        // Synchronise
        barrier(CLK_LOCAL_MEM_FENCE);

        // Next tile
        t++;
    } while (t<numTiles);

    // Store the final results in C
    #pragma unroll
    for (int wm=0; wm<WPTM; wm++) {
        int globalRow = offsetM + tidm + wm*RTSM;
        #pragma unroll
        for (int wn=0; wn<WPTN; wn++) {
            int globalCol = offsetN + tidn + wn*RTSN;
            C[globalCol*M + globalRow] = acc[wm][wn];
        }
    }
}
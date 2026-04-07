// =============================================================================
//  kernel_ultimate.cl
//  Ultimate OpenCL Matrix Multiplication — C = A * B  (column-major storage)
//
//  Techniques combined (skipping TBD kernels 5 and 8):
//    K2  — Local-memory tiling for data reuse
//    K3  — Work-per-thread to amortise per-thread overhead
//    K4  — Wide (vector) global memory loads
//    K6  — 2D register blocking + B cached in private registers
//    K7  — Wide loads feeding the 2D register-blocking structure
//    K9  — Double-buffered pre-fetching (ping-pong local memory)
//   K10  — Arbitrary/incomplete tile support + best verified tile dimensions
//
//  Best parameter set (from K10 benchmark, WIDTH upgraded to 4 as in K7):
//    TSM=128  TSN=128  TSK=16  WPTM=16  WPTN=8  WIDTH=4
//
//  Required global / local NDRange:
//    local  = { RTSM, RTSN } = { TSM/WPTM, TSN/WPTN } = { 8, 16 }
//    global = { ceil(M/TSM)*RTSM,  ceil(N/TSN)*RTSN }
// =============================================================================

// ----------------------------------------------------------------------------
// Tunable compile-time parameters  (override with -D on the host side)
// ----------------------------------------------------------------------------
#ifndef TSM
#define TSM  128    // tile rows of C covered by one work-group
#endif
#ifndef TSN
#define TSN  128    // tile cols of C covered by one work-group
#endif
#ifndef TSK
#define TSK  16     // depth of each K-tile (inner-reduction block size)
#endif
#ifndef WPTM
#define WPTM 16     // output rows computed per thread  (must divide TSM)
#endif
#ifndef WPTN
#define WPTN 8      // output cols computed per thread  (must divide TSN)
#endif
#ifndef WIDTH
#define WIDTH 4     // float vector width for global->local loads (1,2,4,8)
#endif

// ----------------------------------------------------------------------------
// Derived constants
// ----------------------------------------------------------------------------
#define RTSM  (TSM / WPTM)                       // threads in M per work-group
#define RTSN  (TSN / WPTN)                       // threads in N per work-group
#define LPTA  ((TSK * WPTM * WPTN) / TSN)        // scalar loads-per-thread for A
#define LPTB  ((TSK * WPTM * WPTN) / TSM)        // scalar loads-per-thread for B

// ----------------------------------------------------------------------------
// Vector-type alias  (K4 / K7 technique)
// ----------------------------------------------------------------------------
#if WIDTH == 1
    typedef float  floatX;
#elif WIDTH == 2
    typedef float2 floatX;
#elif WIDTH == 4
    typedef float4 floatX;
#elif WIDTH == 8
    typedef float8 floatX;
#endif

// =============================================================================
__kernel void mmul(const int M, const int N, const int K,
                   const __global floatX* restrict A,
                   const __global floatX* restrict B,
                   __global float* C)
{
    // -------------------------------------------------------------------------
    // Thread / work-group IDs
    // -------------------------------------------------------------------------
    const int tidm = get_local_id(0);         // [0, RTSM)
    const int tidn = get_local_id(1);         // [0, RTSN)
    const int gidm = get_group_id(0);
    const int gidn = get_group_id(1);
    const int tid  = tidn * RTSM + tidm;      // flat ID within work-group

    // -------------------------------------------------------------------------
    // Double-buffered local memory  (K9 / K10 ping-pong pre-fetch)
    //
    // Two complete tiles of A and B are kept in local memory at all times.
    // While the compute stage consumes buffer (t%2), the load stage fills
    // buffer ((t+1)%2), hiding global-memory latency behind ALU work.
    //
    // Flat layout: Asub[buf][k * TSM + m],  Bsub[buf][k * TSN + n]
    // -------------------------------------------------------------------------
    __local float Asub[2][TSK * TSM];
    __local float Bsub[2][TSK * TSN];

    // -------------------------------------------------------------------------
    // Private register file  (K6 2D register blocking)
    //
    // Each thread owns a WPTM x WPTN sub-matrix of C entirely in registers.
    // This eliminates all redundant local-memory reads for the output and
    // drastically increases the arithmetic intensity per memory access.
    // -------------------------------------------------------------------------
    float Areg;            // one A scalar reused across WPTN multiplications
    float Breg[WPTN];      // WPTN B scalars cached in registers  (K6 / K7)
    float acc[WPTM][WPTN]; // accumulation registers

    #pragma unroll
    for (int wm = 0; wm < WPTM; wm++) {
        #pragma unroll
        for (int wn = 0; wn < WPTN; wn++) {
            acc[wm][wn] = 0.0f;
        }
    }

    // =========================================================================
    // MACRO: scatter one float4 (or float2/float) vector into a flat local tile
    // =========================================================================
    // Used twice (tile 0 pre-load, then inside the loop for tile t+1).

    // -------------------------------------------------------------------------
    // Pre-load tile 0 into buffer 0 BEFORE the main loop  (K9/K10 pattern)
    //
    // Wide vector loads (K4/K7): each thread issues LPTA/WIDTH float4 loads
    // for A and the same for B.  Reading WIDTH=4 consecutive floats in a
    // single instruction maximises memory bus utilisation.
    //
    // Boundary guards (K10): if M, N or K are not multiples of the tile sizes
    // the out-of-range elements are forced to 0.0 so the arithmetic remains
    // correct without branching inside the inner compute loop.
    // -------------------------------------------------------------------------
    #pragma unroll
    for (int la = 0; la < LPTA / WIDTH; la++) {
        int id      = la * RTSN * RTSM + tid;
        int row     = id % (TSM / WIDTH);   // vectorised row index within tile
        int col     = id / (TSM / WIDTH);   // K-column index within tile

        int tiledA  = TSK * 0 + col;        // absolute K index in A
        int gRowA   = gidm * (TSM / WIDTH) + row;

        floatX vecA = (floatX)(0.0f);
        if (tiledA < K && gRowA * WIDTH < M)
            vecA = A[tiledA * (M / WIDTH) + gRowA];

        #if WIDTH == 1
            Asub[0][col * TSM + row]            = vecA;
        #elif WIDTH == 2
            Asub[0][col * TSM + WIDTH * row + 0] = vecA.x;
            Asub[0][col * TSM + WIDTH * row + 1] = vecA.y;
        #elif WIDTH == 4
            Asub[0][col * TSM + WIDTH * row + 0] = vecA.x;
            Asub[0][col * TSM + WIDTH * row + 1] = vecA.y;
            Asub[0][col * TSM + WIDTH * row + 2] = vecA.z;
            Asub[0][col * TSM + WIDTH * row + 3] = vecA.w;
        #endif
    }

    #pragma unroll
    for (int lb = 0; lb < LPTB / WIDTH; lb++) {
        int id      = lb * RTSN * RTSM + tid;
        int row     = id % (TSN / WIDTH);
        int col     = id / (TSN / WIDTH);

        int tiledB  = TSK * 0 + col;
        int gRowB   = gidn * (TSN / WIDTH) + row;

        floatX vecB = (floatX)(0.0f);
        if (tiledB < K && gRowB * WIDTH < N)
            vecB = B[tiledB * (N / WIDTH) + gRowB];

        #if WIDTH == 1
            Bsub[0][col * TSN + row]            = vecB;
        #elif WIDTH == 2
            Bsub[0][col * TSN + WIDTH * row + 0] = vecB.x;
            Bsub[0][col * TSN + WIDTH * row + 1] = vecB.y;
        #elif WIDTH == 4
            Bsub[0][col * TSN + WIDTH * row + 0] = vecB.x;
            Bsub[0][col * TSN + WIDTH * row + 1] = vecB.y;
            Bsub[0][col * TSN + WIDTH * row + 2] = vecB.z;
            Bsub[0][col * TSN + WIDTH * row + 3] = vecB.w;
        #endif
    }

    // Make tile 0 visible to the whole work-group before compute begins
    barrier(CLK_LOCAL_MEM_FENCE);

    // =========================================================================
    // Main tile loop  (K10 ceiling-division handles arbitrary K)
    // =========================================================================
    const int numTiles = (K + TSK - 1) / TSK;
    int t = 0;
    do {
        // ---------------------------------------------------------------------
        // STAGE 1 — Pre-fetch tile (t+1) into the alternate buffer
        //           while STAGE 2 computes on tile t.
        //
        // This is the core double-buffering idea from K9/K10.  The two stages
        // are interleaved within the same loop iteration: loads go to buffer
        // (tt%2) and the compute below reads from buffer (t%2), where tt=t+1.
        // Because local memory is banked, loads to one buffer do not stall
        // reads from the other.
        // ---------------------------------------------------------------------
        int tt = t + 1;
        if (tt < numTiles) {

            #pragma unroll
            for (int la = 0; la < LPTA / WIDTH; la++) {
                int id      = la * RTSN * RTSM + tid;
                int row     = id % (TSM / WIDTH);
                int col     = id / (TSM / WIDTH);

                int tiledA  = TSK * tt + col;
                int gRowA   = gidm * (TSM / WIDTH) + row;

                floatX vecA = (floatX)(0.0f);
                if (tiledA < K && gRowA * WIDTH < M)
                    vecA = A[tiledA * (M / WIDTH) + gRowA];

                #if WIDTH == 1
                    Asub[tt%2][col * TSM + row]            = vecA;
                #elif WIDTH == 2
                    Asub[tt%2][col * TSM + WIDTH * row + 0] = vecA.x;
                    Asub[tt%2][col * TSM + WIDTH * row + 1] = vecA.y;
                #elif WIDTH == 4
                    Asub[tt%2][col * TSM + WIDTH * row + 0] = vecA.x;
                    Asub[tt%2][col * TSM + WIDTH * row + 1] = vecA.y;
                    Asub[tt%2][col * TSM + WIDTH * row + 2] = vecA.z;
                    Asub[tt%2][col * TSM + WIDTH * row + 3] = vecA.w;
                #endif
            }

            #pragma unroll
            for (int lb = 0; lb < LPTB / WIDTH; lb++) {
                int id      = lb * RTSN * RTSM + tid;
                int row     = id % (TSN / WIDTH);
                int col     = id / (TSN / WIDTH);

                int tiledB  = TSK * tt + col;
                int gRowB   = gidn * (TSN / WIDTH) + row;

                floatX vecB = (floatX)(0.0f);
                if (tiledB < K && gRowB * WIDTH < N)
                    vecB = B[tiledB * (N / WIDTH) + gRowB];

                #if WIDTH == 1
                    Bsub[tt%2][col * TSN + row]            = vecB;
                #elif WIDTH == 2
                    Bsub[tt%2][col * TSN + WIDTH * row + 0] = vecB.x;
                    Bsub[tt%2][col * TSN + WIDTH * row + 1] = vecB.y;
                #elif WIDTH == 4
                    Bsub[tt%2][col * TSN + WIDTH * row + 0] = vecB.x;
                    Bsub[tt%2][col * TSN + WIDTH * row + 1] = vecB.y;
                    Bsub[tt%2][col * TSN + WIDTH * row + 2] = vecB.z;
                    Bsub[tt%2][col * TSN + WIDTH * row + 3] = vecB.w;
                #endif
            }
        }

        // ---------------------------------------------------------------------
        // STAGE 2 — Compute on the current buffer (t % 2)
        //
        // 2D register blocking  (K6 / K7):
        //   For every k-step inside this TSK-depth tile:
        //     a) Load WPTN B-values into Breg[] once — then reuse across WPTM
        //        A-values.  This means WPTN local-memory reads for B serve
        //        WPTM*WPTN MAD operations, giving a WPTM-fold reuse ratio.
        //     b) Load one A scalar per wm-step and dot it against all Breg[].
        //        This gives another WPTN-fold reuse ratio for the A value.
        //   Combined: each local-memory read amortises WIDTH times more work
        //   than the scalar K2 kernel, and WPTM*WPTN times more than K1.
        // ---------------------------------------------------------------------
        #pragma unroll
        for (int k = 0; k < TSK; k++) {

            // Cache WPTN B-values in private registers — read local mem once,
            // reuse WPTM times in the wm-loop below.
            #pragma unroll
            for (int wn = 0; wn < WPTN; wn++) {
                int colB = tidn + wn * RTSN;
                Breg[wn] = Bsub[t % 2][k * TSN + colB];
            }

            // For each of the WPTM rows this thread is responsible for,
            // load one A scalar then accumulate against all cached B values.
            #pragma unroll
            for (int wm = 0; wm < WPTM; wm++) {
                int rowA = tidm + wm * RTSM;
                Areg = Asub[t % 2][k * TSM + rowA];
                #pragma unroll
                for (int wn = 0; wn < WPTN; wn++) {
                    acc[wm][wn] += Areg * Breg[wn];
                }
            }
        }

        // Synchronise: ensure the pre-fetched buffer is fully written before
        // the next iteration's compute stage reads from it.
        barrier(CLK_LOCAL_MEM_FENCE);

        t++;
    } while (t < numTiles);

    // =========================================================================
    // Store WPTM x WPTN results to global C  (K10 boundary guard)
    //
    // The boundary check makes the kernel correct for any M, N, K without
    // the host having to pad matrices to multiples of the tile dimensions.
    // =========================================================================
    #pragma unroll
    for (int wm = 0; wm < WPTM; wm++) {
        int globalRow = gidm * TSM + tidm + wm * RTSM;
        #pragma unroll
        for (int wn = 0; wn < WPTN; wn++) {
            int globalCol = gidn * TSN + tidn + wn * RTSN;
            if (globalRow < M && globalCol < N) {
                C[globalCol * M + globalRow] = acc[wm][wn];
            }
        }
    }
}

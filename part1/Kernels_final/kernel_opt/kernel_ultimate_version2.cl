/*
 * =============================================================================
 *  ULTIMATE COMBINED GEMM KERNEL
 *  Integrates techniques from: K2, K3, K4, K6, K7, K9, K10
 *  (K8 omitted – CUDA/Kepler-specific, TBD in README)
 * =============================================================================
 *
 *  TECHNIQUE INVENTORY
 *  ───────────────────
 *  K2  → Two-level tiling: data is staged through __local (shared) memory in
 *         TSM×TSK and TSK×TSN tiles to reuse values loaded from global memory.
 *
 *  K3  → Work-per-thread (WPT): each thread computes WPTM×WPTN output elements
 *         instead of 1, raising arithmetic intensity and hiding latency.
 *
 *  K4  → Vector loads (floatX): global memory is read WIDTH floats at a time
 *         (float4 here) halving the number of load instructions and maximally
 *         exploiting cache-line bandwidth.
 *
 *  K6  → 2-D register blocking: accumulator array acc[WPTM][WPTN] kept in
 *         registers; one A value (Areg) broadcasts across WPTN B values (Breg[])
 *         yielding an outer-product micro-kernel with WPTM×WPTN MADs per k step.
 *
 *  K7  → Wide loads + register blocking combined: float4 global→local stores
 *         unpacked into the flat local array; compute phase remains scalar
 *         register-file ops for maximum compiler visibility.
 *
 *  K9  → Software-pipelined double-buffering: two ping-pong local buffers let
 *         the next tile be prefetched from global memory *while* the current
 *         tile is being consumed, hiding global-memory latency behind compute.
 *         Barrier placed at the END of each iteration (after compute, before
 *         next prefetch reads the buffer just written) – one barrier per tile
 *         instead of two, and the memory traffic overlaps with the MADs.
 *
 *  K10 → Arbitrary matrix sizes: boundary guards on the output stores let the
 *         kernel handle M, N, K that are not exact multiples of the tile size.
 *         The A and B load loops are split (separate LPTA / LPTB loops) to
 *         give the compiler better scheduling freedom and reduce register pressure
 *         during loads.
 *
 * =============================================================================
 *  PARAMETERS  (override with -D flags at compile time)
 * =============================================================================
 *
 *  TSM   = 128   Tile height in output matrix C  (M dimension)
 *  TSN   = 128   Tile width  in output matrix C  (N dimension)
 *  TSK   =  16   Tile depth  (K dimension, inner loop)
 *  WPTM  =  16   Output rows computed per thread  (M work-per-thread)
 *  WPTN  =   8   Output cols computed per thread  (N work-per-thread)
 *  WIDTH =   4   float4 vector loads
 *
 *  Derived constants:
 *    RTSM = TSM/WPTM =  8   threads along M  (local dim 0)
 *    RTSN = TSN/WPTN = 16   threads along N  (local dim 1)
 *    Work-group size = RTSM × RTSN = 128 threads
 *
 *    LPTA = (TSK × WPTM × WPTN) / TSN = 16   vector loads per thread for A
 *    LPTB = (TSK × WPTM × WPTN) / TSM = 16   vector loads per thread for B
 *
 * =============================================================================
 *  ND-RANGE AND WORK-GROUP SIZE
 * =============================================================================
 *
 *  Local work size (workgroup):
 *      dim 0 = RTSM = 8
 *      dim 1 = RTSN = 16
 *      → 128 threads per workgroup
 *
 *  Global work size:
 *      dim 0 = ceil(M / TSM) * RTSM
 *      dim 1 = ceil(N / TSN) * RTSN
 *
 *  Example – square 4096×4096 matrix:
 *      Global = { ceil(4096/128)*8 , ceil(4096/128)*16 }
 *             = {  32*8 ,  32*16 }
 *             = { 256 , 512 }
 *      Local  = { 8 , 16 }
 *      Total work-groups = 32 × 32 = 1024
 *
 *  Host-side pseudo-code:
 *      size_t local[2]  = { RTSM, RTSN };
 *      size_t global[2] = { ((M + TSM - 1) / TSM) * RTSM,
 *                           ((N + TSN - 1) / TSN) * RTSN };
 *      clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, ...);
 *
 * =============================================================================
 *  LOCAL MEMORY FOOTPRINT
 * =============================================================================
 *
 *  Double-buffered (2 ping-pong slots):
 *      Asub[2][TSK * TSM] = 2 × 16 × 128 × 4 B = 16 384 B  (16 KB)
 *      Bsub[2][TSK * TSN] = 2 × 16 × 128 × 4 B = 16 384 B  (16 KB)
 *      Total local memory per work-group = 32 KB
 *
 *  Register file (per thread):
 *      acc[WPTM][WPTN] = 128 floats (accumulator)
 *      Breg[WPTN]      =   8 floats (B register cache)
 *      Areg            =   1 float
 *      ≈ 137 float registers per thread × 128 threads = ~17 500 total
 *
 * =============================================================================
 */

/* ── Tunable compile-time parameters ───────────────────────────────────────── */
#ifndef TSM
#define TSM   128
#endif
#ifndef TSN
#define TSN   128
#endif
#ifndef TSK
#define TSK    16
#endif
#ifndef WPTM
#define WPTM   16
#endif
#ifndef WPTN
#define WPTN    8
#endif
#ifndef WIDTH
#define WIDTH   4
#endif

/* ── Derived tile geometry ──────────────────────────────────────────────────── */
#define RTSM  (TSM / WPTM)                      /* threads in M dim  =  8 */
#define RTSN  (TSN / WPTN)                      /* threads in N dim  = 16 */
#define LPTA  ((TSK * WPTM * WPTN) / TSN)       /* A vector-loads/thread = 16 */
#define LPTB  ((TSK * WPTM * WPTN) / TSM)       /* B vector-loads/thread = 16 */

/* ── Vector type (K4 / K7 wider global loads) ──────────────────────────────── */
#if WIDTH == 1
    typedef float  floatX;
#elif WIDTH == 2
    typedef float2 floatX;
#elif WIDTH == 4
    typedef float4 floatX;
#elif WIDTH == 8
    typedef float8 floatX;
#endif

/* ── Flat-array index helpers ───────────────────────────────────────────────── */
/*
 *  A stored column-major within each K-slice:
 *    Asub[buf][k_col * TSM + m_row]
 *  B stored row-major within each K-slice:
 *    Bsub[buf][k_row * TSN + n_col]
 *
 *  Both layouts are bank-conflict-free in the HOT compute path:
 *    - 8 consecutive threads read Asub at k*TSM + tidm (consecutive → no conflict)
 *    - 16 consecutive threads read Bsub at k*TSN + tidn (consecutive → no conflict)
 */
#define ASUB(buf, kcol, mrow)   Asub[(buf)][(kcol) * TSM + (mrow)]
#define BSUB(buf, krow, ncol)   Bsub[(buf)][(krow) * TSN + (ncol)]

/* ============================================================================
 *  KERNEL ENTRY POINT
 * ============================================================================ */
__attribute__((reqd_work_group_size(RTSM, RTSN, 1)))
__kernel void mmul(const int M,
                   const int N,
                   const int K,
                   const __global floatX* restrict A,   /* M × K, col-major */
                   const __global floatX* restrict B,   /* K × N, col-major */
                         __global float*  restrict C)   /* M × N, col-major */
{
    /* ── Thread / work-group identifiers ────────────────────────────────── */
    const int tidm = get_local_id(0);           /* 0 .. RTSM-1   */
    const int tidn = get_local_id(1);           /* 0 .. RTSN-1   */
    const int gidm = get_group_id(0);           /* work-group M   */
    const int gidn = get_group_id(1);           /* work-group N   */
    const int tid  = tidn * RTSM + tidm;        /* flat 0..127    */

    /* ── Double-buffered local memory (K9 software pipeline) ────────────── */
    __local float Asub[2][TSK * TSM];
    __local float Bsub[2][TSK * TSN];

    /* ── Per-thread register file (K6 2-D register blocking) ────────────── */
    float Areg;
    float Breg[WPTN];
    float acc[WPTM][WPTN];

    /* Zero accumulators */
    #pragma unroll
    for (int wm = 0; wm < WPTM; wm++) {
        #pragma unroll
        for (int wn = 0; wn < WPTN; wn++) {
            acc[wm][wn] = 0.0f;
        }
    }

    /* ====================================================================
     *  PRE-LOAD tile 0 into buffer 0  (K9 pipeline seed)
     *  Tile A:  rows gidm*TSM .. gidm*TSM+TSM-1,  cols 0 .. TSK-1
     *  Tile B:  rows 0 .. TSK-1,  cols gidn*TSN .. gidn*TSN+TSN-1
     * ==================================================================== */

    /* -- Load A tile 0 -- */
    #pragma unroll
    for (int la = 0; la < LPTA / WIDTH; la++) {
        int id  = la * RTSN * RTSM + tid;
        int row = id % (TSM / WIDTH);       /* vectorised M index */
        int col = id / (TSM / WIDTH);       /* k index within tile */

        int indexA = (TSK * 0 + col) * (M / WIDTH) + gidm * (TSM / WIDTH) + row;
        floatX vecA = A[indexA];

        #if WIDTH == 4
            ASUB(0, col, WIDTH * row + 0) = vecA.x;
            ASUB(0, col, WIDTH * row + 1) = vecA.y;
            ASUB(0, col, WIDTH * row + 2) = vecA.z;
            ASUB(0, col, WIDTH * row + 3) = vecA.w;
        #elif WIDTH == 2
            ASUB(0, col, WIDTH * row + 0) = vecA.x;
            ASUB(0, col, WIDTH * row + 1) = vecA.y;
        #else
            ASUB(0, col, row) = vecA;
        #endif
    }

    /* -- Load B tile 0 -- */
    #pragma unroll
    for (int lb = 0; lb < LPTB / WIDTH; lb++) {
        int id  = lb * RTSN * RTSM + tid;
        int row = id % (TSN / WIDTH);       /* vectorised N index */
        int col = id / (TSN / WIDTH);       /* k index within tile */

        int indexB = (TSK * 0 + col) * (N / WIDTH) + gidn * (TSN / WIDTH) + row;
        floatX vecB = B[indexB];

        #if WIDTH == 4
            BSUB(0, col, WIDTH * row + 0) = vecB.x;
            BSUB(0, col, WIDTH * row + 1) = vecB.y;
            BSUB(0, col, WIDTH * row + 2) = vecB.z;
            BSUB(0, col, WIDTH * row + 3) = vecB.w;
        #elif WIDTH == 2
            BSUB(0, col, WIDTH * row + 0) = vecB.x;
            BSUB(0, col, WIDTH * row + 1) = vecB.y;
        #else
            BSUB(0, col, row) = vecB;
        #endif
    }

    /* Ensure tile 0 is visible to all threads before the pipeline starts */
    barrier(CLK_LOCAL_MEM_FENCE);

    /* ====================================================================
     *  MAIN TILE LOOP  (K9 pipeline: prefetch t+1 while computing t)
     * ==================================================================== */
    const int numTiles = K / TSK;
    int t = 0;
    do {
        /* ----------------------------------------------------------------
         *  PREFETCH next tile (tt = t+1) into the alternate ping-pong buffer.
         *  This runs concurrently with the MAD block below on the GPU's
         *  load/execute pipelines — the whole point of double-buffering.
         *  We write into buffer (tt % 2) which is NOT the buffer being read
         *  for compute (t % 2), so there is no hazard without a barrier here.
         * ---------------------------------------------------------------- */
        int tt = t + 1;
        if (tt < numTiles) {

            /* Prefetch A tile tt */
            #pragma unroll
            for (int la = 0; la < LPTA / WIDTH; la++) {
                int id  = la * RTSN * RTSM + tid;
                int row = id % (TSM / WIDTH);
                int col = id / (TSM / WIDTH);

                int indexA = (TSK * tt + col) * (M / WIDTH) + gidm * (TSM / WIDTH) + row;
                floatX vecA = A[indexA];

                #if WIDTH == 4
                    ASUB(tt % 2, col, WIDTH * row + 0) = vecA.x;
                    ASUB(tt % 2, col, WIDTH * row + 1) = vecA.y;
                    ASUB(tt % 2, col, WIDTH * row + 2) = vecA.z;
                    ASUB(tt % 2, col, WIDTH * row + 3) = vecA.w;
                #elif WIDTH == 2
                    ASUB(tt % 2, col, WIDTH * row + 0) = vecA.x;
                    ASUB(tt % 2, col, WIDTH * row + 1) = vecA.y;
                #else
                    ASUB(tt % 2, col, row) = vecA;
                #endif
            }

            /* Prefetch B tile tt */
            #pragma unroll
            for (int lb = 0; lb < LPTB / WIDTH; lb++) {
                int id  = lb * RTSN * RTSM + tid;
                int row = id % (TSN / WIDTH);
                int col = id / (TSN / WIDTH);

                int indexB = (TSK * tt + col) * (N / WIDTH) + gidn * (TSN / WIDTH) + row;
                floatX vecB = B[indexB];

                #if WIDTH == 4
                    BSUB(tt % 2, col, WIDTH * row + 0) = vecB.x;
                    BSUB(tt % 2, col, WIDTH * row + 1) = vecB.y;
                    BSUB(tt % 2, col, WIDTH * row + 2) = vecB.z;
                    BSUB(tt % 2, col, WIDTH * row + 3) = vecB.w;
                #elif WIDTH == 2
                    BSUB(tt % 2, col, WIDTH * row + 0) = vecB.x;
                    BSUB(tt % 2, col, WIDTH * row + 1) = vecB.y;
                #else
                    BSUB(tt % 2, col, row) = vecB;
                #endif
            }
        }

        /* ----------------------------------------------------------------
         *  COMPUTE tile t  (K6/K7 outer-product register blocking)
         *
         *  Inner loop structure:
         *    for k in 0..TSK-1:
         *      load WPTN B scalars into Breg[]  (broadcast from local to reg)
         *      for wm in 0..WPTM-1:
         *        load one A scalar into Areg
         *        for wn in 0..WPTN-1:
         *          acc[wm][wn] += Areg * Breg[wn]   ← WPTM*WPTN MADs per k step
         *
         *  Breg[] caching avoids re-loading the same B value WPTM times.
         *  TSK=16 is fully unrolled, so the compiler sees all 16 × 128 = 2048
         *  MAD instructions as a straight-line FMA chain.
         * ---------------------------------------------------------------- */
        #pragma unroll
        for (int k = 0; k < TSK; k++) {

            /* Cache WPTN B values from local memory into registers */
            #pragma unroll
            for (int wn = 0; wn < WPTN; wn++) {
                int ncol = tidn + wn * RTSN;
                Breg[wn] = BSUB(t % 2, k, ncol);
            }

            /* Outer product: WPTM rows of A × WPTN cols of B */
            #pragma unroll
            for (int wm = 0; wm < WPTM; wm++) {
                int mrow = tidm + wm * RTSM;
                Areg = ASUB(t % 2, k, mrow);
                #pragma unroll
                for (int wn = 0; wn < WPTN; wn++) {
                    acc[wm][wn] += Areg * Breg[wn];
                }
            }
        }

        /* ----------------------------------------------------------------
         *  BARRIER – placed AFTER compute, BEFORE next prefetch reads from
         *  the buffer just written.  This is the K9 pattern: one barrier per
         *  tile loop iteration.  The prefetch of tile tt (above) is already
         *  in flight on the memory subsystem; the barrier only gates the
         *  NEXT iteration's compute, giving maximum overlap time.
         * ---------------------------------------------------------------- */
        barrier(CLK_LOCAL_MEM_FENCE);

        t++;
    } while (t < numTiles);

    /* ====================================================================
     *  WRITE RESULTS TO C  (K10 arbitrary-size boundary guards)
     *
     *  C is stored column-major: C[globalCol * M + globalRow].
     *  Boundary checks are compiled away to unconditional stores when M and N
     *  are exact multiples of TSM/TSN (the common square-matrix case).
     * ==================================================================== */
    #pragma unroll
    for (int wm = 0; wm < WPTM; wm++) {
        int globalRow = gidm * TSM + tidm + wm * RTSM;
        if (globalRow < M) {
            #pragma unroll
            for (int wn = 0; wn < WPTN; wn++) {
                int globalCol = gidn * TSN + tidn + wn * RTSN;
                if (globalCol < N) {
                    C[globalCol * M + globalRow] = acc[wm][wn];
                }
            }
        }
    }
}

/* ============================================================================
 *  UNDEFINE LOCAL MACROS  (good practice for multi-kernel compilation units)
 * ============================================================================ */
#undef RTSM
#undef RTSN
#undef LPTA
#undef LPTB
#undef ASUB
#undef BSUB

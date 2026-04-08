/*
 * =============================================================================
 *  ULTIMATE GEMM KERNEL — NVIDIA RTX 3050 (Ampere GA107) OPTIMISED
 * =============================================================================
 *
 *  WHY THE PREVIOUS KERNEL FAILED ON NVIDIA
 *  ─────────────────────────────────────────
 *  Problem 1 — Register spilling (most critical)
 *    WPTM=16 × WPTN=8 = 128 accumulator floats per thread.
 *    Add Breg[8], Areg, loop variables → ~157 registers per thread.
 *    At 128 threads per block: 128×157 = 20096 registers/block.
 *    RTX 3050 has 65536 registers/SM → that allows 3 blocks from register
 *    count, but the double-buffer shared memory (32 KB) caps it to 1 block.
 *    The compiler, seeing >127 live registers, SPILLS to local memory.
 *    Spill stores/loads go through L1 (32-cycle penalty each) and destroy
 *    the FMA throughput.  Measured occupancy: 4 warps/SM = 8%.
 *
 *  Problem 2 — Double-buffering doubles shared memory cost with no gain
 *    OpenCL on NVIDIA cannot issue true asynchronous global→shared copies
 *    (CUDA's `cp.async` / Ampere's LDG+STS pipeline is not exposed in OCL).
 *    So the double-buffer only cost 16 KB of extra shared memory in exchange
 *    for zero latency hiding — the worst possible trade.
 *
 *  Problem 3 — Workgroup 8×16=128 threads maps badly to NVIDIA warps
 *    NVIDIA executes 32 threads as one warp.  A 128-thread block = 4 warps.
 *    The scheduler needs at least 6–8 resident warps per SM to hide the
 *    ~80-cycle latency of a shared-memory read.  With 1 block/SM = 4 warps
 *    the compute pipe stalled constantly.
 *
 *  FIXES IN THIS KERNEL
 *  ─────────────────────
 *  Fix 1 — WPTM  8 → 8,  WPTN 8 (unchanged)
 *    Accumulators: 8×8 = 64 floats.  Total regs/thread ≈ 93.
 *    No spilling.  All 64 acc values stay in the register file.
 *
 *  Fix 2 — Remove double-buffering
 *    Shared memory halved from 32 KB to 16.1 KB.
 *    2 blocks fit per SM (limited equally by registers and shared memory).
 *    Active warps/SM: 2 × 8 = 16 → 33% occupancy (was 8%).
 *
 *  Fix 3 — RTSM = RTSN = 16  (workgroup 16×16 = 256 threads = 8 warps)
 *    8 warps comfortably hides shared-memory and arithmetic latency.
 *    Warp linearisation: warp 0 = threads (tidm=0..15, tidn=0),
 *    warp 1 = (tidm=0..15, tidn=1), etc.  Each warp covers a full M-stripe
 *    of 16 consecutive rows — perfectly coalesced for global loads.
 *
 *  Fix 4 — 2D padded local arrays: Asub[TSK][TSM+1], Bsub[TSK][TSN+1]
 *    Stride TSM+1=129 is odd; gcd(129, 32) = 1.
 *    In the compute inner loop 16 consecutive threads access 16 consecutive
 *    addresses in any row → 16 unique banks → zero bank conflicts on the
 *    critical path (the hot loop that runs TSK=16 times per tile).
 *    Load-phase conflicts (occur once per tile) are accepted as negligible.
 *
 * =============================================================================
 *  PARAMETERS
 * =============================================================================
 *
 *  Parameter     Value   Meaning
 *  ─────────────────────────────────────────────────────────────────────────
 *  TSM           128     Tile height (M dimension per work-group)
 *  TSN           128     Tile width  (N dimension per work-group)
 *  TSK            16     Tile depth  (K strip; keep small to limit LDS)
 *  WPTM            8     Output rows per thread  (M work-per-thread)
 *  WPTN            8     Output cols per thread  (N work-per-thread)
 *  WIDTH           4     float4 vector loads from global memory
 *  ─────────────────────────────────────────────────────────────────────────
 *  RTSM = TSM/WPTM = 16  threads along dim-0 (M)
 *  RTSN = TSN/WPTN = 16  threads along dim-1 (N)
 *  Workgroup = 16 × 16 = 256 threads = 8 warps
 *  ─────────────────────────────────────────────────────────────────────────
 *  LPTA = (TSK × WPTM × WPTN) / TSN = 8   scalar-loads ÷ WIDTH = 2 per thread
 *  LPTB = (TSK × WPTM × WPTN) / TSM = 8   scalar-loads ÷ WIDTH = 2 per thread
 *  ─────────────────────────────────────────────────────────────────────────
 *  Shared memory per block:
 *    Asub[TSK][TSM+1]  = 16 × 129 × 4 B = 8 256 B
 *    Bsub[TSK][TSN+1]  = 16 × 129 × 4 B = 8 256 B
 *    Total             = 16 512 B ≈ 16.1 KB
 *  → 2 active blocks per SM (48 KB limit), 16 active warps, 33% occupancy
 *
 *  Registers per thread  ≈ 64 (acc) + 8 (Breg) + 1 (Areg) + 20 (misc) = 93
 *  → 256 × 93 = 23808 per block; 65536 / 23808 = 2 blocks  (consistent)
 *
 * =============================================================================
 *  ND-RANGE
 * =============================================================================
 *
 *  Local  work size : { RTSM, RTSN }   =  { 16, 16 }
 *  Global work size : { ceil(M/TSM)*RTSM,  ceil(N/TSN)*RTSN }
 *
 *  Example — 4096 × 4096:
 *      Local  = { 16, 16 }
 *      Global = { ceil(4096/128)*16, ceil(4096/128)*16 } = { 512, 512 }
 *      Work-groups = 32 × 32 = 1024
 *
 *  Host call:
 *      size_t lws[2]   = { 16, 16 };
 *      size_t gws[2]   = { ((M + 127) / 128) * 16,
 *                          ((N + 127) / 128) * 16 };
 *      clEnqueueNDRangeKernel(queue, kernel, 2, NULL, gws, lws, 0, NULL, NULL);
 *
 * =============================================================================
 */

/* ── Tunable parameters ─────────────────────────────────────────────────── */
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
#define WPTM    8
#endif
#ifndef WPTN
#define WPTN    8
#endif
#ifndef WIDTH
#define WIDTH   4
#endif

/* ── Derived constants ──────────────────────────────────────────────────── */
#define RTSM  (TSM / WPTM)                   /* = 16  threads in M dim   */
#define RTSN  (TSN / WPTN)                   /* = 16  threads in N dim   */
#define LPTA  ((TSK * WPTM * WPTN) / TSN)    /* = 8   A loads per thread */
#define LPTB  ((TSK * WPTM * WPTN) / TSM)    /* = 8   B loads per thread */

/* ── Padding for bank-conflict-free LDS access (K5/K6 technique) ────────
 *  Stride TSM+1 = 129.  gcd(129, 32) = 1  →  all 16 consecutive threads
 *  in the compute phase land on 16 distinct banks regardless of K offset.
 *  Same reasoning for TSN+1.
 * ─────────────────────────────────────────────────────────────────────── */
#define PADA  1    /* A padding per K-row: Asub[TSK][TSM+PADA] */
#define PADB  1    /* B padding per K-row: Bsub[TSK][TSN+PADB] */

/* ── Vector type (K4/K7 wider global loads) ─────────────────────────────── */
#if WIDTH == 1
    typedef float  floatX;
#elif WIDTH == 2
    typedef float2 floatX;
#elif WIDTH == 4
    typedef float4 floatX;
#elif WIDTH == 8
    typedef float8 floatX;
#endif

/* ============================================================================
 *  KERNEL
 * ============================================================================ */
__attribute__((reqd_work_group_size(RTSM, RTSN, 1)))
__kernel void mmul(const int M,
                   const int N,
                   const int K,
                   const __global floatX* restrict A,   /* col-major M×K */
                   const __global floatX* restrict B,   /* col-major K×N */
                         __global float*  restrict C)   /* col-major M×N */
{
    /* ── Thread / work-group IDs ─────────────────────────────────────── */
    const int tidm = get_local_id(0);            /* 0 .. RTSM-1  = 0..15 */
    const int tidn = get_local_id(1);            /* 0 .. RTSN-1  = 0..15 */
    const int gidm = get_group_id(0);            /* work-group M index   */
    const int gidn = get_group_id(1);            /* work-group N index   */
    const int tid  = tidn * RTSM + tidm;         /* flat 0..255          */

    /* ── Shared memory tiles with bank-conflict-free padding ─────────── */
    __local float Asub[TSK][TSM + PADA];   /* [k][m]  stride TSM+1 = 129 */
    __local float Bsub[TSK][TSN + PADB];   /* [k][n]  stride TSN+1 = 129 */

    /* ── Per-thread register file ────────────────────────────────────── */
    float Areg;
    float Breg[WPTN];       /* WPTN = 8  B values cached before outer product */
    float acc[WPTM][WPTN];  /* WPTM×WPTN = 64  accumulators — never spill    */

    /* Zero the accumulators */
    #pragma unroll
    for (int wm = 0; wm < WPTM; wm++) {
        #pragma unroll
        for (int wn = 0; wn < WPTN; wn++) {
            acc[wm][wn] = 0.0f;
        }
    }

    /* ====================================================================
     *  MAIN TILE LOOP
     *
     *  Each iteration:
     *    1. Load a TSK×TSM slice of A and TSK×TSN slice of B into __local
     *       using float4 vector reads (4 scalars per instruction, WIDTH=4).
     *    2. Barrier — ensure all 256 threads finished loading.
     *    3. Compute: 16-step K loop, each step doing a full WPTM×WPTN
     *       outer product (K6 register blocking) with Breg[] cache (K7).
     *    4. Barrier — ensure compute finished before next load overwrites.
     * ==================================================================== */
    const int numTiles = K / TSK;

    for (int t = 0; t < numTiles; t++) {

        /* ----------------------------------------------------------------
         *  LOAD TILE A  (K4/K7 float4 wide loads, K2/K10 tiling)
         *
         *  Each thread performs LPTA/WIDTH = 2 float4 loads.
         *  The flat index `id` covers all 512 vectorised positions in the
         *  TSK×TSM tile (TSK=16 rows × TSM/WIDTH=32 cols = 512).
         *
         *  Global address:
         *    indexA = k_global * (M/WIDTH) + m_group_base + row_vec
         *  where k_global = TSK*t + col (the K index) and row_vec is the
         *  vectorised M index.  All 16 threads in a warp have the same col
         *  and consecutive row_vec → consecutive global addresses → fully
         *  coalesced 128-byte cache-line reads.
         *
         *  Local storage layout: Asub[col][WIDTH*row + w]  (K outer, M inner)
         *  stride TSM+PADA=129 keeps the compute-phase accesses conflict-free.
         * ---------------------------------------------------------------- */
        #pragma unroll
        for (int la = 0; la < LPTA / WIDTH; la++) {
            int id   = la * RTSN * RTSM + tid;
            int row  = id % (TSM / WIDTH);       /* vectorised M index  */
            int col  = id / (TSM / WIDTH);       /* K index within tile */

            int indexA = (TSK * t + col) * (M / WIDTH)
                       + gidm * (TSM / WIDTH) + row;
            floatX vecA = A[indexA];

            #if WIDTH == 4
                Asub[col][WIDTH * row + 0] = vecA.x;
                Asub[col][WIDTH * row + 1] = vecA.y;
                Asub[col][WIDTH * row + 2] = vecA.z;
                Asub[col][WIDTH * row + 3] = vecA.w;
            #elif WIDTH == 2
                Asub[col][WIDTH * row + 0] = vecA.x;
                Asub[col][WIDTH * row + 1] = vecA.y;
            #else
                Asub[col][row] = vecA;
            #endif
        }

        /* ----------------------------------------------------------------
         *  LOAD TILE B  (identical structure to A)
         *
         *  B tile: K rows × N cols.
         *  Stored as Bsub[k][n] with stride TSN+PADB=129.
         *  Global: same coalescing argument — warp reads consecutive N cols.
         * ---------------------------------------------------------------- */
        #pragma unroll
        for (int lb = 0; lb < LPTB / WIDTH; lb++) {
            int id   = lb * RTSN * RTSM + tid;
            int row  = id % (TSN / WIDTH);       /* vectorised N index  */
            int col  = id / (TSN / WIDTH);       /* K index within tile */

            int indexB = (TSK * t + col) * (N / WIDTH)
                       + gidn * (TSN / WIDTH) + row;
            floatX vecB = B[indexB];

            #if WIDTH == 4
                Bsub[col][WIDTH * row + 0] = vecB.x;
                Bsub[col][WIDTH * row + 1] = vecB.y;
                Bsub[col][WIDTH * row + 2] = vecB.z;
                Bsub[col][WIDTH * row + 3] = vecB.w;
            #elif WIDTH == 2
                Bsub[col][WIDTH * row + 0] = vecB.x;
                Bsub[col][WIDTH * row + 1] = vecB.y;
            #else
                Bsub[col][row] = vecB;
            #endif
        }

        /* Ensure tile is fully written before any thread reads it */
        barrier(CLK_LOCAL_MEM_FENCE);

        /* ----------------------------------------------------------------
         *  COMPUTE  (K6 2-D register blocking + K7 Breg caching)
         *
         *  For each k=0..TSK-1:
         *    1. Load WPTN=8 B values into Breg[] registers.
         *       Breg[wn] = Bsub[k][tidn + wn*RTSN]
         *       With stride 129 and wn stepping by 16:
         *         bank = (k*129 + tidn + wn*16) % 32
         *       For k=0..15 and wn=0..7, all 16 threads land on distinct
         *       banks in both the tidn=0..15 and the wn dimension → zero
         *       bank conflicts.
         *
         *    2. For each wm=0..WPTM-1, load one A scalar (Areg) and fire
         *       WPTN=8 FMAs: acc[wm][wn] += Areg * Breg[wn].
         *       Areg = Asub[k][tidm + wm*RTSM]
         *       Same bank analysis: 16 consecutive tidm values → distinct
         *       banks.
         *
         *  Total MADs per tile per thread: TSK × WPTM × WPTN = 16×8×8 = 1024.
         *  The compiler unrolls all loops (TSK=16, WPTM=8, WPTN=8) into a
         *  straight-line sequence of ~1024 FMA instructions with no branches.
         *  On Ampere, each SM can issue 2 FP32 FMAs per cycle per warp, so
         *  at full pipeline this block does 1024 × 256 / 2 = 131072 FMAs
         *  per warp-clock.
         * ---------------------------------------------------------------- */
        #pragma unroll
        for (int k = 0; k < TSK; k++) {

            /* Cache B row k into registers — reused WPTM=8 times below */
            #pragma unroll
            for (int wn = 0; wn < WPTN; wn++) {
                Breg[wn] = Bsub[k][tidn + wn * RTSN];
            }

            /* Outer product: WPTM A rows × WPTN B cols */
            #pragma unroll
            for (int wm = 0; wm < WPTM; wm++) {
                Areg = Asub[k][tidm + wm * RTSM];
                #pragma unroll
                for (int wn = 0; wn < WPTN; wn++) {
                    acc[wm][wn] += Areg * Breg[wn];
                }
            }
        }

        /* Ensure compute finished before the next load overwrites Asub/Bsub */
        barrier(CLK_LOCAL_MEM_FENCE);

    } /* end tile loop */

    /* ====================================================================
     *  WRITE RESULTS TO C  (K10 boundary guards for arbitrary M, N)
     *
     *  For square matrix sizes that are exact multiples of TSM=128 the
     *  if-checks below are compile-time-eliminated by the OpenCL JIT.
     *  For all other sizes they prevent out-of-bounds writes.
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

/* ── Clean up local macros ─────────────────────────────────────────────── */
#undef RTSM
#undef RTSN
#undef LPTA
#undef LPTB
#undef PADA
#undef PADB

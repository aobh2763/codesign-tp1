// =============================================================================
//  myGEMM_optimised.cl
//
//  Pure OpenCL SGEMM kernel — no CUDA, no vendor-specific extensions.
//  Combines every optimisation from the tutorial pages 3-13 that is
//  expressible in standard OpenCL C:
//
//  [K2]  Local-memory tiling          – cache tiles in on-chip __local memory
//  [K3]  Work-per-thread (column)     – each thread computes WPT output values
//  [K4]  Vector loads (WIDTH)         – 128-bit global loads with float4
//  [K5]  Transposed B + rect tiles    – B pre-transposed, TSM/TSN/TSK tunable
//  [K6]  2-D register blocking        – WPTM×WPTN acc[] held in registers
//  [K7]  Vectors + register blocking  – float4 loads feeding the reg-blocked loop
//  [K9]  Double-buffered prefetching  – overlap next-tile load with current FMAs
//  [K10] Decoupled A/B load loops     – independent TSM/TSN, zero-pad for any size
//
//  What is NOT included (CUDA / Kepler-only, K8):
//    - __ldg texture-cache intrinsic
//    - warp-shuffle (__shfl)
//    - 64-bit PTX addressing
//    - SM 3.5 assembly tricks
//
// =============================================================================
//  QUICK-START
//  -----------
//  Build options (pass to clBuildProgram):
//    -DTSM=128 -DTSN=128 -DTSK=16 -DWPTM=8 -DWPTN=8 -DWIDTH=4
//
//  Host launch config:
//    local  = { RTSM, RTSN }      e.g. { 16, 16 }
//    global = { M/WPTM, N/WPTN }  e.g. { 256, 256 } for N=2048
//
//  Host responsibilities:
//    1. Run the `transpose` kernel on B  → produces Bt = B^T
//    2. Pad A  so that rows(M) % TSM == 0  and  cols(K) % TSK == 0
//       Pad Bt so that rows(N) % TSN == 0  and  cols(K) % TSK == 0
//       (fill extra elements with 0.0f)
//    3. Launch `myGEMM`
//    4. Copy only the un-padded M×N region of C back to the host
// =============================================================================


// -----------------------------------------------------------------------------
//  [1]  COMPILE-TIME TUNING PARAMETERS
//       Override any of these with -D flags at build time.
// -----------------------------------------------------------------------------

#ifndef TSM
#  define TSM   128   // Tile size in the M dimension (rows of A, rows of C)
#endif
#ifndef TSN
#  define TSN   128   // Tile size in the N dimension (rows of Bt, cols of C)
#endif
#ifndef TSK
#  define TSK    16   // Tile size in the K dimension (cols of A, cols of Bt)
                      // Keep small: local memory = 2*(TSK*TSM + TSK*TSN)*4 bytes
#endif
#ifndef WPTM
#  define WPTM    8   // Work-per-thread in M: each thread owns WPTM rows of C
#endif
#ifndef WPTN
#  define WPTN    8   // Work-per-thread in N: each thread owns WPTN cols of C
#endif
#ifndef WIDTH
#  define WIDTH   4   // Vector width for global loads: 1 (scalar), 2, or 4
                      // Matrix sizes must be divisible by WIDTH.
                      // Use WIDTH=1 if float4 causes issues on your device.
#endif

// Derived constants — do NOT change these
#define RTSM  (TSM / WPTM)                      // Work-group size, dim 0 (e.g. 16)
#define RTSN  (TSN / WPTN)                      // Work-group size, dim 1 (e.g. 16)
#define LPTA  ((TSK * TSM) / (RTSM * RTSN))    // Global loads per thread for A tile
#define LPTB  ((TSK * TSN) / (RTSM * RTSN))    // Global loads per thread for B tile


// -----------------------------------------------------------------------------
//  [2]  VECTOR TYPE  [K4, K7]
//       floatX is float4 / float2 / float depending on WIDTH.
//       floatX_get(v, i) extracts the i-th scalar component.
// -----------------------------------------------------------------------------
#if WIDTH == 4
    typedef float4 floatX;
    // Cast to float* so we can index components 0-3 uniformly
    #define floatX_get(vec, i)  (((float*)&(vec))[i])
#elif WIDTH == 2
    typedef float2 floatX;
    #define floatX_get(vec, i)  (((float*)&(vec))[i])
#else
    // WIDTH == 1 : plain scalar, no vector instructions
    typedef float  floatX;
    #define floatX_get(vec, i)  (vec)
#endif


// -----------------------------------------------------------------------------
//  [3]  TRANSPOSE KERNEL  [K5]
//
//  Computes  output[j][i] = input[i][j]  for a P×Q matrix.
//  Uses a shared-memory shuffle buffer so BOTH the global read and the
//  global write are coalesced (threads in the same warp access consecutive
//  addresses).
//
//  Launch config:
//    local  = { TRANSPOSEX, TRANSPOSEY }  = { 16, 16 }
//    global = { ceil(P/16)*16, ceil(Q/16)*16 }
//
//  For GEMM: call transpose(N, K, B, Bt)  to get  Bt = B^T  (shape K×N→N×K)
// -----------------------------------------------------------------------------
#define TRANSPOSEX 16
#define TRANSPOSEY 16

__kernel void transpose(const int P,              // number of columns in input
                        const int Q,              // number of rows    in input
                        const __global float* restrict input,   // P×Q
                              __global float*          output)  // Q×P
{
    const int tx  = get_local_id(0);
    const int ty  = get_local_id(1);

    // Global indices into the input matrix
    const int id0 = get_group_id(0) * TRANSPOSEX + tx;   // column index (0..P-1)
    const int id1 = get_group_id(1) * TRANSPOSEY + ty;   // row    index (0..Q-1)

    // Shared shuffle buffer — indexed [ty][tx] on read, [tx][ty] on write
    // This swap is what performs the transpose inside shared memory.
    __local float buf[TRANSPOSEX][TRANSPOSEY];

    // Step 1: coalesced READ from global input → shared buffer
    //   Threads with consecutive tx read consecutive columns → coalesced
    if (id0 < P && id1 < Q)
        buf[ty][tx] = input[id1 * P + id0];

    // Wait until every thread in the work-group has filled its cell
    barrier(CLK_LOCAL_MEM_FENCE);

    // Step 2: coalesced WRITE from shared buffer → global output
    //   We swap group indices so threads with consecutive tx now write
    //   consecutive rows of the transposed output → coalesced again
    const int nid0 = get_group_id(1) * TRANSPOSEY + tx;  // new col index (0..Q-1)
    const int nid1 = get_group_id(0) * TRANSPOSEX + ty;  // new row index (0..P-1)

    if (nid0 < Q && nid1 < P)
        output[nid1 * Q + nid0] = buf[tx][ty];
}


// -----------------------------------------------------------------------------
//  [4]  TILE-LOAD MACROS  [K5, K7, K10]
//
//  These macros cooperatively load one TSK-wide tile of A or Bt into the
//  corresponding __local buffer slot.
//
//  Why macros?  The prefetch loop (K9) needs to inline the load code into
//  two different places (before the loop and inside the loop) while keeping
//  the barrier placement correct.  A helper function would require an extra
//  barrier or extra __local pointer indirection.
//
//  LOAD_TILE_A(t, buf)
//    Loads columns [t*TSK .. (t+1)*TSK-1] of A (shape M×K, col-major)
//    into Asub[buf][0 .. TSK*TSM-1].
//    Layout in Asub: Asub[buf][k_local * TSM + m_local]
//
//  LOAD_TILE_B(t, buf)
//    Loads columns [t*TSK .. (t+1)*TSK-1] of Bt (shape N×K, col-major)
//    into Bsub[buf][0 .. TSK*TSN-1].
//    Layout in Bsub: Bsub[buf][k_local * TSN + n_local]
//
//  The [K10] decoupled loops mean A uses TSM in its modulo and B uses TSN,
//  so the two tile dimensions can differ without breaking index arithmetic.
//
//  The [K7] vector loads fetch WIDTH floats per instruction; the WIDTH
//  components are then scattered into consecutive scalar positions in the
//  __local array so the compute macro can read them as plain scalars.
// -----------------------------------------------------------------------------

#define LOAD_TILE_A(t, buf)                                                   \
    for (int _la = 0; _la < LPTA / WIDTH; ++_la) {                            \
        /* Flatten 2D thread ID into a single index 0..(RTSM*RTSN-1) */       \
        int _tid = tidn * RTSM + tidm;                                        \
        /* Each iteration handles a different chunk of the tile */             \
        int _id  = _la * RTSN * RTSM + _tid;                                 \
        /* [K10] row/col decomposition uses TSM so A and B loops are          \
           independent and TSM can differ from TSN */                         \
        int _row = _id % (TSM / WIDTH);   /* position within a tile column */ \
        int _col = _id / (TSM / WIDTH);   /* which k-column inside the tile */\
        /* Global memory address: column-major A, offset by work-group gm */  \
        int _gRow = gm / WIDTH + _row;                                        \
        int _gCol = (t) * TSK + _col;                                         \
        /* [K7] Single 128-bit (float4) or 64-bit (float2) global load */     \
        floatX _vec = A[_gCol * (M / WIDTH) + _gRow];                         \
        /* Scatter WIDTH floats into the scalar __local array */               \
        for (int _w = 0; _w < WIDTH; ++_w)                                    \
            Asub[(buf)][_col * TSM + _row * WIDTH + _w] =                     \
                floatX_get(_vec, _w);                                         \
    }

#define LOAD_TILE_B(t, buf)                                                   \
    for (int _lb = 0; _lb < LPTB / WIDTH; ++_lb) {                            \
        int _tid = tidn * RTSM + tidm;                                        \
        int _id  = _lb * RTSN * RTSM + _tid;                                 \
        /* [K10] row/col uses TSN here, independent of TSM */                 \
        int _row = _id % (TSN / WIDTH);                                       \
        int _col = _id / (TSN / WIDTH);                                       \
        int _gRow = gn / WIDTH + _row;                                        \
        int _gCol = (t) * TSK + _col;                                         \
        floatX _vec = Bt[_gCol * (N / WIDTH) + _gRow];                        \
        for (int _w = 0; _w < WIDTH; ++_w)                                    \
            Bsub[(buf)][_col * TSN + _row * WIDTH + _w] =                     \
                floatX_get(_vec, _w);                                         \
    }


// -----------------------------------------------------------------------------
//  [5]  COMPUTE MACRO  [K6]
//
//  COMPUTE_TILE(buf)
//    Iterates k = 0..TSK-1 over the tile currently in Asub[buf] / Bsub[buf].
//    For each k:
//      1. Load WPTM A-values from __local into private registers (regA).
//      2. Load WPTN B-values from __local into private registers (regB).
//      3. Compute the outer product: all WPTM*WPTN acc[][] cells updated
//         with fma() — zero additional memory reads for step 3.
//
//  Memory traffic per k-iteration:
//    Reads : WPTM + WPTN  local-memory reads  (e.g. 8+8 = 16)
//    FMAs  : WPTM * WPTN                       (e.g. 8*8 = 64)
//  → ratio : 4 FMAs per local read  (vs 1/3 in the naive kernel)
// -----------------------------------------------------------------------------

#define COMPUTE_TILE(buf)                                                     \
    for (int _k = 0; _k < TSK; ++_k) {                                        \
        /* Step 1: WPTM A-values → private registers */                       \
        float _regA[WPTM];                                                    \
        for (int _wm = 0; _wm < WPTM; ++_wm)                                 \
            _regA[_wm] = Asub[(buf)][_k * TSM + tidm + _wm * RTSM];          \
        /* Step 2: WPTN B-values → private registers */                       \
        float _regB[WPTN];                                                    \
        for (int _wn = 0; _wn < WPTN; ++_wn)                                 \
            _regB[_wn] = Bsub[(buf)][_k * TSN + tidn + _wn * RTSN];          \
        /* Step 3: outer-product FMAs — all register operands, no mem reads */\
        for (int _wm = 0; _wm < WPTM; ++_wm)                                 \
            for (int _wn = 0; _wn < WPTN; ++_wn)                             \
                acc[_wm][_wn] = fma(_regA[_wm], _regB[_wn], acc[_wm][_wn]); \
    }


// -----------------------------------------------------------------------------
//  [6]  MAIN GEMM KERNEL
//
//  Computes  C = A * B  (single-precision, column-major)
//
//  Arguments:
//    M   — number of rows    in A and C  (must be multiple of TSM)
//    N   — number of columns in B and C  (must be multiple of TSN)
//    K   — shared inner dimension        (must be multiple of TSK)
//    A   — input  M×K matrix, column-major, cast to floatX*
//    Bt  — input  N×K matrix = B transposed, column-major, cast to floatX*
//    C   — output M×N matrix, column-major, plain float*
//
//  Launch parameters:
//    local  = { RTSM, RTSN }       e.g. { 16, 16 }  → 256 threads/work-group
//    global = { M/WPTM, N/WPTN }   e.g. { 256, 256 } for M=N=2048
// -----------------------------------------------------------------------------
__kernel __attribute__((reqd_work_group_size(RTSM, RTSN, 1)))
void myGEMM(const int M,
            const int N,
            const int K,
            const __global floatX* restrict A,    // M×K  col-major  (vector view)
            const __global floatX* restrict Bt,   // N×K  col-major  (vector view)
                  __global float*           C)    // M×N  col-major  (scalar write)
{
    // -------------------------------------------------------------------------
    //  Thread and work-group identifiers
    // -------------------------------------------------------------------------
    const int tidm = get_local_id(0);       // Local thread index, dim 0 (0..RTSM-1)
    const int tidn = get_local_id(1);       // Local thread index, dim 1 (0..RTSN-1)
    const int gm   = get_group_id(0) * TSM; // First M-row this work-group owns
    const int gn   = get_group_id(1) * TSN; // First N-col this work-group owns

    // -------------------------------------------------------------------------
    //  [K9] Double-buffered __local memory
    //
    //  Two buffer slots (index 0 and 1) allow us to prefetch tile t+1 into
    //  one slot while computing tile t from the other slot.
    //  Flattened to 2D because OpenCL does not support 3D __local arrays.
    //
    //  Asub[slot][k_local * TSM + m_local]   — TSK rows × TSM cols per slot
    //  Bsub[slot][k_local * TSN + n_local]   — TSK rows × TSN cols per slot
    //
    //  Total local memory used:
    //    2 * (TSK*TSM + TSK*TSN) * 4 bytes
    //    = 2 * (16*128 + 16*128) * 4 = 32 KB  (fits in 48 KB per SM)
    // -------------------------------------------------------------------------
    __local float Asub[2][TSK * TSM];
    __local float Bsub[2][TSK * TSN];

    // -------------------------------------------------------------------------
    //  [K6] Per-thread register accumulator
    //
    //  acc[wm][wn] accumulates the dot product for the output element at:
    //    global row = gm + tidm + wm * RTSM
    //    global col = gn + tidn + wn * RTSN
    //
    //  With WPTM=WPTN=8 this is 64 floats per thread, held entirely in
    //  registers — no __local or global reads during accumulation.
    // -------------------------------------------------------------------------
    float acc[WPTM][WPTN];
    for (int wm = 0; wm < WPTM; ++wm)
        for (int wn = 0; wn < WPTN; ++wn)
            acc[wm][wn] = 0.0f;

    // Total number of TSK-wide tiles that span the K dimension
    const int numTiles = K / TSK;

    // -------------------------------------------------------------------------
    //  [K9] Prefetch tile 0 into slot 0 BEFORE the main loop.
    //  This primes the pipeline: the very first compute iteration will
    //  find its data already waiting in slot 0.
    //  We need one barrier here because the main loop has no barrier
    //  between "use slot cur" and "start loading into slot nxt".
    // -------------------------------------------------------------------------
    LOAD_TILE_A(0, 0)
    LOAD_TILE_B(0, 0)
    barrier(CLK_LOCAL_MEM_FENCE);   // ensure tile 0 is fully visible to all threads

    // -------------------------------------------------------------------------
    //  Main tile loop  [K2, K6, K9, K10]
    //
    //  Each iteration:
    //    1. Start loading tile (t+1) into the OTHER buffer slot  [K9 prefetch]
    //    2. Compute tile t from the CURRENT slot                 [K6 reg block]
    //    3. Barrier — wait for the (t+1) loads to complete before
    //                 the next iteration uses that slot
    //
    //  Timeline (with double buffering):
    //
    //  t=0: [load tile 1 into slot 1]  overlapped with  [compute tile 0 from slot 0]
    //       barrier
    //  t=1: [load tile 2 into slot 0]  overlapped with  [compute tile 1 from slot 1]
    //       barrier
    //  ...
    //
    //  The barrier at the END (not the beginning) of the loop body ensures
    //  the prefetch loads finish before the next iteration reads them.
    // -------------------------------------------------------------------------
    for (int t = 0; t < numTiles; ++t) {

        // Which buffer slot holds tile t, and which will hold tile t+1?
        const int cur = t & 1;       // alternates 0, 1, 0, 1, ...
        const int nxt = cur ^ 1;     // the other slot:  1, 0, 1, 0, ...

        // [K9] Start prefetching tile t+1 into slot nxt
        // (guard with if so we don't issue out-of-bounds loads on the last tile)
        if (t + 1 < numTiles) {
            LOAD_TILE_A(t + 1, nxt)
            LOAD_TILE_B(t + 1, nxt)
        }

        // [K6] Compute using tile t which is already in slot cur
        COMPUTE_TILE(cur)

        // Wait for the prefetch loads (slot nxt) to be visible before the
        // next iteration's COMPUTE_TILE reads from that slot
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // -------------------------------------------------------------------------
    //  Write the WPTM × WPTN accumulated results back to global C  [K6]
    //
    //  Each of the 64 acc values maps to a unique (row, col) in C.
    //  Stride between consecutive wm-iterations: RTSM rows  (= 16)
    //  Stride between consecutive wn-iterations: RTSN cols  (= 16)
    //  This gives a regular but non-contiguous write pattern; the compiler
    //  can unroll both loops fully since WPTM and WPTN are compile-time constants.
    // -------------------------------------------------------------------------
    for (int wm = 0; wm < WPTM; ++wm) {
        const int globalRow = gm + tidm + wm * RTSM;
        for (int wn = 0; wn < WPTN; ++wn) {
            const int globalCol = gn + tidn + wn * RTSN;
            C[globalCol * M + globalRow] = acc[wm][wn];
        }
    }
}

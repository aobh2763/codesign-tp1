// ============================================================
// Tunable GEMM Kernel (matches provided Python host code)
// Requires B to be PRE-TRANSPOSED on host
// ============================================================

__kernel void myGEMM5(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C)
{
    // Thread identifiers
    const int row = get_local_id(0);                 // [0..TS)
    const int col = get_local_id(1);                 // [0..RTS)

    const int globalRow = TS * get_group_id(0) + row;
    const int globalCol = TS * get_group_id(1) + col;

    // Local memory (note padding to avoid bank conflicts)
    __local float Asub[TSDK][TS];
    __local float Bsub[TS][TSDK + 2];

    // Accumulators
    float acc[WPT];
    for (int w = 0; w < WPT; w++) {
        acc[w] = 0.0f;
    }

    // Number of tiles
    const int numTiles = K / TSDK;

    for (int t = 0; t < numTiles; t++) {

        // Load tiles into local memory
        for (int l = 0; l < LPT; l++) {

            const int tiledIndex = TSDK * t + col + l * RTS;

            // Bounds-safe indices
            int aRow = globalRow;
            int aCol = tiledIndex;

            int bRow = globalCol + l * RTS;
            int bCol = tiledIndex;

            // Load A (row-major)
            Asub[col + l * RTS][row] =
                A[aRow * K + aCol];

            // Load B (PRE-TRANSPOSED → now row-major access)
            Bsub[row][col + l * RTS] =
                B[bRow * K + bCol];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute
        for (int k = 0; k < TSDK; k++) {
            for (int w = 0; w < WPT; w++) {
                acc[w] +=
                    Asub[k][row] *
                    Bsub[col + w * RTS][k];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store results
    for (int w = 0; w < WPT; w++) {
        int outCol = globalCol + w * RTS;

        if (globalRow < M && outCol < N) {
            C[globalRow * N + outCol] = acc[w];
        }
    }
}
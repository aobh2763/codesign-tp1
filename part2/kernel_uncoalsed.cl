// Fully UNCOALESCED naive GEMM
__kernel void mmul(
        const int M,
        const int N,
        const int K,
        const __global float* A,
        const __global float* B,
        __global float* C)
{
    // Thread computes C[row,col]
    const int row = get_global_id(0);
    const int col = get_global_id(1);

    if (row >= M || col >= N)
        return;

    float acc = 0.0f;

    for (int k = 0; k < K; k++)
    {
        // Column-wise traversal (BAD)
        float a = A[k * M + row];   // stride access
        float b = B[col * K + k];   // stride access

        acc += a * b;
    }

    // Also uncoalesced write
    C[col * M + row] = acc;
}
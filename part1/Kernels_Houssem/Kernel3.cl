
#define blksz 16
#define WPT 4

__kernel void mmul(
                const unsigned int             N,
                __global const float* restrict A,
                __global const float* restrict B,
                __global       float* restrict C,
                __local        float* restrict Awrk,
                __local        float* restrict Bwrk)
{
    int kloc, Kblk;

    // Initialise the accumulation registers
    float Ctmp[WPT];
    for (int w=0; w<WPT; w++)
       Ctmp[w] = 0.0f;

    //  This work-item will compute element C(i,j)
    const int i = get_global_id(0)*WPT;
    const int j = get_global_id(1);

    // Element C(i,j) is in block C(Iblk,Jblk)
    const int Iblk = get_group_id(0);
    const int Jblk = get_group_id(1);

    // C(i,j) is element C(iloc, jloc) of block C(Iblk, Jblk)
    const int iloc = get_local_id(0);
    const int jloc = get_local_id(1);

    // The number of blocks are the same in each dimension
    const int Num_BLK = N/blksz;

    


    // C(Iblk,Jblk) = (sum over Kblk) A(Iblk,Kblk)*B(Kblk,Jblk)
    for (Kblk = 0;  Kblk<Num_BLK;  Kblk++)
    {
       // Load A(Iblk,Kblk) and B(Kblk,Jblk) into local memory.
       // Each work-item loads a single element of the two blocks
       // which are shared with the entire work-group.

       
       for (int w = 0; w < WPT; w++) {
        int colOffset = iloc + w * (blksz / WPT);

        int kA = Kblk * blksz + colOffset;
        Awrk[jloc*blksz + colOffset] = A[j * N + kA];

        int kB = Kblk * blksz + jloc;
        Bwrk[jloc*blksz + colOffset] = B[kB * N + (i + colOffset)];
        }

       barrier(CLK_LOCAL_MEM_FENCE);

       // Compute dot products over local blocks to find
       // the contribution to C(i,j) from this block
       #pragma unroll
       for (kloc=0; kloc<blksz; kloc++){
        for (int w = 0; w < WPT; w++) {
            Ctmp[w] += Awrk[jloc*blksz + kloc] * Bwrk[kloc*blksz + iloc + w * (blksz / WPT)];
        }
       }

       barrier(CLK_LOCAL_MEM_FENCE);
       
    }
 
    // update global C matrix 
    for (int w = 0; w < WPT; w++) {
        // For safe edgecase
        int col = i + w * (blksz / WPT);
        if (col < N) {
            C[j*N + col] = Ctmp[w];
        }
    }
}


#define blksz 16
#define WIDTH 4
#if WIDTH == 1
    typedef float floatX;
#elif WIDTH == 2
    typedef float2 floatX;
#elif WIDTH == 4
    typedef float4 floatX;
#endif

__kernel void mmul(
                const unsigned int             N,
                __global const float* restrict A,
                __global const float* restrict B,
                __global       floatX* restrict C,
                __local        float* restrict Awrk,
                __local        float* restrict Bwrk)
{


    // C(i,j) is element C(iloc, jloc) of block C(Iblk, Jblk)
    const int iloc = get_local_id(0);
    const int jloc = get_local_id(1);
    const int globalRow = (blksz/WIDTH)*get_group_id(0) + iloc; // 0..M/WIDTH
    const int globalCol = (blksz/WIDTH)*get_group_id(1) + jloc; // 0..N

    // Local memory to fit a tile of TS*TS elements of A and B
    __local floatX Asub[blksz][blksz/WIDTH];
    __local floatX Bsub[blksz][blksz/WIDTH];

    // Initialise the accumulation registers
    #if WIDTH == 1
        floatX acc = 0.0f;
    #elif WIDTH == 2
        floatX acc = { 0.0f, 0.0f };
    #elif WIDTH == 4
        floatX acc = { 0.0f, 0.0f, 0.0f, 0.0f };
    #endif

    // The number of blocks are the same in each dimension
    const int Num_BLK = N/blksz;

    // C(Iblk,Jblk) = (sum over Kblk) A(Iblk,Kblk)*B(Kblk,Jblk)
    for (int t=0; t<Num_BLK; t++) {
 
        // Load one tile of A and B into local memory
        const int tiledRow = (blksz/WIDTH)*t + iloc;
        const int tiledCol = blksz*t + jloc;
        Asub[jloc][iloc] = A[tiledCol*(N/WIDTH) + globalRow];
        Bsub[jloc][iloc] = B[globalCol*(N/WIDTH) + tiledRow];
       
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        floatX vecA, vecB;
        float valB;
        for (int k=0; k<blksz/WIDTH; k++) {
            vecB = Bsub[jloc][k];
            for (int w=0; w<WIDTH; w++) {
                vecA = Asub[WIDTH*k + w][iloc];
                #if WIDTH == 1
                    valB = vecB;
                    acc += vecA * valB;
                #elif WIDTH == 2
                    switch (w) {
                        case 0: valB = vecB.x; break;
                        case 1: valB = vecB.y; break;
                    }
                    acc.x += vecA.x * valB;
                    acc.y += vecA.y * valB;
                #elif WIDTH == 4
                    switch (w) {
                        case 0: valB = vecB.x; break;
                        case 1: valB = vecB.y; break;
                        case 2: valB = vecB.z; break;
                        case 3: valB = vecB.w; break;
                    }
                    acc.x += vecA.x * valB;
                    acc.y += vecA.y * valB;
                    acc.z += vecA.z * valB;
                    acc.w += vecA.w * valB;
               #endif
            }
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final results in C
    C[globalCol * (N/WIDTH) + globalRow] = acc;
}

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

#define WIDTH 2

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

#define THREADSX 8
#define THREADSY 8
#define RX 8
#define RY 4
#define RK (RY)

// Typedefs for clBlas-mimic kernel (myGEMM11)
#if RX == 2
    typedef float2 floatA;
    typedef float2 floatC;
#elif RX == 4
    typedef float4 floatA;
    typedef float4 floatC;
#elif RX == 8
    typedef float8 floatA;
    typedef float8 floatC;
#endif

#if RK == 2
    typedef float2 floatB;
#elif RK == 4
    typedef float4 floatB;
#elif RK == 8
    typedef float8 floatB;
#endif

// Mimic clBlas (4x8 register tiling with vector data-types)
__kernel void myGEMM11(const int M, const int N, const int K,
                       const __global floatA* restrict A,
                       const __global floatB* restrict B,
                       __global floatC* C) {
    
    // Allocate register space
    float aReg[RK][RX];
    float bReg[RY][RK];
    float acc[RY][RX];

    // Initialise the accumulation registers
    #pragma unroll
    for (int y=0; y<RY; y++) {
        for (int x=0; x<RX; x++) {
            acc[y][x] = 0.0;
        }
    }

    // Loop over all tiles
    const int numTiles = K/RK;
    for (int t=0; t<numTiles; t++) {

        // Load a tile of A and B into register memory
        #pragma unroll
        for (int y=0; y<RY; y++) {

            // Load the data
            floatA aVec = A[(RK*t + y)*(M/RX) + get_global_id(0)];
            floatB bVec = B[(RY*get_global_id(1) + y)*numTiles + t];

            // Store the vector of A into registers
            #if RX == 2
                aReg[y][0] = aVec.x;
                aReg[y][1] = aVec.y;
            #elif RX == 4
                aReg[y][0] = aVec.x;
                aReg[y][1] = aVec.y;
                aReg[y][2] = aVec.z;
                aReg[y][3] = aVec.w;
            #elif RX == 8
                aReg[y][0] = aVec.s0;
                aReg[y][1] = aVec.s1;
                aReg[y][2] = aVec.s2;
                aReg[y][3] = aVec.s3;
                aReg[y][4] = aVec.s4;
                aReg[y][5] = aVec.s5;
                aReg[y][6] = aVec.s6;
                aReg[y][7] = aVec.s7;
            #endif

            // Store the vector of B into registers
            #if RK == 2
                bReg[y][0] = bVec.x;
                bReg[y][1] = bVec.y;
            #elif RK == 4
                bReg[y][0] = bVec.x;
                bReg[y][1] = bVec.y;
                bReg[y][2] = bVec.z;
                bReg[y][3] = bVec.w;
            #elif RK == 8
                bReg[y][0] = bVec.s0;
                bReg[y][1] = bVec.s1;
                bReg[y][2] = bVec.s2;
                bReg[y][3] = bVec.s3;
                bReg[y][4] = bVec.s4;
                bReg[y][5] = bVec.s5;
                bReg[y][6] = bVec.s6;
                bReg[y][7] = bVec.s7;
            #endif
        }

        // Perform the computations
        #pragma unroll
        for (int k=0; k<RK; k++) {
            #pragma unroll
            for (int y=0; y<RY; y++) {
                #pragma unroll
                for (int x=0; x<RX; x++) {
                    acc[y][x] += aReg[k][x] * bReg[y][k];
                }
            }
        }
    }

    // Store the final results in C
    #pragma unroll
    for (int y=0; y<RY; y++) {
        floatC accVec;
        #if RX == 2
            accVec.x = acc[y][0];
            accVec.y = acc[y][1];
        #elif RX == 4
            accVec.x = acc[y][0];
            accVec.y = acc[y][1];
            accVec.z = acc[y][2];
            accVec.w = acc[y][3];
        #elif RX == 8
            accVec.s0 = acc[y][0];
            accVec.s1 = acc[y][1];
            accVec.s2 = acc[y][2];
            accVec.s3 = acc[y][3];
            accVec.s4 = acc[y][4];
            accVec.s5 = acc[y][5];
            accVec.s6 = acc[y][6];
            accVec.s7 = acc[y][7];
        #endif
        C[(y + RY*get_global_id(1)) * (M/RX) + get_global_id(0)] = accVec;
    }
}
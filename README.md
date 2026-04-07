# Kernel Optimisation for Matrix Multiplication

## Kernel 1 : Naive approach
### Best : 
localsize = 32
1.4306464195251465 seconds at 384270.91864562343 MFLOPS

## Kernel 2 : Tiling in the local memory
### Best :
TS = 32
1.0610089302062988 seconds at 518144.37959641626 MFLOPS

## Kernel 3 : More work per thread
### Best
TS =  32  and WPT =  16
0.41141748428344727 seconds at 1336248.0567530866 MFLOPS

## Kernel 4 : Wider data types
### Best
TS =  32  and WIDTH =  4
0.38169121742248535 seconds at 1440315.5975147518 MFLOPS

## Kernel 5 : Transposed input matrix and rectangular tiles
### Best
TBD

## Kernel 6 : 2D register blocking
### Best
TSM =  128  TSN =  128  TSK =  16  WPTM =  16  WPTN =  8
0.21154117584228516 seconds at 2598812.3196302513 MFLOPS

## Kernel 7 : Wider loads with register blocking
### Best
TSM =  64  TSN =  128  TSK =  16  WPTM =  8  WPTN =  16  WIDTH =  4
0.19498491287231445 seconds at 2819478.7267874754 MFLOPS

## Kernel 8 : CUDA and Kepler-specific optimisations
TBD

## Kernel 9 : Pre-fetching
TSM =  128  TSN =  128  TSK =  16  WPTM =  8  WPTN =  8  WITDH =  4
0.93320631980896 seconds at 589104.2551024971 MFLOPS

## Kernel 10 : Incomplete tiles and arbitrary matrix sizes
TSM =  128  TSN =  128  TSK =  16  WPTM =  16  WPTN =  8  WIDTH = 2
0.14084863662719727 seconds at 3903167.4501977004 MFLOPS

## Kernel 11 : Mystery kernel
THREADSX =  16  THREADSY =  4  RX =  4  RY =  8
0.16794228553771973 seconds at 3273480.6015518066 MFLOPS

## Ultimate Kernel : K2 + K3 + K4 + K6 + K7 + K9 + K10
0.08525609970092773 seconds at 6448287.170261176 MFLOPS
# Kernel Optimisation for Matrix Multiplication (Coalsed)

## Kernel 1 : Naive approach
### Best (Regular)
localsize = 32 <br />
1.4306464195251465 seconds at 384270.91864562343 MFLOPS
### Best (Optimised GPU)
localsize = 32 <br />
1.0054569244384766 seconds at 546772.1197454833 MFLOPS

## Kernel 2 : Tiling in the local memory
### Best (Regular)
TS = 32 <br />
1.0610089302062988 seconds at 518144.37959641626 MFLOPS
### Best (Optimised GPU)
TS = 32 <br />
0.8868663311004639 seconds at 619885.7647531147 MFLOPS

## Kernel 3 : More work per thread
### Best (Regular)
TS =  32  and WPT =  16 <br />
0.41141748428344727 seconds at 1336248.0567530866 MFLOPS
### Best (Optimised GPU)
TS =  32  and WPT =  16 <br />
0.27342700958251953 seconds at 2010612.6849991577 MFLOPS

## Kernel 4 : Wider data types
### Best (Regular)
TS =  32  and WIDTH =  4 <br />
0.38169121742248535 seconds at 1440315.5975147518 MFLOPS
### Best (Optimised GPU)
TS =  32  and WIDTH =  4 <br />
0.3903772830963135 seconds at 1408267.9440964418 MFLOPS

## Kernel 5 : Transposed input matrix and rectangular tiles
### Best (Regular)
TS =  64  WPT =  8  TSDK =  16 <br />
0.7188866138458252 seconds at 764732.3003372859 MFLOPS
### Best (Optimised GPU)
TS =  64  WPT =  8  TSDK =  16 <br />
0.5890092849731445 seconds at 933356.7872585672 MFLOPS

## Kernel 6 : 2D register blocking
### Best (Regular)
TSM =  128  TSN =  128  TSK =  16  WPTM =  16  WPTN =  8 <br />
0.21154117584228516 seconds at 2598812.3196302513 MFLOPS
### Best (Optimised GPU)
TSM =  64  TSN =  128  TSK =  16  WPTM =  16  WPTN =  8 <br />
0.15489435195922852 seconds at 3549230.8591904463 MFLOPS

## Kernel 7 : Wider loads with register blocking
TSM =  64  TSN =  128  TSK =  16  WPTM =  8  WPTN =  16  WIDTH =  4 <br />
### Best (Regular)
0.19498491287231445 seconds at 2819478.7267874754 MFLOPS
### Best (Optimised GPU)
TSM =  64  TSN =  128  TSK =  16  WPTM =  16  WPTN =  8  WIDTH =  4 <br />
0.14102578163146973 seconds at 3898264.611818487 MFLOPS

## Kernel 8 : CUDA and Kepler-specific optimisations
### Best (Regular)
TBD
### Best (Optimised GPU)
TBD

## Kernel 9 : Pre-fetching
### Best (Regular)
TSM =  128  TSN =  128  TSK =  16  WPTM =  8  WPTN =  8  WITDH =  4 <br />
0.93320631980896 seconds at 589104.2551024971 MFLOPS
### Best (Optimised GPU)
TSM =  128  TSN =  128  TSK =  16  WPTM =  16  WPTN =  8  WIDTH =  4 <br />
0.14127492904663086 seconds at 3891389.7716879486 MFLOPS

## Kernel 10 : Incomplete tiles and arbitrary matrix sizes
### Best (Regular)
TSM =  128  TSN =  128  TSK =  16  WPTM =  16  WPTN =  8  WIDTH =  2 <br />
0.14084863662719727 seconds at 3903167.4501977004 MFLOPS
### Best (Optimised GPU)
TSM =  128  TSN =  128  TSK =  16  WPTM =  16  WPTN =  8  WIDTH =  2 <br />
0.14129257202148438 seconds at 3890903.860143521 MFLOPS

## Kernel 11 : Mystery kernel
### Best (Regular)
THREADSX =  16  THREADSY =  4  RX =  4  RY =  8 <br />
0.16794228553771973 seconds at 3273480.6015518066 MFLOPS
### Best (Optimised GPU)
THREADSX =  16  THREADSY =  4  RX =  4  RY =  8 <br />
0.16967391967773438 seconds at 3240072.5752794775 MFLOPS

## Ultimate Kernel : K2 + K3 + K4 + K6 + K7 + K9 + K10
### Best (Regular)
0.08525609970092773 seconds at 6448287.170261176 MFLOPS
### Best (Optimised GPU)
0.0844123363494873 seconds at 6512742.540520134 MFLOPS
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
### Best (Regular)
TSM =  64  TSN =  128  TSK =  16  WPTM =  8  WPTN =  16  WIDTH =  4 <br />
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

# Final measurements
For perfectly optimised NVIDIA GPU RTX 3050:
- Plugged in
- High-performance NVIDIA processor
- Power management mode: Prefer max performance
- Power mode : Best performance when plugged in
- OMEN gaming hub : Performance mode
- N = 2048
- COUNT = 256 for more precision

## Kernel 1 : Naive approach
### Best
localsize = 32 <br />
9.505536556243896 seconds at 462682.61502976995 MFLOPS

## Kernel 2 : Tiling in the local memory
### Best
TS = 16 <br />
9.380849838256836 seconds at 468832.417844272 MFLOPS

## Kernel 3 : More work per thread
### Best
TS = 32 WPT = 16 <br />
4.803513050079346 seconds at 915589.5831346501 MFLOPS

## Kernel 4 : Wider data types
### Best
TS = 32 WIDTH = 8 <br />
5.251865386962891 seconds at 837425.5977736233 MFLOPS

## Kernel 5 : Transposed input matrix and rectangular tiles
### Best
TS = 64 WPT = 8 TSDK = 64 <br />
5.421699285507202 seconds at 811193.3693667706 MFLOPS

## Kernel 6 : 2D register blocking
### Best
TSM = 64 TSN = 128 TSK = 16 WPTM = 16 WPTN = 8 <br />
1.3257341384887695 seconds at 3317442.3011520393 MFLOPS

## Kernel 7 : Wider loads with register blocking
### Best
TSM = 64 TSN = 64 TSK = 16 WPTM = 16 WPTN = 8 <br />
1.1861083507537842 seconds at 3707963.5332716405 MFLOPS

## Kernel 8 : CUDA and Kepler-specific optimisations
### Best
TBD

## Kernel 9 : Pre-fetching
### Best
TSM = 128 TSN = 128 TSK = 16 WPTM = 16 WPTN = 8 <br />
1.1688785552978516 seconds at 3762620.5829255697 MFLOPS

## Kernel 10 : Incomplete tiles and arbitrary matrix sizes
### Best
TSM = 128 TSN = 128 TSK = 16 WPTM = 16 WPTN = 8 <br />
1.1455025672912598 seconds at 3839403.451974749 MFLOPS

## Kernel 11 : Mystery kernel
### Best
THREADSX = 8 THREADSY = 8 RX = 4 RY = 8 <br />
1.4211819171905518 seconds at 3094640.0723970872 MFLOPS

## Ultimate Kernel : K2 + K3 + K4 + K6 + K7 + K9 + K10
### Best (2048x and 8192x)
0.9831287860870361 seconds at 4473520.227811377 MFLOPS
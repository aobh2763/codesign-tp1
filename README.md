# Kernel Optimisation for Matrix Multiplication (Coalsed)

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
9.741274929046630 seconds at 439481.48102976995 MFLOPS

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

# Distributing Workload Between GPUs
## Intel GPU : Ultimate Kernel
702.9331228733063 seconds at 50053.65509739037 MFLOPS

# NVIDIA GPU : Naive Uncoalsed Kernel
32.75971984863281 seconds at 537006.6082891178 MFLOPS

# Dual GPU Matrix Multiplication
19.049 seconds at 577.2 GFLOPS
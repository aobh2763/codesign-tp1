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
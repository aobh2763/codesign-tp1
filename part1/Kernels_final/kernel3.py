from helper import *
from definitions import *

import pyopencl as cl
import numpy
from time import time
from time import sleep

N = 2048

# Number of elements in the matrix
size = N * N

#--------------------------------------------------------------------------------
# CHOOSE KERNEL TO EXECUTE (0: i=dim(0),j=dim(1) ; 1:i=dim(1), j=dim(0)
#--------------------------------------------------------------------------
kernel_name="part1/Kernels_final/kernel3.cl"


# Set up OpenCL
context = cl.create_some_context()
queue = cl.CommandQueue(context)

# Reset host buffers - just to play it safe
h_A = numpy.empty(size).astype(numpy.float32)
h_A.fill(AVAL)
h_B = numpy.empty(size).astype(numpy.float32)
h_B.fill(BVAL)
h_C = numpy.empty(size).astype(numpy.float32)

# Create OpenCL buffers
d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_A)
d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_B)
d_c = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=h_C.nbytes)


#--------------------------------------------------------------------------------
# OpenCL matrix multiplication ... Naive: Each WI computes one element
# C_elemnt.cl : i= get_global_id(0) - j=get_global_id(1)
#--------------------------------------------------------------------------------
bestmflops = 0.0
bestcouple = (0, 0)
for TS in [8, 16, 32]:
    for WPT in [1, 2, 4, 8, 16, 32]:
        if (WPT > TS):
            break
        if (TS//WPT > 32):
            continue
        
        kernelsource = open(kernel_name).read()
        program = cl.Program(context, kernelsource).build(
            options=[f"-DTS={TS}", f"-DWPT={WPT}"]
        )
        mmul = program.mmul
        mmul.set_scalar_arg_dtypes([numpy.int32, numpy.int32, numpy.int32, None, None, None])

        # Do the multiplication COUNT times


        print("Starting", COUNT , " OpenCL Matrix Multiplications \n")
        print("for TS = ", TS, " and WPT = ", WPT, "\n")
        start_time = time()

        for i in range(COUNT):    
            h_C.fill(0.0)
            try: 
                # Work-group computes a block of C. This size is also set
                # in a #define inside the kernel function. Note this blocksize
                # must evenly divide the matrix order

                mmul(queue, (N,N//WPT), (TS,TS//WPT), N, N, N, d_a, d_b, d_c)
                
                #mmul(queue, (N,N), (localsize,localsize), numpy.int32 (N), d_a, d_b, d_c,d_Awrk, d_Bwrk)
                queue.finish()
            except:
                print (" ===  Error ===\n")    

        run_time = time() - start_time
            
        print ("mmum queued")

        #reading the result h_C
        cl.enqueue_copy(queue, h_C, d_c)

        #cl.enqueue_read_buffer(queue, d_c, h_C).wait()
        print (h_C[0])
        
        curr = results (N, COUNT, run_time)
        if curr > bestmflops:
            bestmflops = curr
            bestcouple = (TS, WPT)
            
print ("Best performance at TS = ", bestcouple[0], " and WPT = ", bestcouple[1], " with ", bestmflops, " MFLOPS\n")
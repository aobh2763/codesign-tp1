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
kernel_name="part1/Kernels_final/kernel5.cl"


# Set up OpenCL
context = cl.create_some_context()
queue = cl.CommandQueue(context)

# Reset host buffers - just to play it safe
h_A = numpy.empty(size).astype(numpy.float32)
h_A.fill(AVAL)
h_B = numpy.empty(size).astype(numpy.float32)
h_B.fill(BVAL)
h_Bt = numpy.empty(size).astype(numpy.float32)
h_Bt.fill(0.0)
h_C = numpy.empty(size).astype(numpy.float32)

# Create OpenCL buffers
d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_A)
d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_B)
d_c = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=h_C.nbytes)

d_bt = cl.Buffer(context,  cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_Bt)


#--------------------------------------------------------------------------------
# OpenCL matrix multiplication ... Naive: Each WI computes one element
# C_elemnt.cl : i= get_global_id(0) - j=get_global_id(1)
#--------------------------------------------------------------------------------
bestmflops = 0.0
besttuple = (0, 0, 0, 0, 0)
MAX_WG_SIZE = 1024
LOCAL_MEM_BYTES = 48*1024
WARP_SIZE = 32

for TSM in [32, 64, 128]:
    for TSN in [32, 64, 128]:
        for TSK in [16, 32, 64]:
            for WPTM in [1, 2, 4, 8]:
                if TSM % WPTM != 0:
                    continue
                for WPTN in [1, 2, 4, 8]:
                    if TSN % WPTN != 0:
                        continue

                    RTSM = TSM // WPTM
                    RTSN = TSN // WPTN

                    # --- work-group size ---
                    if RTSM*RTSN > MAX_WG_SIZE:
                        continue

                    # --- local memory check ---
                    local_mem = 4*(TSM*TSK + TSN*TSK)
                    if local_mem > LOCAL_MEM_BYTES:
                        continue

                    # --- integer loads per thread ---
                    threads = RTSM * RTSN
                    if (TSK*TSM) % threads != 0:
                        continue

                    LPT = (TSK*TSM)//threads
                    
                    kernelsource = open(kernel_name).read()
                    program = cl.Program(context, kernelsource).build(
                        options=[f"-DTS={TSM}", f"-DWIDTH={TSN}", f"-DTSK={TSK}", f"-DWPTM={WPTM}", f"-DWPTN={WPTN}"]
                    )
                    transpose = program.transpose
                    transpose.set_scalar_arg_dtypes([numpy.int32, numpy.int32, None, None])
                    
                    mmul = program.mmul
                    mmul.set_scalar_arg_dtypes([numpy.int32, numpy.int32, numpy.int32, None, None, None])

                    # Do the multiplication COUNT times
                    
                    
                    num_groups_M = numpy.ceil(N / TSM)
                    num_groups_N = numpy.ceil(N / TSN)


                    print("Starting", COUNT , " OpenCL Matrix Multiplications \n")
                    print("for TSM = ", TSM, " TSN = ", TSN, " TSK = ", TSK, " WPTM = ", WPTM, " WPTN = ", WPTN, "\n")
                    start_time = time()

                    for i in range(COUNT):    
                        h_C.fill(0.0)
                        try: 
                            # Work-group computes a block of C. This size is also set
                            # in a #define inside the kernel function. Note this blocksize
                            # must evenly divide the matrix order
                            
                            transpose(queue, (N,N), (32,32), N, N, d_b, d_bt)

                            mmul(queue, (num_groups_M*RTSM, num_groups_N*RTSN), (RTSM, RTSN), N, N, N, d_a, d_bt, d_c)
                            
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
                        besttuple = (TSM, TSN, TSK, WPTM, WPTN)
                        
            print ("Best performance at TSM = ", besttuple[0], " TSN = ", besttuple[1], " TSK = ", besttuple[2], " WPTM = ", besttuple[3], " WPTN = ", besttuple[4], " with ", bestmflops, " MFLOPS\n")
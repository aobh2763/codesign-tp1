import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

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
kernel_name="part1/Kernels_final/kernel11/kernel11.cl"


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
bestsecs = numpy.inf
besttuple = (0, 0, 0, 0, 0)
MAX_WG_SIZE = 1024
LOCAL_MEM_BYTES = 48*1024
WARP_SIZE = 32

for RX in [2, 4, 8]:           # Number of columns per thread (X dimension)
    for RY in [2, 4, 8]:        # Number of rows per thread (Y dimension)
        RK = RY
        for THREADSX in [4, 8, 16]:   # Work-group X dimension
            for THREADSY in [4, 8, 16]:  # Work-group Y dimension
                
                # Constraints
                threads_per_wg = THREADSX * THREADSY
                if threads_per_wg > MAX_WG_SIZE:
                    continue
                
                # Each thread computes RX * RY elements
                # Total work-group computes (THREADSX*RX) x (THREADSY*RY) elements
                tileM = THREADSY * RY
                tileN = THREADSX * RX
                
                # Global size must be multiple of tile size
                if N % tileM != 0 or N % tileN != 0:
                    continue
                
                # Check register pressure (approx 4-8 bytes per register)
                # Each thread needs: RX*RY (acc) + RK*RX (Areg) + RY*RK (Breg)
                registers_per_thread = (RX * RY) + (RK * RX) + (RY * RK)
                if registers_per_thread > 256:  # Typical GPU limit
                    continue
                
                kernelsource = open(kernel_name).read()
                program = cl.Program(context, kernelsource).build(
                    options=[f"-DTHREADSX={THREADSX}", f"-DTHREADSY={THREADSY}", f"-DRX={RX}", f"-DRY={RY}"]
                )
                
                mmul = program.mmul
                mmul.set_scalar_arg_dtypes([numpy.int32, numpy.int32, numpy.int32, None, None, None])

                # Do the multiplication COUNT times
                global_size_x = (N + RX - 1) // RX  # Number of floatX elements per row
                global_size_y = (N + RY - 1) // RY  # Number of RY-row blocks

                print("Starting", COUNT , " OpenCL Matrix Multiplications \n")
                print("for THREADSX =", THREADSX, " THREADSY =", THREADSY, " RX =", RX, " RY =", RY)
                start_time = time()

                for i in range(COUNT):    
                    h_C.fill(0.0)
                    try: 
                        # Work-group computes a block of C. This size is also set
                        # in a #define inside the kernel function. Note this blocksize
                        # must evenly divide the matrix order

                        mmul(queue, (global_size_x, global_size_y), (THREADSX, THREADSY), N, N, N, d_a, d_b, d_c)
                        
                        #mmul(queue, (N,N), (localsize,localsize), numpy.int32 (N), d_a, d_b, d_c,d_Awrk, d_Bwrk)
                        queue.finish()
                    except(Exception) as e:
                        print (" ===  Error ===\n")    
                        print(e)

                run_time = time() - start_time
                    
                print ("mmum queued")

                #reading the result h_C
                cl.enqueue_copy(queue, h_C, d_c).wait()

                #cl.enqueue_read_buffer(queue, d_c, h_C).wait()
                print (h_C[0])
                
                curr, currs = results (N, COUNT, run_time)
                if curr > bestmflops:
                    bestmflops = curr
                    bestsecs = currs
                    besttuple = (THREADSX, THREADSY, RX, RY)
                        
print ("Best performance at THREADSX = ", besttuple[0], " THREADSY = ", besttuple[1], " RX = ", besttuple[2], " RY = ", besttuple[3], " with ", bestmflops, " MFLOPS\n")
print (bestsecs, "seconds at", bestmflops, "MFLOPS")
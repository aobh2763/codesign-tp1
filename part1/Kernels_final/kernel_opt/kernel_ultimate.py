import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from helper import *
from definitions import *

import numpy

import pyopencl as cl

from time import time
from time import sleep

# A[N][N], B[N][N], C[N][N]
N = 2048

# Number of elements in the matrix
size = N * N
#true value
cval = float(N) * AVAL * BVAL


#--------------------------------------------------------------------------------
# CHOOSE KERNEL TO EXECUTE (0: i=dim(0),j=dim(1) ; 1:i=dim(1), j=dim(0)
#--------------------------------------------------------------------------------
print ("Matrix multiplication",N,"*",N," repeated ",COUNT," times, j=0, i=1 :\n")
kernel_name="part1/Kernels_final/kernel_opt/kernel_ultimate.cl"

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
kernelsource = open(kernel_name).read()
program = cl.Program(context, kernelsource).build()
mmul = program.mmul
mmul.set_scalar_arg_dtypes([numpy.int32, numpy.int32, numpy.int32, None, None, None])


start_time = time()

for i in range(COUNT):
    try:
        mmul(queue, (128,128), (8,16), numpy.int32 (N), numpy.int32 (N), numpy.int32 (N), d_a, d_b, d_c)
        queue.finish()
    except:
        print (" ===  Error for localsize =", (8,16), "===\n")    

run_time = time() - start_time

print ("\n End of", COUNT, "Matrix Multiplications\n")

results (N, COUNT , run_time)

#reading the result h_C
cl.enqueue_copy(queue, h_C, d_c)
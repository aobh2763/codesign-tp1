from helper import *
from definitions import *

import numpy
import pyopencl as cl
from time import time

N    = 8192
size = N * N
cval = float(N) * AVAL * BVAL

alpha = 0.917
rows_nv = int(round(alpha * N))
rows_ig = N - rows_nv

# Round to fit the workgroup sizes of the kernels (16 for NVIDIA, 128 for iGPU)
rows_nv = (rows_nv // 16) * 16
rows_ig = (rows_ig // 128) * 128
print(f"Split: NVIDIA rows 0..{rows_nv-1} ({rows_nv}), iGPU rows {rows_nv}..{N-1} ({rows_ig})")

h_A = numpy.empty(size).astype(numpy.float32)
h_A.fill(AVAL)
h_B = numpy.empty(size).astype(numpy.float32)
h_B.fill(BVAL)
h_C = numpy.empty(size).astype(numpy.float32)

all_platforms = cl.get_platforms()
all_gpus = [d for p in all_platforms for d in p.get_devices(cl.device_type.GPU)]

nvidia_dev = next(d for d in all_gpus if 'NVIDIA' in d.name or 'GeForce' in d.name)
igpu_dev   = next(d for d in all_gpus if d is not nvidia_dev)

print(f"\nDevice 1 (NVIDIA) : {nvidia_dev.name}")
print(f"Device 2 (iGPU)   : {igpu_dev.name}\n")

ctx_nv   = cl.Context([nvidia_dev])
ctx_ig   = cl.Context([igpu_dev])
queue_nv = cl.CommandQueue(ctx_nv)
queue_ig = cl.CommandQueue(ctx_ig)

d_a_nv = cl.Buffer(
    ctx_nv,
    cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
    hostbuf=h_A[:rows_nv*N]
)
d_b_nv = cl.Buffer(ctx_nv, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_B)
d_c_nv = cl.Buffer(ctx_nv, cl.mem_flags.WRITE_ONLY, size=rows_nv * N * h_C.itemsize)

d_a_ig = cl.Buffer(
    ctx_ig,
    cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
    hostbuf=h_A[rows_nv*N:]
)
d_b_ig = cl.Buffer(ctx_ig, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_B)
d_c_ig = cl.Buffer(ctx_ig, cl.mem_flags.WRITE_ONLY, size=rows_ig * N * h_C.itemsize)

src_nv = open("part2/kernel_uncoalsed.cl").read()
src_ig = open("part2/kernel_ultimate.cl").read()

prog_nv = cl.Program(ctx_nv, src_nv).build()
prog_ig = cl.Program(ctx_ig, src_ig).build()

mmul_nv = prog_nv.mmul
mmul_ig = prog_ig.mmul

mmul_nv.set_scalar_arg_dtypes([numpy.int32, numpy.int32, numpy.int32, None, None, None])
mmul_ig.set_scalar_arg_dtypes([numpy.int32, numpy.int32, numpy.int32, None, None, None])

print(f"Starting {COUNT} dual-GPU matrix multiplications\n")
start_time = time()

for i in range(COUNT):
    mmul_nv(queue_nv, (rows_nv, N), (16, 16),
               numpy.int32(rows_nv), numpy.int32(N), numpy.int32(N), d_a_nv, d_b_nv, d_c_nv)

    mmul_ig(queue_ig, (rows_ig, N), (8, 16),
               numpy.int32(rows_ig), numpy.int32(N), numpy.int32(N), d_a_ig, d_b_ig, d_c_ig)

    # Wait for both to finish before next iteration
    queue_nv.flush()
    queue_ig.flush()
    
    queue_nv.finish()
    queue_ig.finish()

run_time = time() - start_time

print(f"End of {COUNT} multiplications\n")
results(N, COUNT, run_time)

h_C_nv = numpy.empty(rows_nv * N, dtype=numpy.float32)
h_C_ig = numpy.empty(rows_ig * N, dtype=numpy.float32)

cl.enqueue_copy(queue_nv, h_C_nv, d_c_nv)
cl.enqueue_copy(queue_ig, h_C_ig, d_c_ig)

queue_nv.finish()
queue_ig.finish()

# Paste slices back into h_C
h_C[:rows_nv * N] = h_C_nv
h_C[rows_nv * N:] = h_C_ig

total_flops  = 2.0 * N**3
elapsed_avg  = run_time / COUNT
gflops_dual  = total_flops / elapsed_avg / 1e9

print(f"\nDual-GPU performance : {gflops_dual:.1f} GFLOPS")
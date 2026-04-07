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
besttuple  = (0, 0, 0)

MAX_WG_SIZE      = 1024
LOCAL_MEM_BYTES  = 48 * 1024

for TS in [32]:
    for WPT in [8]:

        # --- C1: WPT must divide TS (RTS must be a whole number ≥ 1) ---
        if TS % WPT != 0:
            continue
        RTS = TS // WPT

        # --- C2: work-group size (RTS × TS) must not exceed hardware limit ---
        if RTS * TS > MAX_WG_SIZE:
            continue

        for TSDK in [16]:

            # --- C3: LPT must be a whole number ---
            if (TSDK * WPT) % TS != 0:
                continue
            LPT = (TSDK * WPT) // TS

            # --- C4: local memory ---
            # Asub[TSDK][TS]  →  TSDK * TS  floats
            # Bsub[TS][TSDK+2] →  TS * (TSDK + 2) floats
            local_mem_bytes = 4 * (TSDK * TS + TS * (TSDK + 2))
            if local_mem_bytes > LOCAL_MEM_BYTES:
                continue

            # --- C5: N must be divisible by TS (tile rows/cols divide evenly) ---
            if N % TS != 0:
                continue

            # --- C6: N must be divisible by TSDK (K-loop tiles divide evenly) ---
            if N % TSDK != 0:
                continue

            # ── all constraints passed — safe to compile and run ──
            kernelsource = open(kernel_name).read()
            program = cl.Program(context, kernelsource).build(
                options=[f"-DTS={TS}", f"-DWPT={WPT}", f"-DTSDK={TSDK}"]
            )
            mmul = program.mmul
            mmul.set_scalar_arg_dtypes(
                [numpy.int32, numpy.int32, numpy.int32, None, None, None]
            )

            # global size matches the local tile dimensions exactly
            global_rows = RTS * (N // RTS)   # always exact because N%TS==0 and RTS|TS
            global_cols = TS  * (N // TS)

            print(f"Starting {COUNT} runs | TS={TS} WPT={WPT} TSDK={TSDK} "
                  f"RTS={RTS} LPT={LPT} "
                  f"local_mem={local_mem_bytes}B "
                  f"wg=({RTS}×{TS}={RTS*TS})")

            start_time = time()
            for i in range(COUNT):
                cl.enqueue_fill_buffer(
                    queue, d_c, numpy.float32(0.0), 0, h_C.nbytes
                )
                mmul(queue, (global_rows, global_cols), (RTS, TS),
                     N, N, N, d_a, d_b, d_c)
                queue.finish()
            run_time = time() - start_time

            cl.enqueue_copy(queue, h_C, d_c)
            print(f"  h_C[0] = {h_C[0]}")

            curr = results(N, COUNT, run_time)
            if curr > bestmflops:
                bestmflops = curr
                besttuple  = (TS, WPT, TSDK)

print(f"\nBest: TS={besttuple[0]} WPT={besttuple[1]} TSDK={besttuple[2]} "
      f"→ {bestmflops:.1f} MFLOPS")
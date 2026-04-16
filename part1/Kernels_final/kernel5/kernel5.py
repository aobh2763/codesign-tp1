import sys
import os

import csv
results_log = []

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from helper import *
from definitions import *

import pyopencl as cl
import numpy
from time import time

N    = 2048
size = N * N

kernel_name = "part1/Kernels_final/kernel5/kernel5.cl"

# ── OpenCL setup ────────────────────────────────────────────────
context = cl.create_some_context()
queue   = cl.CommandQueue(context)

# ── Host buffers ─────────────────────────────────────────────────
h_A  = numpy.empty(size, dtype=numpy.float32); h_A.fill(AVAL)
h_B  = numpy.empty(size, dtype=numpy.float32); h_B.fill(BVAL)
h_Bt = numpy.zeros(size, dtype=numpy.float32)
h_C  = numpy.empty(size, dtype=numpy.float32)

# ── Device buffers ───────────────────────────────────────────────
d_a  = cl.Buffer(context, cl.mem_flags.READ_ONLY  | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_A)
d_b  = cl.Buffer(context, cl.mem_flags.READ_ONLY  | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_B)
d_bt = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_Bt)
d_c  = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=h_C.nbytes)

# ── Transpose kernel uses fixed 16x16 local size (hardcoded in kernel) ──
TRANS_LOCAL = 16                           # must match TRANSPOSEX/TRANSPOSEY
trans_gx = int(numpy.ceil(N / TRANS_LOCAL)) * TRANS_LOCAL
trans_gy = int(numpy.ceil(N / TRANS_LOCAL)) * TRANS_LOCAL

# ── Search constraints ───────────────────────────────────────────
MAX_WG_SIZE     = 1024
LOCAL_MEM_BYTES = 48 * 1024

bestmflops = 0.0
bestsecs   = numpy.inf
besttuple  = (0, 0, 0)          # (TS, WPT, TSDK)

# Kernel5 uses a single TS for both M and N tile dimensions,
# and a separate TSDK for the K tile. WPT coarsens the column
# dimension only (WPTM is implicitly 1).
for TS in [32, 64, 128]:
    for TSDK in [16, 32, 64]:
        for WPT in [1, 2, 4, 8]:

            # TS must be divisible by WPT
            if TS % WPT != 0:
                continue

            RTS = TS // WPT           # reduced tile size (cols per thread)

            # work-group size: TS rows × RTS cols
            if TS * RTS > MAX_WG_SIZE:
                continue

            # LPT = (TSDK * WPT) / TS  must be a positive integer
            if (TSDK * WPT) % TS != 0:
                continue
            LPT = (TSDK * WPT) // TS
            if LPT < 1:
                continue

            # local memory: Asub[TSDK][TS] + Bsub[TS][TSDK+2] floats
            local_mem = 4 * (TSDK * TS + TS * (TSDK + 2))
            if local_mem > LOCAL_MEM_BYTES:
                continue

            # ── Build kernel ─────────────────────────────────────
            kernelsource = open(kernel_name).read()
            try:
                program = cl.Program(context, kernelsource).build(options=[
                    f"-DTS={TS}",
                    f"-DWPT={WPT}",
                    f"-DTSDK={TSDK}",
                ])
            except cl.RuntimeError as e:
                print(f"Build error TS={TS} WPT={WPT} TSDK={TSDK}: {e}")
                continue

            transpose_k = program.transpose
            transpose_k.set_scalar_arg_dtypes(
                [numpy.int32, numpy.int32, None, None])

            mmul = program.myGEMM5       # kernel entry point is myGEMM5
            mmul.set_scalar_arg_dtypes(
                [numpy.int32, numpy.int32, numpy.int32, None, None, None])

            # global ND-range for mmul: groups of (TS, RTS) threads
            num_groups_M = int(numpy.ceil(N / TS))
            num_groups_N = int(numpy.ceil(N / TS))
            mmul_gx = num_groups_M * TS
            mmul_gy = num_groups_N * RTS

            print(f"Starting {COUNT} multiplications | "
                  f"TS={TS} WPT={WPT} TSDK={TSDK} "
                  f"(RTS={RTS} LPT={LPT} wg={TS}x{RTS})")

            start_time = time()
            ok = True
            for i in range(COUNT):
                h_C.fill(0.0)
                try:
                    # 1. Transpose B → Bt
                    transpose_k(queue,
                                (trans_gx, trans_gy),
                                (TRANS_LOCAL, TRANS_LOCAL),
                                numpy.int32(N), numpy.int32(N),
                                d_b, d_bt)

                    # 2. Matrix multiply using pre-transposed Bt
                    mmul(queue,
                         (mmul_gx, mmul_gy),
                         (TS, RTS),
                         numpy.int32(N), numpy.int32(N), numpy.int32(N),
                         d_a, d_bt, d_c)

                    queue.finish()
                except Exception as e:
                    print(f"  === Runtime error: {e} ===")
                    ok = False
                    break

            run_time = time() - start_time

            if not ok:
                continue

            cl.enqueue_copy(queue, h_C, d_c)
            print(f"  h_C[0] = {h_C[0]}")

            curr, currs = results(N, COUNT, run_time)
            results_log.append({"TS": TS, "WPT": WPT, "TSDK": TSDK, "MFLOPS": curr, "secs": currs})
            if curr > bestmflops:
                bestmflops = curr
                bestsecs   = currs
                besttuple  = (TS, WPT, TSDK)
                
with open("tuning_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["TS", "WPT", "TSDK", "MFLOPS", "secs"])
    writer.writeheader()
    writer.writerows(results_log)
    
print("Best performance at TS={}, WPT={}, TSDK={}".format(*besttuple))
print (bestsecs, "seconds at", bestmflops, "MFLOPS")
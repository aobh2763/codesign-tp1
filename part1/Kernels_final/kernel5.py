from helper import *
from definitions import *

import pyopencl as cl
import numpy
from time import time

N    = 2048
size = N * N

kernel_name = "part1/Kernels_final/kernel5.cl"

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

# ── Transpose global size (must be multiples of 32) ─────────────
TRANS_LOCAL = 32
trans_gx = int(numpy.ceil(N / TRANS_LOCAL)) * TRANS_LOCAL
trans_gy = int(numpy.ceil(N / TRANS_LOCAL)) * TRANS_LOCAL

# ── Search constraints ───────────────────────────────────────────
MAX_WG_SIZE    = 1024
LOCAL_MEM_BYTES = 48 * 1024

bestmflops = 0.0
besttuple  = (0, 0, 0, 0, 0)

for TSM in [32, 64, 128]:
    for TSN in [32, 64, 128]:
        for TSK in [16, 32, 64]:
            for WPTM in [1]:            # kernel only supports WPTM=1
                if TSM % WPTM != 0:
                    continue
                for WPTN in [1, 2, 4, 8]:
                    if TSN % WPTN != 0:
                        continue

                    RTSM = TSM // WPTM
                    RTSN = TSN // WPTN

                    # work-group size check
                    if RTSM * RTSN > MAX_WG_SIZE:
                        continue

                    # local memory check: Asub(TSK*TSM) + Bsub(TSN*TSK) floats
                    local_mem = 4 * (TSM * TSK + TSN * TSK)
                    if local_mem > LOCAL_MEM_BYTES:
                        continue

                    # LPT must be a whole number
                    threads = RTSM * RTSN
                    if (TSK * TSM) % threads != 0:
                        continue
                    LPT = (TSK * TSM) // threads

                    # ── Build kernel ─────────────────────────────────────
                    kernelsource = open(kernel_name).read()
                    try:
                        program = cl.Program(context, kernelsource).build(options=[
                            f"-DTSM={TSM}",
                            f"-DTSN={TSN}",
                            f"-DTSK={TSK}",
                            f"-DWPTM={WPTM}",
                            f"-DWPTN={WPTN}",
                        ])
                    except cl.RuntimeError as e:
                        print(f"Build error TSM={TSM} TSN={TSN} TSK={TSK}: {e}")
                        continue

                    transpose_k = program.transpose
                    transpose_k.set_scalar_arg_dtypes(
                        [numpy.int32, numpy.int32, None, None])

                    mmul = program.mmul
                    mmul.set_scalar_arg_dtypes(
                        [numpy.int32, numpy.int32, numpy.int32, None, None, None])

                    # global ND-range for mmul
                    num_groups_M = int(numpy.ceil(N / TSM))
                    num_groups_N = int(numpy.ceil(N / TSN))
                    mmul_gx = num_groups_M * RTSM
                    mmul_gy = num_groups_N * RTSN

                    print(f"Starting {COUNT} multiplications | "
                          f"TSM={TSM} TSN={TSN} TSK={TSK} WPTM={WPTM} WPTN={WPTN}")

                    start_time = time()
                    ok = True
                    for i in range(COUNT):
                        h_C.fill(0.0)
                        try:
                            # 1. Transpose B → Bt  (N×N square matrix)
                            transpose_k(queue,
                                        (trans_gx, trans_gy),
                                        (TRANS_LOCAL, TRANS_LOCAL),
                                        numpy.int32(N), numpy.int32(N),
                                        d_b, d_bt)

                            # 2. Matrix multiply using pre-transposed B
                            mmul(queue,
                                 (mmul_gx, mmul_gy),
                                 (RTSM, RTSN),
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

                    curr = results(N, COUNT, run_time)
                    if curr > bestmflops:
                        bestmflops = curr
                        besttuple  = (TSM, TSN, TSK, WPTM, WPTN)

            print(f"Best so far: TSM={besttuple[0]} TSN={besttuple[1]} "
                  f"TSK={besttuple[2]} WPTM={besttuple[3]} WPTN={besttuple[4]} "
                  f"→ {bestmflops:.1f} MFLOPS")
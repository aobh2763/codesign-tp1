import pyopencl as cl
import numpy as np
from time import time

# ================================
# Problem size
# ================================
N = 1024
COUNT = 10

AVAL = 1.0
BVAL = 1.0

# ================================
# Initialize matrices
# ================================
h_A = np.full((N, N), AVAL, dtype=np.float32)
h_B = np.full((N, N), BVAL, dtype=np.float32)

# 🔥 IMPORTANT: Pre-transpose B
h_B = h_B.T.copy()

h_C = np.empty((N, N), dtype=np.float32)

# ================================
# OpenCL setup
# ================================
context = cl.create_some_context()
queue = cl.CommandQueue(context)

# Buffers
d_A = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_A)
d_B = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_B)
d_C = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_C.nbytes)

# ================================
# Parameter search space
# ================================
TS_values   = [8, 16, 32]
WPT_values  = [1, 2, 4, 8]
TSDK_values = [8, 16, 32]
LPT_values  = [1, 2, 4, 8]

# Load kernel template
with open("part1/Kernels_Houssem/Kernel5.cl", "r") as f:
    kernel_template = f.read()

best_time = float("inf")
best_config = None
best_gflops = 0.0

print("Starting tuning...\n")

# ================================
# Tuning loop
# ================================
for TS in TS_values:
    for WPT in WPT_values:

        if TS % WPT != 0:
            continue

        RTS = TS // WPT

        for TSDK in TSDK_values:
            for LPT in LPT_values:

                # 🔥 CORRECT constraint
                if RTS * LPT != TSDK:
                    continue

                # Build kernel with defines
                kernel_code = f"""
                #define TS {TS}
                #define WPT {WPT}
                #define RTS {RTS}
                #define TSDK {TSDK}
                #define LPT {LPT}
                """ + kernel_template

                try:
                    program = cl.Program(context, kernel_code).build()
                    kernel = program.myGEMM5

                    kernel.set_scalar_arg_dtypes([
                        np.int32, np.int32, np.int32,
                        None, None, None
                    ])

                    # Work sizes
                    local = (TS, RTS)
                    global_size = (N, N // WPT)

                    # Timing
                    start = time()

                    for _ in range(COUNT):
                        kernel(queue, global_size, local,
                               np.int32(N), np.int32(N), np.int32(N),
                               d_A, d_B, d_C)
                        queue.finish()

                    elapsed = time() - start

                    # ================================
                    # GFLOPS calculation
                    # ================================
                    gflops = (2 * (N**3) * COUNT) / (elapsed * 1e9)

                    print(f"TS={TS}, WPT={WPT}, TSDK={TSDK}, LPT={LPT} "
                          f"-> {elapsed:.4f}s | {gflops:.2f} GFLOPS")

                    # Track best
                    if elapsed < best_time:
                        best_time = elapsed
                        best_config = (TS, WPT, TSDK, LPT)
                        best_gflops = gflops

                except Exception:
                    print(f"Invalid config TS={TS}, WPT={WPT}, TSDK={TSDK}, LPT={LPT}")

# ================================
# Copy result safely
# ================================
try:
    cl.enqueue_copy(queue, h_C, d_C)
except:
    print("⚠️ Skipping result copy due to previous kernel failure")

# ================================
# Final results
# ================================
print("\n===============================")
print("Best configuration:")
print(f"TS={best_config[0]}, WPT={best_config[1]}, "
      f"TSDK={best_config[2]}, LPT={best_config[3]}")
print(f"Time: {best_time:.4f} s")
print(f"Performance: {best_gflops:.2f} GFLOPS")
print("===============================")
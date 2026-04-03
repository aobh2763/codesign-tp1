# =============================================================================
#  matmul_optimised_host.py
#
#  Pure PyOpenCL host for myGEMM_optimised.cl
#  No CUDA, no vendor-specific libraries.
#  Tested platform: NVIDIA GeForce GTX 1650 Ti  (also runs on Intel OpenCL)
#
#  What this script does, step by step:
#    1.  Allocate and fill host matrices A and B (float32, column-major)
#    2.  Pad A and B so their dimensions are exact multiples of the tile sizes
#    3.  Copy A and B to the GPU
#    4.  Run the `transpose` kernel on GPU to produce Bt = B^T  (one-time cost)
#    5.  Run `myGEMM` COUNT times and measure average throughput
#    6.  Copy C back, validate correctness, print GFLOPS
#
#  Column-major convention (matches the OpenCL kernel):
#    Element [row][col] is stored at offset  col * num_rows + row
#    i.e. columns are contiguous in memory, just like Fortran / BLAS.
#
# =============================================================================

import numpy as np
import pyopencl as cl
import os
from time import time
from definitions_opt import *
from helper_opt import *

# =============================================================================
#  [D]  COMPUTE PADDED DIMENSIONS  [K10]
# =============================================================================

# After padding, M_pad % TSM == 0, N_pad % TSN == 0, K_pad % TSK == 0
# Also pad to a multiple of WIDTH for the vector load addressing.
M_pad = round_up(N, max(TSM, WIDTH))
N_pad = round_up(N, max(TSN, WIDTH))
K_pad = round_up(N, max(TSK, WIDTH))   # K == N for square matrices

print(f"Original matrix size : {N} × {N}")
print(f"Padded  matrix size  : M_pad={M_pad}, N_pad={N_pad}, K_pad={K_pad}")
assert M_pad % TSM == 0, f"M_pad={M_pad} not divisible by TSM={TSM}"
assert N_pad % TSN == 0, f"N_pad={N_pad} not divisible by TSN={TSN}"
assert K_pad % TSK == 0, f"K_pad={K_pad} not divisible by TSK={TSK}"
assert M_pad % WIDTH == 0, f"M_pad must be divisible by WIDTH={WIDTH}"
assert N_pad % WIDTH == 0, f"N_pad must be divisible by WIDTH={WIDTH}"

# =============================================================================
#  [E]  HOST MATRICES  (column-major, padded)
# =============================================================================

# Original unpadded matrices
h_A_orig = np.full(N * N, AVAL, dtype=np.float32)   # A: N×N all-ones
h_B_orig = np.full(N * N, BVAL, dtype=np.float32)   # B: N×N all-ones

# Padded versions
h_A  = pad_matrix(h_A_orig, N, N, M_pad, K_pad)   # A_padded: M_pad × K_pad
h_B  = pad_matrix(h_B_orig, N, N, N,    K_pad)    # B_padded: N     × K_pad
                                                   # (Bt will be K_pad × N_pad)

h_C  = np.zeros(M_pad * N_pad, dtype=np.float32)   # Output buffer (padded)

# =============================================================================
#  [F]  OPENCL SETUP
# =============================================================================

# Let the user choose their platform/device (or set PYOPENCL_CTX env var)
context = cl.create_some_context()
queue   = cl.CommandQueue(context,
              properties=cl.command_queue_properties.PROFILING_ENABLE)

device = queue.device
print(f"\nDevice : {device.name}")
print(f"Max local mem : {device.local_mem_size // 1024} KB")
local_mem_needed = 2 * (TSK * TSM+ TSK * TSN) * 4
print(f"Kernel needs  : {local_mem_needed // 1024} KB local mem per work-group")
assert local_mem_needed <= device.local_mem_size, \
    "Not enough local memory! Reduce TSM, TSN, or TSK."

# =============================================================================
#  [G]  BUILD THE KERNEL
#
#  We pass the tuning parameters as -D preprocessor flags so the kernel
#  sees the same values we have here.  This is the standard way to tune
#  OpenCL kernels without rewriting source.
# =============================================================================

BUILD_OPTS = (
    f"-DTSM={TSM} -DTSN={TSN} -DTSK={TSK} "
    f"-DWPTM={WPTM} -DWPTN={WPTN} -DWIDTH={WIDTH} "
    f"-cl-fast-relaxed-math"          # allow FMA contraction and fast divisions
)

kernel_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "myGEMM_optimised.cl")
with open(kernel_file) as f:
    source = f.read()

print(f"\nBuilding kernel with: {BUILD_OPTS}")
try:
    program = cl.Program(context, source).build(options=BUILD_OPTS)
except cl.RuntimeError as e:
    print("Build failed:\n", e.args[2].decode())   # print the compiler log
    raise

transpose_kn = program.transpose
gemm_kn      = program.myGEMM

# Declare which arguments are scalars (int32) vs buffers (None)
# transpose : P(int), Q(int), input(buf), output(buf)
transpose_kn.set_scalar_arg_dtypes([np.int32, np.int32, None, None])
# myGEMM    : M(int), N(int), K(int), A(buf), Bt(buf), C(buf)
gemm_kn.set_scalar_arg_dtypes([np.int32, np.int32, np.int32, None, None, None])

# =============================================================================
#  [H]  GPU BUFFERS
# =============================================================================

mf = cl.mem_flags

# A: read-only, copy from host immediately
d_A  = cl.Buffer(context, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=h_A)

# B: read-only source for the transpose kernel
d_B  = cl.Buffer(context, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=h_B)

# Bt: read-write target of the transpose kernel, then read-only input to GEMM
#     Size = N_pad * K_pad floats  (shape: N_pad × K_pad, column-major)
d_Bt = cl.Buffer(context, mf.READ_WRITE,
                 size=int(N_pad * K_pad * np.dtype(np.float32).itemsize))

# C: write-only output of GEMM
d_C  = cl.Buffer(context, mf.WRITE_ONLY,
                 size=int(M_pad * N_pad * np.dtype(np.float32).itemsize))

# =============================================================================
#  [I]  STEP 1 — TRANSPOSE B  [K5]
#
#  We call the `transpose` kernel once (outside the timed loop).
#  Input  d_B  has shape N × K_pad (column-major)  → P=K_pad, Q=N
#  Output d_Bt has shape K_pad × N (column-major)  → Bt[k][n] = B[n][k]
#
#  The transpose kernel launch:
#    local  = { TRANSPOSEX, TRANSPOSEY } = { 16, 16 }
#    global = rounded-up to multiples of 16
# =============================================================================

global_tr = (
    round_up(K_pad, TRANSPOSEX),   # covers the P=K_pad direction
    round_up(N,     TRANSPOSEY),   # covers the Q=N     direction
)
local_tr = (TRANSPOSEX, TRANSPOSEY)

# B is stored as N rows × K_pad cols column-major → call it P=K_pad, Q=N
# so the transpose kernel reads input[q * P + p] = B[n * K_pad + k]
# and writes output[p * Q + q] = Bt[k * N + n]  ← Bt is K_pad × N column-major
transpose_kn(queue, global_tr, local_tr,
             np.int32(K_pad),   # P = number of columns of B (= K_pad)
             np.int32(N),       # Q = number of rows    of B (= N)
             d_B, d_Bt)
queue.finish()
print("B transposed successfully → Bt ready on GPU.")

# =============================================================================
#  [J]  STEP 2 — TIMED GEMM LOOP  [K2,K3,K4,K5,K6,K7,K9,K10]
#
#  Launch config:
#    local  = { RTSM, RTSN }         e.g. { 16, 16 }
#    global = { M_pad/WPTM, N_pad/WPTN }   e.g. { 256, 256 }
#
#  The kernel computes C_padded = A_padded * B_padded.
#  After the loop we copy back only the un-padded N×N region.
# =============================================================================

global_gemm = (M_pad // WPTM, N_pad // WPTN)
local_gemm  = (RTSM, RTSN)

print(f"\nGEMM launch config:")
print(f"  global work size = {global_gemm}  ({global_gemm[0]*global_gemm[1]} total threads)")
print(f"  local  work size = {local_gemm}   ({local_gemm[0]*local_gemm[1]} threads/work-group)")
print(f"\nStarting {COUNT} iterations ...\n")

errors     = 0
start_time = time()

for i in range(COUNT):
    try:
        gemm_kn(queue, global_gemm, local_gemm,
                np.int32(M_pad),   # M
                np.int32(N_pad),   # N
                np.int32(K_pad),   # K
                d_A, d_Bt, d_C)
        queue.finish()
    except Exception as exc:
        errors += 1
        print(f"  Iteration {i} failed: {exc}")

total_time = time() - start_time

# =============================================================================
#  [K]  STEP 3 — READ BACK AND VALIDATE
#
#  C_padded is M_pad × N_pad.  We only care about the top-left N × N block.
#  For column-major storage, column c of the un-padded result starts at
#  offset  c * M_pad  in the padded buffer.
# =============================================================================

cl.enqueue_copy(queue, h_C, d_C)
queue.finish()

# Extract the un-padded N×N result from the padded M_pad×N_pad column-major buffer
C_2d_padded = h_C.reshape(N_pad, M_pad).T      # shape (M_pad, N_pad) row-major
C_result    = C_2d_padded[:N, :N]              # top-left N×N

# Expected value: every element = sum of N ones × ones = N  (for A=B=1)
expected = float(N) * AVAL * BVAL
correct  = np.allclose(C_result, expected, rtol=1e-3)

print(f"C[0,0]     = {C_result[0,0]:.1f}   (expected {expected:.1f})")
print(f"C[N//2,0]  = {C_result[N//2, 0]:.1f}")
print(f"C[N-1,N-1] = {C_result[-1,-1]:.1f}")
print(f"Result correct: {correct}")

if not correct:
    # Show where errors are
    diff = np.abs(C_result - expected)
    print(f"Max error: {diff.max():.4f}  at index {np.unravel_index(diff.argmax(), diff.shape)}")

# =============================================================================
#  [L]  PERFORMANCE REPORT
# =============================================================================

avg_time_s = total_time / COUNT
# GEMM floating-point count: 2*N^3 (N multiply-adds per element, N^2 elements)
flops   = 2.0 * N * N * N
gflops  = flops / avg_time_s / 1e9
run_time = time() - start_time
mflops = 2.0 * COUNT * N * N * N/(1000000.0* run_time)

print(f"\n{'='*54}")
print(f"  Matrix size     : {N} × {N}")
print(f"  Iterations      : {COUNT}  (errors: {errors})")
print(f"  Total time      : {total_time:.4f} s")
print(f"  Avg time/iter   : {avg_time_s * 1000:.3f} ms")
print(f"  Throughput      : {mflops:.2f} GFLOPS")
print(f"{'='*54}")
print()
print("Tuning parameters used:")
print(f"  TSM={TSM}  TSN={TSN}  TSK={TSK}")
print(f"  WPTM={WPTM}  WPTN={WPTN}  WIDTH={WIDTH}")
print(f"  Work-group: {RTSM}×{RTSN} = {RTSM*RTSN} threads")

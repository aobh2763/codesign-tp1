"""
Heterogeneous dual-GPU matrix multiplication.

  NVIDIA GPU  → naive uncoalesced kernel  → rows [0        .. rows_nv - 1]
  Intel iGPU  → optimised tiled kernel   → rows [rows_nv  .. N - 1      ]

Matrix layout convention used by BOTH kernels:
  Matrices are stored column-major (Fortran order) so that
  element (row, col) lives at index  col*M + row.

The naive kernel reads A and B with this convention directly.
The optimised kernel uses the same convention but issues wide
(float2 / float4) loads, so A and B must be padded so that
  M % WIDTH == 0  and  N % WIDTH == 0.
With N=8192 and WIDTH=2 both conditions are satisfied trivially.

Work-group sizes
  NVIDIA (16×16 threads)  → global = (rows_nv, N)   local = (16, 16)
  iGPU   (8×16  threads)  → global = (rows_ig, N)   local = (RTSM=8, RTSN=16)
    where RTSM = TSM/WPTM = 128/16 = 8
          RTSN = TSN/WPTN = 128/8  = 16

Both kernels write C in column-major order:
  C[col * M + row] = acc
so the output buffer can be stitched back together without reordering.
"""

import numpy as np
import pyopencl as cl
from time import time
import os
import sys

# ---------------------------------------------------------------------------
# Constants – adjust to match your helper / definitions files
# ---------------------------------------------------------------------------
AVAL  = 1.0          # fill value for A
BVAL  = 1.0          # fill value for B
COUNT = 10           # number of timed iterations

# ---------------------------------------------------------------------------
# Matrix dimensions
# ---------------------------------------------------------------------------
N    = 8192
size = N * N

# Expected result value for correctness check
cval = float(N) * AVAL * BVAL

# ---------------------------------------------------------------------------
# Work split
#   alpha  = fraction of rows handled by the NVIDIA GPU (naive kernel)
#   1-alpha = fraction of rows handled by the Intel iGPU (optimised kernel)
# ---------------------------------------------------------------------------
alpha = 0.917

rows_nv_raw = int(round(alpha * N))
rows_ig_raw = N - rows_nv_raw

# Round to the nearest multiple of the respective tile sizes
NV_TILE = 16    # local size in the M dimension for the naive kernel
IG_TILE = 128   # TSM for the optimised kernel

rows_nv = (rows_nv_raw // NV_TILE) * NV_TILE
rows_ig = N - rows_nv
# Make sure iGPU share is also a multiple of IG_TILE (adjust if needed)
rows_ig = (rows_ig // IG_TILE) * IG_TILE
rows_nv = N - rows_ig          # recompute so they sum to N exactly

print(f"Matrix size     : {N}×{N}")
print(f"Split: NVIDIA rows  0 .. {rows_nv-1:5d}  ({rows_nv} rows, {alpha*100:.1f}%)")
print(f"       iGPU   rows {rows_nv} .. {N-1:5d}  ({rows_ig} rows, {(1-alpha)*100:.1f}%)")

# ---------------------------------------------------------------------------
# Host arrays – column-major (Fortran) storage
#   A : shape (K, M) stored as (N, N) column-major  → A[k, row] = A_flat[row + k*N]
#   B : shape (N, K) stored column-major            → B[col, k] = B_flat[k + col*N]
# ---------------------------------------------------------------------------
h_A = np.full(size, AVAL, dtype=np.float32)
h_B = np.full(size, BVAL, dtype=np.float32)
h_C = np.zeros(size, dtype=np.float32)

# ---------------------------------------------------------------------------
# Device discovery
# ---------------------------------------------------------------------------
all_platforms = cl.get_platforms()
all_gpus = [d for p in all_platforms for d in p.get_devices(cl.device_type.GPU)]

if len(all_gpus) < 2:
    print("\nERROR: fewer than 2 GPU devices found.")
    print("Available devices:")
    for d in all_gpus:
        print(f"  {d.name}")
    sys.exit(1)

nvidia_dev = next(
    (d for d in all_gpus if 'NVIDIA' in d.name or 'GeForce' in d.name or 'RTX' in d.name),
    None
)
if nvidia_dev is None:
    print("Could not identify NVIDIA device – using first GPU as NVIDIA.")
    nvidia_dev = all_gpus[0]

igpu_dev = next((d for d in all_gpus if d is not nvidia_dev), None)
if igpu_dev is None:
    print("Could not identify iGPU device.")
    sys.exit(1)

print(f"\nDevice 1 (NVIDIA) : {nvidia_dev.name}")
print(f"Device 2 (iGPU)   : {igpu_dev.name}\n")

# ---------------------------------------------------------------------------
# Contexts and queues
# ---------------------------------------------------------------------------
ctx_nv   = cl.Context([nvidia_dev])
ctx_ig   = cl.Context([igpu_dev])
queue_nv = cl.CommandQueue(ctx_nv, properties=cl.command_queue_properties.PROFILING_ENABLE)
queue_ig = cl.CommandQueue(ctx_ig, properties=cl.command_queue_properties.PROFILING_ENABLE)

# ---------------------------------------------------------------------------
# Device buffers
#
# NVIDIA receives rows [0 .. rows_nv-1] of A.
#   In column-major layout A is stored as K columns of M elements each.
#   For a submatrix of M'=rows_nv rows we send the first rows_nv elements
#   of every column, which is NOT contiguous in the flat array.
#
#   Simple fix: copy the full A to both devices and let each kernel address
#   only its own rows via gidm * TSM.  This wastes some bandwidth but
#   avoids a costly gather on the host.  For production code you would
#   pack the strided slice explicitly.
#
# B is the same for both devices (full N×K matrix).
# ---------------------------------------------------------------------------

# -- NVIDIA buffers ----------------------------------------------------------
# A slice for NVIDIA: rows 0..rows_nv-1.
# In column-major A[k, row] is at A_flat[row + k*N].
# The sub-matrix for NVIDIA has 'rows_nv' rows; we pass M=rows_nv to the
# kernel so it only processes those rows.  We still need the full K columns
# of A but only the first rows_nv elements of each column.
# Easiest correct approach: pass A sub-matrix packed column-by-column.
h_A_nv = np.empty(rows_nv * N, dtype=np.float32)
for k in range(N):
    h_A_nv[k*rows_nv : (k+1)*rows_nv] = h_A[k*N : k*N + rows_nv]

d_a_nv = cl.Buffer(ctx_nv, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                   hostbuf=h_A_nv)
d_b_nv = cl.Buffer(ctx_nv, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                   hostbuf=h_B)
d_c_nv = cl.Buffer(ctx_nv, cl.mem_flags.WRITE_ONLY,
                   size=rows_nv * N * h_C.itemsize)

# -- iGPU buffers ------------------------------------------------------------
# A slice for iGPU: rows rows_nv..N-1.
h_A_ig = np.empty(rows_ig * N, dtype=np.float32)
for k in range(N):
    h_A_ig[k*rows_ig : (k+1)*rows_ig] = h_A[k*N + rows_nv : k*N + N]

d_a_ig = cl.Buffer(ctx_ig, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                   hostbuf=h_A_ig)
d_b_ig = cl.Buffer(ctx_ig, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                   hostbuf=h_B)
d_c_ig = cl.Buffer(ctx_ig, cl.mem_flags.WRITE_ONLY,
                   size=rows_ig * N * h_C.itemsize)

# ---------------------------------------------------------------------------
# Kernel sources
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))

kernel_nv_path = os.path.join(script_dir, "kernel_uncoalsed.cl")
kernel_ig_path = os.path.join(script_dir, "kernel_ultimate.cl")

with open(kernel_nv_path, "r") as f:
    src_nv = f.read()

with open(kernel_ig_path, "r") as f:
    src_ig = f.read()

# ---------------------------------------------------------------------------
# Build programs
# ---------------------------------------------------------------------------
prog_nv = cl.Program(ctx_nv, src_nv).build()
prog_ig = cl.Program(ctx_ig, src_ig).build(
    options="-DTSM=128 -DTSN=128 -DTSK=16 -DWPTM=16 -DWPTN=8 -DWIDTH=2"
)

mmul_nv = prog_nv.mmul
mmul_ig = prog_ig.mmul

mmul_nv.set_scalar_arg_dtypes([np.int32, np.int32, np.int32, None, None, None])
mmul_ig.set_scalar_arg_dtypes([np.int32, np.int32, np.int32, None, None, None])

# ---------------------------------------------------------------------------
# Work-group / global sizes
#
# NVIDIA naive kernel:
#   get_global_id(0) → row,  get_global_id(1) → col
#   global = (rows_nv, N),   local = (16, 16)
#
# iGPU optimised kernel:
#   get_local_id(0)  → tidm  (range [0, RTSM) = [0,8))
#   get_local_id(1)  → tidn  (range [0, RTSN) = [0,16))
#   get_group_id(0)  → gidm  selects tile in M direction
#   get_group_id(1)  → gidn  selects tile in N direction
#   global = (rows_ig / WPTM, N / WPTN)
#           = (rows_ig/16,    N/8)
#   local  = (RTSM, RTSN) = (8, 16)
# ---------------------------------------------------------------------------
RTSM = 128 // 16   # = 8
RTSN = 128 // 8    # = 16
WPTM = 16
WPTN = 8

global_nv = (rows_nv, N)
local_nv  = (16, 16)

global_ig = (rows_ig // WPTM, N // WPTN)
local_ig  = (RTSM, RTSN)

print(f"NVIDIA global={global_nv}  local={local_nv}")
print(f"iGPU   global={global_ig}  local={local_ig}")

# ---------------------------------------------------------------------------
# Timed loop
# ---------------------------------------------------------------------------
print(f"\nStarting {COUNT} dual-GPU matrix multiplications …\n")
start_time = time()

for i in range(COUNT):
    # Enqueue both kernels – they run concurrently on separate devices.
    mmul_nv(queue_nv, global_nv, local_nv,
            np.int32(rows_nv), np.int32(N), np.int32(N),
            d_a_nv, d_b_nv, d_c_nv)

    mmul_ig(queue_ig, global_ig, local_ig,
            np.int32(rows_ig), np.int32(N), np.int32(N),
            d_a_ig, d_b_ig, d_c_ig)

    # Flush both queues to push work to the devices without stalling the CPU.
    queue_nv.flush()
    queue_ig.flush()

    # Wait for both to finish before the next iteration.
    queue_nv.finish()
    queue_ig.finish()

run_time = time() - start_time
print(f"End of {COUNT} multiplications\n")

# ---------------------------------------------------------------------------
# Read back and assemble C
# ---------------------------------------------------------------------------
h_C_nv = np.empty(rows_nv * N, dtype=np.float32)
h_C_ig = np.empty(rows_ig * N, dtype=np.float32)

cl.enqueue_copy(queue_nv, h_C_nv, d_c_nv)
cl.enqueue_copy(queue_ig, h_C_ig, d_c_ig)
queue_nv.finish()
queue_ig.finish()

# Both kernels write C in column-major order: C[col*M + row]
# Reconstruct full column-major C from the two sub-results.
# Sub-result for NVIDIA: C_nv[col*rows_nv + local_row]   for local_row in [0, rows_nv)
# Sub-result for iGPU  : C_ig[col*rows_ig + local_row]   for local_row in [0, rows_ig)
# Full C:               C[col*N    + row]
for col in range(N):
    h_C[col*N        : col*N + rows_nv] = h_C_nv[col*rows_nv : (col+1)*rows_nv]
    h_C[col*N+rows_nv: col*N + N      ] = h_C_ig[col*rows_ig : (col+1)*rows_ig]

# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------
tolerance = 1e-3
max_err   = np.max(np.abs(h_C - cval))
ok        = max_err < tolerance * abs(cval)

print(f"Expected value : {cval:.6g}")
print(f"Max error      : {max_err:.6g}")
print(f"Correctness    : {'PASS ✓' if ok else 'FAIL ✗'}")

# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------
total_flops = 2.0 * N**3
elapsed_avg = run_time / COUNT
gflops      = total_flops / elapsed_avg / 1e9

print(f"\nTotal wall time : {run_time:.3f} s  ({COUNT} iterations)")
print(f"Average per run : {elapsed_avg*1000:.1f} ms")
print(f"Dual-GPU GFLOPS : {gflops:.1f} GFLOPS")
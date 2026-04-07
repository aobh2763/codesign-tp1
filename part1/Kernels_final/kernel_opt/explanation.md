# Ultimate Kernel — Design Explanation

## Which kernels were used (and which were skipped)

Kernels 5 and 8 are marked TBD in the README and were excluded entirely.
Kernels 1 (naive) and 2 (basic tiling) served as the foundation that every
later kernel builds on, so their ideas are absorbed rather than directly
copied. The concrete techniques brought in from each kernel are described
below.

---

## 1. Local-Memory Tiling  (from Kernel 2)

**Problem it solves:** The naive kernel (K1) reads every element of A and B
from slow global memory once per output element, producing O(M·N·K) global
reads with zero reuse.

**Mechanism:** The work-group cooperatively loads a TSK×TSM sub-block of A
and a TSK×TSN sub-block of B into `__local` memory (on-chip SRAM, ~100×
lower latency than global DRAM). Every thread in the group then reads the
values it needs from local memory. Because the tile is shared, each global
load is reused TSM times for A and TSN times for B.

**Parameters kept from K2:** the conceptual tile structure. The actual tile
sizes (TSM=128, TSN=128, TSK=16) come from the K10 benchmark.

---

## 2. Work-Per-Thread / Per-Thread Output Blocking  (from Kernel 3)

**Problem it solves:** With one output element per thread the per-thread
overhead (index arithmetic, synchronisation amortisation) is high relative
to useful FLOPs.

**Mechanism:** Each thread computes multiple output elements. In K3 this is
a 1-D strip of WPT elements along one axis. The ultimate kernel generalises
this to a 2-D block (WPTM × WPTN), described further in section 4.

**Parameters:** WPTM=16, WPTN=8 → each thread computes 128 output values,
amortising all overhead 128× compared to K1.

---

## 3. Wide (Vector) Global Memory Loads  (from Kernels 4 and 7)

**Problem it solves:** Issuing one 32-bit load per element wastes memory
bus bandwidth. Modern GPUs have wide load/store units that can fetch 64,
128, or 256 bits in a single instruction.

**Mechanism:** The kernel declares `typedef float4 floatX` (WIDTH=4) and
reads 4 consecutive floats from A and B in a single load instruction. This
halves the number of load instructions compared to scalar code and keeps the
memory bus fully saturated.

**Why WIDTH=4 instead of K10's WIDTH=2:** K7 demonstrated that float4 loads
with the same tile structure reached 2.82 GFLOPS vs K10's 3.90 GFLOPS, but
K10 used WIDTH=2 because its padding/scatter code was less optimised. The
ultimate kernel inherits K10's double-buffered scatter structure and applies
it with WIDTH=4, which should match or exceed K10's performance.

**Scatter after load:** After a wide load the WIDTH scalar components are
scattered into the flat local-memory tile with `#if WIDTH == 4` blocks.
This is copied exactly from K7 and K10.

---

## 4. 2-D Register Blocking + B Cached in Registers  (from Kernels 6 and 7)

**Problem it solves:** Even with local memory, reading the same B value
from local memory once per wm-step in the innermost loop is wasteful. Local
memory has finite bandwidth (~2–4 reads per cycle depending on the GPU).

**Mechanism — 2D blocking (K6):**
Each thread is assigned a WPTM × WPTN sub-rectangle of C. The accumulation
array `float acc[WPTM][WPTN]` lives entirely in registers — never written
back until the very end. This means the full output tile is computed with
zero global-memory writes during the tile loop.

**Mechanism — Breg caching (K6/K7):**
For each k-step inside a TSK tile, the kernel first loads WPTN B-values
into the private array `float Breg[WPTN]`. It then iterates over the WPTM
A-values, and for each A scalar it loops over all WPTN pre-loaded B values.
This means:

- Each local B read is reused WPTM=16 times.
- Each local A read is reused WPTN=8 times.
- The ratio of FLOPs to local-memory accesses becomes
  (2·WPTM·WPTN) / (WPTM + WPTN) ≈ 10.7, versus 1.0 for K1.

**Parameters (from K6 best run):**
TSM=128, TSN=128, TSK=16, WPTM=16, WPTN=8.
Thread count per work-group: RTSM×RTSN = 8×16 = 128.

---

## 5. Double-Buffered Pre-fetching  (from Kernels 9 and 10)

**Problem it solves:** Even with local-memory tiling, every tile requires
a `barrier()` after the load, during which the compute units sit idle
waiting for global loads to complete. K9 measured this stall at ~5× slowdown
(0.93 s vs 0.14 s for K10 which improved the implementation).

**Mechanism:** Two copies of each tile array are allocated:
`Asub[2][TSK*TSM]` and `Bsub[2][TSK*TSN]`. The tile indices alternate
between `t%2` and `(t+1)%2`. Within each iteration:

1. **Stage 1** loads tile `t+1` into buffer `(t+1)%2`.
2. **Stage 2** computes using tile `t` in buffer `t%2`.

Both stages execute concurrently on a GPU that can overlap memory and ALU
work. The barrier at the end of the iteration only needs to ensure Stage 1's
stores are visible before the *next* iteration's Stage 2 reads them — it
does not stall Stage 2 of the current iteration.

The initial tile-0 pre-load before the loop mirrors K9's design exactly.

**Key difference from K9:** K10 (and this kernel) use the correct flat
`[2][TSK*TSM]` layout and omit a redundant barrier that K9 placed between
Stage 1 and Stage 2, which negated the double-buffering benefit and caused
K9's poor result (0.93 s).

---

## 6. Arbitrary Matrix Sizes / Incomplete Tile Handling  (from Kernel 10)

**Problem it solves:** All earlier kernels (K2–K9) require M, N, and K to
be exact multiples of the tile dimensions. This limits usability and forces
the host to pad matrices, wasting memory and computation.

**Mechanism:** Two guard checks are added:

1. **Load guard:** before each wide vector load the kernel checks
   `tiledIndex < K` and `globalRow * WIDTH < M` (or N). Out-of-range
   elements are replaced with `(floatX)(0.0f)`, which contributes nothing
   to the dot-product accumulation.

2. **Store guard:** before writing C the kernel checks
   `globalRow < M && globalCol < N`, preventing writes outside the valid
   output region.

These checks are on the load/store paths only, not inside the innermost
compute loop, so they carry negligible cost on large matrices.

---

## Combined parameter summary

| Parameter | Value | Source |
|-----------|-------|--------|
| TSM       | 128   | K10 best |
| TSN       | 128   | K10 best |
| TSK       | 16    | K10 best |
| WPTM      | 16    | K10 best |
| WPTN      | 8     | K10 best |
| WIDTH     | 4     | K7 best (upgraded from K10's 2) |
| RTSM      | 8     | derived |
| RTSN      | 16    | derived |
| Work-group| 128   | 8 × 16  |
| Outputs/thread | 128 | 16 × 8 |

---

## What was deliberately NOT included

**Kernel 5 (transposed B + rectangular tiles):** Marked TBD — excluded.

**Kernel 8 (CUDA/Kepler shuffle instructions):** Marked TBD and uses
`__shfl()`, a CUDA/NVIDIA-specific intrinsic unavailable in portable OpenCL.
The register-caching in `Breg[]` (K6) achieves the same goal portably.

**Kernel 11 (mystery / clBLAS-style):** K11 achieves 3.27 GFLOPS via a
different approach — entirely register-based (no local memory), purely
vectorised types (`floatA`, `floatB`, `floatC`), and a different loop
structure. Its best time is 15% slower than K10's. Its core idea (wide
register loads) is already captured by the WIDTH=4 mechanism. Merging K11's
register-only loop structure would conflict with the double-buffered local
memory that produces K10's superior throughput, so it was not included.

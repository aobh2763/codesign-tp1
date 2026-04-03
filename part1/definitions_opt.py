
N     = 2048   # Matrix order: A(N×N), B(N×N), C(N×N)
               # Must be divisible by TSM, TSN, TSK, and WIDTH after padding.
COUNT = 20     # Number of timed GEMM iterations

AVAL  = 1.0    # Fill value for A  (all ones makes the expected result easy: N)
BVAL  = 1.0    # Fill value for B

# =============================================================================
#  [B]  KERNEL TUNING PARAMETERS
#       These MUST match the #defines compiled into myGEMM_optimised.cl.
#       If you change them here, also pass them as -D flags in BUILD_OPTS.
# =============================================================================

TSM   = 128   # Tile size M:  work-group covers TSM rows of A / C
TSN   = 128   # Tile size N:  work-group covers TSN cols of Bt / C
TSK   =  16    # Tile size K:  inner dimension is processed in chunks of TSK
WPTM  =   8    # Work-per-thread M: each thread computes WPTM rows
WPTN  =   8    # Work-per-thread N: each thread computes WPTN cols
WIDTH =   4    # Vector width: 4 → float4 (128-bit loads), 1 → scalar

# Derived values (mirror the #defines in the kernel)
RTSM  = TSM // WPTM    # Work-group size, dim 0  (e.g. 16)
RTSN  = TSN // WPTN    # Work-group size, dim 1  (e.g. 16)

# The transpose kernel's local work-group size (hardcoded in the kernel too)
TRANSPOSEX = 16
TRANSPOSEY = 16

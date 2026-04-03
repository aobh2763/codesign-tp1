# =============================================================================
#  [C]  HELPER: PAD A MATRIX TO A MULTIPLE OF `multiple`
#
#  The kernel assumes M % TSM == 0, N % TSN == 0, K % TSK == 0.
#  We zero-pad the matrix to satisfy these requirements [K10].
#  The host strips the extra rows/columns when reading C back.
#
#  `mat`      : 1-D numpy array, column-major  (length = rows * cols)
#  `rows`     : original number of rows
#  `cols`     : original number of columns
#  `pad_rows` : target row count  (>= rows, multiple of row_multiple)
#  `pad_cols` : target col count  (>= cols, multiple of col_multiple)
#  Returns    : 1-D numpy array of length pad_rows * pad_cols, column-major
# =============================================================================
import numpy as np

def pad_matrix(mat, rows, cols, pad_rows, pad_cols):
    """Zero-pad a column-major matrix to (pad_rows × pad_cols)."""
    # Reshape to 2D (col-major → index as [row, col] by transposing)
    m2d = mat.reshape(cols, rows).T          # shape (rows, cols)  row-major view
    padded = np.zeros((pad_rows, pad_cols), dtype=np.float32)
    padded[:rows, :cols] = m2d
    # Return as column-major 1-D array
    return padded.T.reshape(-1).astype(np.float32)


def round_up(value, multiple):
    """Round `value` up to the nearest multiple of `multiple`."""
    return ((value + multiple - 1) // multiple) * multiple


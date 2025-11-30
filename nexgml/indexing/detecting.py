import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, spmatrix
from typing import Optional

def isbiner(x, axis: Optional[int] | None = None) -> bool | np.ndarray:
    """
    Check if array contains only binary values {0,1}.
    Supports dense (list/numpy) and sparse CSR/CSC matrices.
    
    ## Args:
        **x**: *np.ndarray | spmatrix | list*
        Input data to be checked.

        **axis**: *Optional[int] | None*
        Axis along which to check binary values:
            - If `None`, checks the entire array/matrix.
            - If `0`, checks column-wise.
            - If `1`, checks row-wise.

    ## Returns:
        **bool | np.ndarray**: *True if input contains only binary values, False otherwise.*

    ## Raises:
        **None**
    """
    # ---------- Sparse ----------
    if isinstance(x, (csr_matrix, csc_matrix)):
        if axis is None:
            # Data nonzero harus bernilai 0 atau 1
            return np.all((x.data == 0) | (x.data == 1))
        if axis == 0:
            # Per kolom → kolom valid bila semua nnz adalah 0/1
            valid = np.zeros(x.shape[1], dtype=bool)
            for j in range(x.shape[1]):
                col = x.getcol(j)
                valid[j] = np.all((col.data == 0) | (col.data == 1))
            return valid
        if axis == 1:
            valid = np.zeros(x.shape[0], dtype=bool)
            for i in range(x.shape[0]):
                row = x.getrow(i)
                valid[i] = np.all((row.data == 0) | (row.data == 1))
            return valid
        return False

    # ---------- Dense ----------
    arr = np.asarray(x)

    if axis is None:
        return np.all((arr == 0) | (arr == 1))

    return np.all((arr == 0) | (arr == 1), axis=axis)

def isone_hot(x: np.ndarray | spmatrix | list, axis: Optional[int] | None = None) -> bool | np.ndarray:
    """
    Robust one-hot detector for dense or sparse data.

    ## Args:
        **x**: *np.ndarray | spmatrix | list*
        Input data to be checked.

        **axis**: *Optional[int] | None*
        Axis along which to check one-hot encoding:
            - If `None`, checks the entire array/matrix.
            - If `0`, checks column-wise.
            - If `1`, checks row-wise.

    ## Returns:
        **bool | np.ndarray**: *True if input is one-hot encoded, False otherwise.*

    ## Raises:
        **None**
    """

    # ---------- Sparse (CSR/CSC) ----------
    if isinstance(x, (csr_matrix, csc_matrix)):

        # 1D sparse tidak valid
        if x.ndim != 2:
            return False

        # Nonzero harus 1
        if not np.all(x.data == 1):
            return False if axis is None else np.zeros(x.shape[axis], dtype=bool)

        if axis is None:
            # Per baris harus <= 1 nonzero
            nnz_per_row = np.diff(x.indptr)
            return np.all(nnz_per_row == 1)

        elif axis == 1:  # row-wise check
            nnz = np.diff(x.indptr)
            return nnz == 1

        elif axis == 0:  # column-wise check
            valid = np.zeros(x.shape[1], dtype=bool)
            for j in range(x.shape[1]):
                col = x.getcol(j)
                valid[j] = (col.nnz == 1 and np.all(col.data == 1))
            return valid

        return False


    # ---------- Dense ----------
    arr = np.asarray(x)

    # ---------- 1D ----------
    if arr.ndim == 1:
        if axis is not None:
            raise ValueError("axis for 1D data must be None.")
        if not isbiner(arr):
            return False
        return arr.sum() == 1

    # ---------- 2D ----------
    if arr.ndim == 2:

        if axis is None:
            # Full matrix check
            if not isbiner(arr):
                return False
            row_sum = arr.sum(axis=1)
            return np.all(row_sum == 1)

        elif axis == 1:  # row-wise
            if not isbiner(arr, axis=1).all():
                return False
            row_sum = arr.sum(axis=1)
            return row_sum == 1

        elif axis == 0:  # column-wise
            if not isbiner(arr, axis=0).all():
                return False
            col_sum = arr.sum(axis=0)
            return col_sum == 1

        return False

    # >2D data not supported
    return False


def _detect_interval_1d(arr: np.ndarray) -> tuple[bool, int]:
    """
    Detect if a 1D array has constant intervals between nonzero elements.
    """

    nz_idx = np.where(arr != 0)[0]

    if len(nz_idx) < 1:
        return False, 0
    
    diffs = np.diff(nz_idx)

    if len(diffs) == 0:
        return True, 0

    interval = diffs[0]
    if np.all(diffs == interval) and interval > 0:
        return True, interval
    return False, 0



def isizi(x, return_style=False, return_intervals=False):
    """
    Detect whether data is in IZI sparse interval format (RI / CI / R-CI).

    ## Args:
        **x**: *np.ndarray | spmatrix | list*
            Input data to be checked.

        **return_style**: *bool*
            Whether to return detected style ("ri", "ci", "rci").

        **return_intervals**: *bool*
            Whether to return detected interval size.

    ## Returns:
        **bool | tuple**:
            Detection results depending on return flags.
    """
    # ---------- Dense normalization ----------
    if isinstance(x, (csr_matrix, csc_matrix)):
        arr = x.toarray()
    else:
        arr = np.asarray(x)

    if arr.ndim != 2:
        out = False
        if return_style or return_intervals:
            return (out, None, 0)
        return out

    rows, cols = arr.shape

    # ---------- Detect RI (row-intervals) ----------
    # We look for vertical spacing pattern: every original row appears in multiples
    # Example: intervals=2 → pattern: row indices = [0,0, 1,1, 2,2, ...]
    # Check spacing of NONZERO entries per column.
    def detect_RI(arr):
        intervals = []
        for row in range(arr.shape[0]):
            nz = np.where(arr[:, row] != 0)[0]
            if len(nz) <= 1:
                continue

            dif = np.diff(nz)
            if np.all(dif == dif[0]) and dif[0] > 0:
                intervals.append(dif[0])

            else:
                return False, 0
            
        if len(intervals) == 0:
            return False, 0
        
        return True, max(intervals)

    # ---------- Detect CI (column-intervals) ----------
    def detect_CI(arr):
        intervals = []
        for col in range(arr.shape[1]):
            nz = np.where(arr[col] != 0)[0]
            if len(nz) <= 1:
                continue

            dif = np.diff(nz)
            if np.all(dif == dif[0]) and dif[0] > 0:
                intervals.append(dif[0])

            else:
                
                return False, 0
        if len(intervals) == 0:
            return False, 0
        return True, max(intervals)

    is_RI, ri_int = detect_RI(arr)
    is_CI, ci_int = detect_CI(arr)

    # ---------- Determine style ----------
    if is_RI and is_CI:
        style = "rci"
        interval = max(ri_int, ci_int)
    elif is_RI:
        style = "ri"
        interval = ri_int
    elif is_CI:
        style = "ci"
        interval = ci_int
    else:
        style = None
        interval = 0

    is_izi = style is not None

    # ---------- Output ----------
    if return_style and return_intervals:
        return is_izi, style, interval
    if return_style:
        return is_izi, style
    if return_intervals:
        return is_izi, interval
    return is_izi

import numpy as np
from scipy import sparse

def izi_matrix(X: sparse.spmatrix, intervals=2, style="ri", dtype=None) -> sparse.spmatrix:
    """
    Transform matrix into IZI (Included Zero-Intervals) format.

    ## Args:
        **X**: *sparse.spmatrix*
            Input sparse matrix.

        **intervals**: *int*
            Interval spacing.

        **style**: *str*
            "ri", "ci", or "rci".

    ## Returns:
        **sparse.spmatrix**: IZI-transformed matrix.

    ## Raises:
        **ValueError**: If intervals < 1 or style is invalid.
    """
    if intervals < 1:
        raise ValueError("intervals must be >= 1")

    # normalize to COO for consistent indexing
    if not sparse.issparse(X):
        X = sparse.coo_matrix(X, copy=False)
    else:
        X = X.tocoo(copy=False)

    r, c = X.row, X.col
    data = X.data
    n_rows, n_cols = X.shape

    # ------------------- Row Interval (RI) -------------------
    if style == "ri":
        new_r = r * intervals
        new_c = c
        shape = (n_rows * intervals, n_cols)

    # ------------------- Column Interval (CI) -------------------
    elif style == "ci":
        new_r = r
        new_c = c * intervals
        shape = (n_rows, n_cols * intervals)

    # ------------------- R-CI (both) -------------------
    elif style == "rci":
        new_r = r * intervals
        new_c = c * intervals
        shape = (n_rows * intervals, n_cols * intervals)

    else:
        raise ValueError("style must be 'ri', 'ci', or 'rci'")

    out = sparse.coo_matrix((data, (new_r, new_c)), shape=shape, dtype=dtype or data.dtype)

    # choose best storage
    return out.tocsr() if shape[0] >= shape[1] else out.tocsc()
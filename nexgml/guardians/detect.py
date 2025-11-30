import numpy as np
from scipy import sparse

def hasinf(arr: np.ndarray | list | sparse.spmatrix) -> bool:
    """Check if there's an infinity value in an array, list, or sparse matrix.
    
    ## Args:
        **arr**: *np.ndarray* or *list* or *spmatrix*
        Input array to be processed.

    ## Returns:
        **bool**: *Condition, if there's an infinity value will return True, if not then return False.*

    ## Raises:
        **None**

    ## Notes:
      This function only tell if there's an infinity value in a matrix, not returning bool mask.

    ## Usage Example:
    ```python
    >>> X = [[nan, 0.9, inf, -8],
             [1, 2, 6, 6]]
    >>> has_inf = hasinf(arr=X)
    >>>
    >>> print("Has infinity:", has_inf)
    ```
    """
    arr = np.asarray(arr)
    stat = np.any(np.isinf(arr))

    return stat

def hasnan(arr: np.ndarray | list | sparse.spmatrix) -> bool:
    """Check if there's a NaN in an array, list, or sparse matrix.
    
    ## Args:
        **arr**: *np.ndarray* or *list* or *spmatrix*
        Input array to be processed.

    ## Returns:
        **bool**: *Condition, if there's a NaN will return True, if not then return False.*

    ## Raises:
        **None**

    ## Notes:
      This function only tell if there's a Nan in a matrix, not returning bool mask.

    ## Usage Example:
    ```python
    >>> X = [[nan, 0.9, inf, -8],
             [1, 2, 6, 6]]
    >>> has_nan = hasnan(arr=X)
    >>>
    >>> print("Has nan:", has_nan)
    ```
    """
    arr = np.asarray(arr)
    stat = np.any(np.isnan(arr))

    return stat
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
    """
    arr = np.asarray(arr)
    stat = np.any(np.isnan(arr))

    return stat
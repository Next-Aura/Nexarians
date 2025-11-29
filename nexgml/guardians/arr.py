import numpy as np
from warnings import warn
from scipy import sparse

def safe_array(arr: np.ndarray, max_value: float=1e10, min_value: float=-1e10) -> np.ndarray:
    """Safely convert array to finite numbers within specified bounds.
    
    ## Args:
        **arr**: *np.ndarray* 
        Input array to be processed.

        **max_value**: *float* 
        Maximum allowable value in the array.

        **min_value**: *float* 
        Minimum allowable value in the array.

    ## Returns:
        **np.ndarray**: *Processed array with values clipped to the specified bounds.*

    ## Raises:
        **RuntimeWarning**: *Warns if any values were clipped due to overflow.*
    """
    # Replace NaN and inf with finite numbers
    arr = np.asarray(arr)
    arr = np.nan_to_num(arr, nan=0.0, posinf=max_value, neginf=min_value)
    # Clip values to avoid extreme overflow
    arr = np.clip(arr, min_value, max_value)
    if np.any(arr == min_value) or np.any(arr == max_value) or np.any(arr == 0.0):
        warn("There's NaN or infinity value.", RuntimeWarning)

    return arr

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
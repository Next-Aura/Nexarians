import numpy as np
from warnings import warn

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

    ## Notes:
      This function is specified for 1D array.

    ## Usage Example:
    ```python
    >>> X = [nan, 1e10, nan, -inf]
    >>> safe_one = safe_array(arr=X, max_value=1e12, min_value=-1e12)
    >>>
    >>> print("Safe array:", safe_one)
    ```
    """
    # Replace NaN and inf with finite numbers
    arr = np.asarray(arr)
    arr = np.nan_to_num(arr, nan=0.0, posinf=max_value, neginf=min_value)
    # Clip values to avoid extreme overflow
    arr = np.clip(arr, min_value, max_value)
    if np.any(arr == min_value) or np.any(arr == max_value) or np.any(arr == 0.0):
        warn("There's NaN or infinity value.", RuntimeWarning)

    return arr
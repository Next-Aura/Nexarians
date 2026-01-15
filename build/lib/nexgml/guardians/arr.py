import numpy as np
from warnings import warn
from scipy.sparse import issparse, spmatrix
from nexgml import guardians as grd

def safe_array(arr: np.ndarray | spmatrix, min_value: float=-1e10, max_value: float=1e10, clip: bool=True, warn_me: bool=True, dtype: np.float32=np.float32) -> np.ndarray:
    """Safely convert array to finite numbers within specified bounds.
    
    ## Args:
        **arr**: *np.ndarray* or *spmatrix*
        Input array to be processed.

        **min_value**: *float* 
        Minimum allowable value in the array.

        **max_value**: *float* 
        Maximum allowable value in the array.

        **clip**: *bool*
        If true will clip array value that greater than max_value or less than min_value.

        **warn_me**: *bool*
        If True, throw warning when there's an invalid value.

        **dtype**: *np.float32*
        Data type output.

    ## Returns:
        **np.ndarray**: *Processed array with values clipped to the specified bounds.*

    ## Raises:
        **RuntimeWarning**: *Warns if any values were clipped due to overflow.*

    ## Notes:
      This function is specified for 1D array.

    ## Usage Example:
    ```python
    >>> X = np.array([nan, 1e10, nan, -inf])
    >>> safe_one = safe_array(arr=X, max_value=1e12, min_value=-1e12)
    >>>
    >>> print("Safe array:", safe_one)
    >>> # print: 'warn("There's NaN or infinity value.", RuntimeWarning)'
    >>> #        'Safe array: [ 0.e+00  1.e+10  0.e+00 -1.e+12]'
    ```
    """
    # Replace NaN and inf with finite numbers
    if issparse(arr):
       arr = arr.astype(dtype)

    else:
      arr = np.asarray(arr, dtype=dtype)
    
    if bool(warn_me):
        if np.any(np.isnan(arr)):
            warn("There's NaN(s) during the process.", RuntimeWarning)

        if not np.all(np.isfinite(arr)):
            warn("There's infinity value(s) during the process.", RuntimeWarning)

    if not np.all(np.isfinite(arr)) or np.any(np.isnan(arr)):
      arr = np.nan_to_num(arr, nan=0.0, posinf=max_value, neginf=min_value)
    
    if clip:
      # Clip values to avoid extreme overflow
      arr = np.clip(arr, min_value, max_value, dtype=dtype)

    return arr

def issafe_array(arr: np.ndarray | spmatrix) -> bool:
  """
  Check if array is safe for numerical operation.

  ## Args:
    **arr**: *np.ndarray* or *spmatrix*
    Input array to be processed.

  ## Returns:
    **bool**: *Array's safety condition*

  ## Raises:
    **None**

  ## Notes:
    This function is specified for 1D array.

  ## Usage Example
  ```python
  >>> arr = np.array([1, 0, 7, 1.9, 0.6, np.nan])
  >>> print(f"Is safe: {issafe_array(arr)}")
  >>> # print: 'Is safe: False'
  ```
  """
  if grd.hasnan(arr) or grd.hasinf(arr):
    return False
  
  if arr.size == 0:
    return False
  
  return True
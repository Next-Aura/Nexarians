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
    >>> # print: 'Has infinity: True'
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
    >>> # print: 'Has nan: True'
    ```
    """
    arr = np.asarray(arr)
    stat = np.any(np.isnan(arr))

    return stat

def iscontinuous(a: np.ndarray | list | sparse.spmatrix) -> bool:
    """Check if the array values are continuous.

    ## Args:
        **a**: *np.ndarray* or *list* or *spmatrix*
        Input array to be processed.

    ## Returns:
        **bool**: *True if the array is continuous (float or complex dtype), False if discrete (integer or other).*

    ## Raises:
        **None**

    ## Notes:
      Continuous data typically has float or complex dtypes, discrete has integer or other dtypes.

    ## Usage Example:
    ```python
    >>> X = [1.0, 2.5, 3.7]
    >>> is_cont = iscontinious(a=X)
    >>>
    >>> print("Is continuous:", is_cont)
    >>> # print: 'Is continuous: True'
    ```
    """
    a = np.asarray(a)
    return np.issubdtype(a.dtype, np.floating) or np.issubdtype(a.dtype, np.complexfloating)

def isdiscrete(a: np.ndarray | list | sparse.spmatrix) -> bool:
    """Check if the array values are discrete.

    ## Args:
        **a**: *np.ndarray* or *list* or *spmatrix*
        Input array to be processed.

    ## Returns:
        **bool**: *True if the array is discrete (integer or other), False if continious (float or complex dtype).*

    ## Raises:
        **None**

    ## Notes:
      Discrete data typically has int or other dtypes, continious has float or complex dtypes.

    ## Usage Example:
    ```python
    >>> X = [1, 2, 3]
    >>> is_disc = isdiscrete(a=X)
    >>>
    >>> print("Is Discrete:", is_disc)
    >>> # print: 'Is Discrete: True'
    ```
    """
    a = np.asarray(a)
    return not iscontinuous(a)
import numpy as np  # Numpy for numerical computations
from nexgml.guardians import issafe_array

def lasso(a: np.ndarray, alpha: float, dtype=np.float32) -> float:
    """
    Calculate lasso (L1) penalty.

    ## Args:
        **a**: *np.ndarray*
        Argument that'll be regulazed.

        **alpha**: *float*
        Penalty strength.

        **dtype**: *DTypeLike, default=np.float32*
        Data type output.

    ## Returns:
      **float**: *Calculated loss.*

    ## Raises:
      **ValueError**: *If array data argument has size 0, NaN, or infinity value.*

    ## Notes:
      Calculation is helped by numpy for reaching C-like speed.

    ## Usage Example:
    ```python
    >>> coef = 0.00025
    >>> alpha = 0.0001
    >>>
    >>> penalty = lasso(a=coef, alpha=alpha)
    >>> print("Penalty: ", penalty)
    >>> # Print: 'Penalty:  0.000000025'
    ```
    """
    # Check array safety
    if not issafe_array(a):
       raise ValueError("Array data argument is not safe for numerical operation."
                        "Please check the size or the value if there's NaN or infinity.")

    return dtype(alpha) * np.sum(np.abs(a), dtype=dtype)

def ridge(a: np.ndarray, alpha: float, dtype=np.float32) -> float:
    """
    Calculate ridge (L2) penalty.

    ## Args:
        **a**: *np.ndarray*
        Argument that'll be regulazed.

        **alpha**: *float*
        Penalty strength.

        **dtype**: *DTypeLike, default=np.float32*
        Data type output.

    ## Returns:
      **float**: *Calculated loss.*

    ## Raises:
      **ValueError**: *If array data argument has size 0, NaN, or infinity value.*

    ## Notes:
      Calculation is helped by numpy for reaching C-like speed.

    ## Usage Example:
    ```python
    >>> coef = 0.00025
    >>> alpha = 0.0001
    >>>
    >>> penalty = ridge(a=coef, alpha=alpha)
    >>> print("Penalty: ", penalty)
    >>> # Print: Penalty:  0.00000000000625
    ```
    """
    # Check array safety
    if not issafe_array(a):
       raise ValueError("Array data argument is not safe for numerical operation."
                        "Please check the size or the value if there's NaN or infinity.")

    return dtype(alpha) * np.sum(a**2, dtype=dtype)

def elasticnet(a: np.ndarray, alpha: float, l1_ratio: float=0.5, dtype=np.float32) -> float:
    """
    Calculate elatic net penalty.

    ## Args:
        **a**: *np.ndarray*
        Argument that'll be regulazed.

        **alpha**: *float*
        Penalty strength.

        **l1_ratio**: *float*
        Penalties ratio between L1 and L2.

        **dtype**: *DTypeLike, default=np.float32*
        Data type output.

    ## Returns:
      **float**: *Calculated loss.*

    ## Raises:
      **ValueError**: *If array data argument has size 0, NaN, or infinity value.*

    ## Notes:
      Calculation is helped by numpy for reaching C-like speed.

    ## Usage Example:
    ```python
    >>> coef = 0.00025
    >>> alpha = 0.0001
    >>>
    >>> penalty = elasticnet(a=coef, alpha=alpha, l1_ratio=0.6)
    >>> print("Penalty: ", penalty)
    >>> # Print: 'Penalty:  0.0000000150025'
    ```
    """
    # Check array safety
    if not issafe_array(a):
       raise ValueError("Array data argument is not safe for numerical operation."
                        "Please check the size or the value if there's NaN or infinity.")

    l1_ratio = dtype(l1_ratio)
    # L1 part
    l1 = l1_ratio * np.sum(np.abs(a), dtype=dtype)
    # L2 part
    l2 = (dtype(1) - l1_ratio) * np.sum(a**2, dtype=dtype)
    # Total with alpha as regulation strength
    penalty = dtype(alpha) * (l1 + l2)
    return penalty
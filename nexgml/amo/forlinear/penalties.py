import numpy as np  # Numpy for numerical computations

def lasso(a: np.ndarray, alpha: float) -> float:
    """
    Calculate lasso (L1) penalty.

    ## Args:
        **a**: *np.ndarray*
        Argument that'll be regulazed.

        **alpha**: *float*
        Penalty strength.

    ## Returns:
      **float**: *Calculated loss.*

    ## Returns:
      **None**

    ## Notes:
      Calculation is helped by numpy for reaching C-like speed.

    ## Usage Example:
    ```python
    >>> coef = 0.00025
    >>> alpha = 0.0001
    >>>
    >>> penalty = lasso(a=coef, alpha=alpha)
    ```
    """
    return alpha * np.sum(np.abs(a))

def ridge(a: np.ndarray, alpha: float) -> float:
    """
    Calculate ridge (L1) penalty.

    ## Args:
        **a**: *np.ndarray*
        Argument that'll be regulazed.

        **alpha**: *float*
        Penalty strength.

    ## Returns:
      **float**: *Calculated loss.*

    ## Returns:
      **None**

    ## Notes:
      Calculation is helped by numpy for reaching C-like speed.

    ## Usage Example:
    ```python
    >>> coef = 0.00025
    >>> alpha = 0.0001
    >>>
    >>> penalty = ridge(a=coef, alpha=alpha)
    ```
    """
    return alpha * np.sum(a**2)

def elasticnet(a: np.ndarray, alpha: float, l1_ratio: float=0.5) -> float:
    """
    Calculate elatic net penalty.

    ## Args:
        **a**: *np.ndarray*
        Argument that'll be regulazed.

        **alpha**: *float*
        Penalty strength.

        **l1_ratio**: *float*
        Penalties ratio between L1 and L2.

    ## Returns:
      **float**: *Calculated loss.*

    ## Returns:
      **None**

    ## Notes:
      Calculation is helped by numpy for reaching C-like speed.

    ## Usage Example:
    ```python
    >>> coef = 0.00025
    >>> alpha = 0.0001
    >>>
    >>> penalty = elasticnet(a=coef, alpha=alpha, l1_ratio=0.6)
    ```
    """
    # L1 part
    l1 = l1_ratio * np.sum(np.abs(a))
    # L2 part
    l2 = (1 - l1_ratio) * np.sum(a**2)
    # Total with alpha as regulation strength
    penalty = alpha * (l1 + l2)
    return penalty
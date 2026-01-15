import numpy as np                          # Numpy for numerical computations
from nexgml.guardians import safe_array, issafe_array  # For numerical stability

def categorical_ce(y_true: np.ndarray, y_pred_proba: np.ndarray, weighting: bool=True, mean: bool=True, dtype=np.float32, epsilon: float=1e-8) -> np.ndarray | float:
    """
    Calculate classification loss using categorical cross-entropy formula.

    ## Args:
        **y_true**: *np.ndarray*
        True labels data.

        **y_pred_proba**: *np.ndarray*
        Labels prediction probability.

        **weighting**: *bool, default=True*
        Class weighting for data with imbalance class.

        **mean**: *bool, default=True*
        Return loss mean or not.

        **dtype**: *DTypeLike, default=np.float32*
        Data type output.

        **epsilon**: *float*
        Small value for numerical stability.

    ## Returns:
        **np.ndarray** or **float**: *Labels prediction probability loss. 
        If mean is True will return float, and if not then will return np.ndarray*

    ## Raises:
      **ValueError**: *If y_true or y_pred_proba data has size 0, NaN, or infinity value.*

    ## Notes:
      Calculation is helped by numpy for reaching C-like speed.

    ## Usage Example:
    ```python
    >>> pred_proba = X @ coef + bias
    >>>
    >>> loss = categorical_ce(y_true=y, y_pred_proba=pred_proba, mean=True, epsilon=1e-10)
    ```
    """
    # Check array safety
    if not issafe_array(y_true) or not issafe_array(y_pred_proba):
       raise ValueError("y_true or y_pred_proba data is not safe for numerical operation."
                        "Please check the size or the value if there's NaN or infinity.")

    y_true, y_pred_proba = np.asarray(y_true, dtype=dtype), np.asarray(y_pred_proba, dtype=dtype)
    epsilon = dtype(epsilon)
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon, dtype=dtype)

    class_counts = np.sum(y_true, axis=0, dtype=dtype)
    n_classes = len(class_counts)
    total = np.sum(class_counts, dtype=dtype)
    class_weights = total / (n_classes * (class_counts + epsilon))

    if np.sum(class_weights, dtype=dtype) == 0 or not weighting:
        class_weights = np.ones_like(class_weights, dtype=dtype)

    else:
        class_weights = class_weights / np.sum(class_weights, dtype=dtype)

    loss = safe_array(-np.sum(class_weights * y_true * np.log(y_pred_proba), axis=1), dtype=dtype)

    if mean:
        return np.mean(loss, dtype=dtype)
    
    else:
        return loss

def binary_ce(y_true: np.ndarray, y_pred_proba: np.ndarray, mean: bool=True, dtype=np.float32, epsilon: float=1e-8) -> np.ndarray | float:
    """
    Calculate classification loss using binary cross-entropy formula.

    ## Args:
        **y_true**: *np.ndarray*
        True labels data.

        **y_pred_proba**: *np.ndarray*
        Labels prediction probability.

        **mean**: *bool, default=True*
        Return loss mean or not.

        **dtype**: *DTypeLike, default=np.float32*
        Data type output.

        **epsilon**: *float*
        Small value for numerical stability.

    ## Returns:
        **np.ndarray** or **float**: *Labels prediction probability loss. 
        If mean is True will return float, and if not then will return np.ndarray*

    ## Raises:
      **ValueError**: *If y_true or y_pred_proba data has size 0, NaN, or infinity value.*

    ## Notes:
      Calculation is helped by numpy for reaching C-like speed.

    ## Usage Example:
    ```python
    >>> pred_proba = X @ coef + bias
    >>>
    >>> loss = binary_ce(y_true=y, y_pred_proba=pred_proba, mean=True, epsilon=1e-10)
    ```
    """
    # Check array safety
    if not issafe_array(y_true) or not issafe_array(y_pred_proba):
       raise ValueError("y_true or y_pred_proba data is not safe for numerical operation."
                        "Please check the size or the value if there's NaN or infinity.")

    epsilon = dtype(epsilon)
    y_pred_clip = np.clip(y_pred_proba, epsilon, 1 - epsilon, dtype=dtype)

    loss = safe_array(-(y_true * np.log(y_pred_clip) + (1 - y_true) * np.log(1 - y_pred_clip)), dtype=dtype)

    if mean:
        return np.mean(loss, dtype=dtype)
    
    else:
        return loss
    
def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray, dtype=np.float32) -> float:
    """
    Calculate regression loss using mean squared error (MSE) formula.

    ## Args:
        **y_true**: *np.ndarray*
        True target data.

        **y_pred**: *np.ndarray*
        Target prediction.

        **dtype**: *DTypeLike, default=np.float32*
        Data type output.

    ## Returns:
        **float**: *Target prediction loss.*

    ## Raises:
      **ValueError**: *If y_true or y_pred data has size 0, NaN, or infinity value.*

    ## Notes:
      Calculation is helped by numpy for reaching C-like speed.

    ## Usage Example:
    ```python
    >>> pred = X @ coef + bias
    >>>
    >>> loss = mean_squared_error(y_true=y, y_pred=pred)
    ```
    """
    # Check array safety
    if not issafe_array(y_true) or not issafe_array(y_pred):
       raise ValueError("y_true or y_pred data is not safe for numerical operation."
                        "Please check the size or the value if there's NaN or infinity.")

    return safe_array(np.mean((y_true - y_pred)**2, dtype=dtype), dtype=dtype)

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray, dtype=np.float32) -> float:
    """
    Calculate regression loss using mean absolute error (MAE) formula.

    ## Args:
        **y_true**: *np.ndarray*
        True target data.

        **y_pred**: *np.ndarray*
        Target prediction.

        **dtype**: *DTypeLike, default=np.float32*
        Data type output.

    ## Returns:
        **float**: *Target prediction loss.*

    ## Raises:
      **ValueError**: *If y_true or y_pred data has size 0, NaN, or infinity value.*

    ## Notes:
      Calculation is helped by numpy for reaching C-like speed.

    ## Usage Example:
    ```python
    >>> pred = X @ coef + bias
    >>>
    >>> loss = mean_absolute_error(y_true=y, y_pred=pred)
    ```
    """
    # Check array safety
    if not issafe_array(y_true) or not issafe_array(y_pred):
       raise ValueError("y_true or y_pred data is not safe for numerical operation."
                        "Please check the size or the value if there's NaN or infinity.")

    return safe_array(np.mean(np.abs(y_true - y_pred), dtype=dtype), dtype=dtype)

def smoothl1(y_true: np.ndarray, y_pred: np.ndarray, delta: float=1.0, dtype=np.float32) -> float:
    """
    Calculate regression loss using smooth L1 (huber) loss formula.

    ## Args:
        **y_true**: *np.ndarray*
        True target data.

        **y_pred**: *np.ndarray*
        Target prediction.
        
        **delta**: *float, default=1.0*
        Function threshold between operation.

        **dtype**: *DTypeLike, default=np.float32*
        Data type output.

    ## Returns:
        **float**: *Target prediction loss.*

    ## Raises:
      **ValueError**: *If y_true or y_pred data has size 0, NaN, or infinity value.*

    ## Notes:
      Calculation is helped by numpy for reaching C-like speed.

    ## Usage Example:
    ```python
    >>> pred = X @ coef + bias
    >>>
    >>> loss = smoothl1_loss(y_true=y, y_pred=pred, delta=1.0)
    ```
    """
    # Check array safety
    if not issafe_array(y_true) or not issafe_array(y_pred):
       raise ValueError("y_true or y_pred data is not safe for numerical operation."
                        "Please check the size or the value if there's NaN or infinity.")

    diff = np.abs(y_true - y_pred)
    loss = np.where(diff < delta, 0.5 * diff**2 / delta, diff - 0.5 * delta)

    return safe_array(np.mean(loss, dtype=dtype), dtype=dtype)
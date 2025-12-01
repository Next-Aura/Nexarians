import numpy as np  # Numpy for numerical computations

def categorical_ce(y_true: np.ndarray, y_pred_proba: np.ndarray, mean: bool=True, epsilon: float=1e-8) -> np.ndarray:
    """
    Calculate classification loss using categorical cross-entropy formula.

    ## Args:
        **y_true**: *np.ndarray*
        True labels data.

        **y_pred_proba**: *np.ndarray*
        Labels prediction probability.

        **mean**: *bool, default=True*
        Return loss mean or not.

        **epsilon**: *float*
        Small value for numerical stability.

    ## Returns:
        **np.ndarray**: *Labels prediction probability loss.*

    ## Raises:
        **None**

    ## Notes:
      Calculation is helped by numpy for reaching C-like speed.

    ## Usage Example:
    ```python
    >>> pred_proba = X @ coef + bias
    >>>
    >>> loss = categorical_ce(y_true=y, y_pred_proba=pred_proba, mean=True, epsilon=1e-10)
    ```
    """
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)

    class_counts = np.sum(y_true, axis=0)
    n_classes = len(class_counts)
    total = np.sum(class_counts)
    class_weights = total / (n_classes * class_counts + 1e-8)

    if np.sum(class_weights) == 0:
        class_weights = np.ones_like(class_weights)

    else:
        class_weights = class_weights / np.sum(class_weights)

    loss = -np.sum(class_weights * y_true * np.log(y_pred_proba), axis=1)

    if mean:
        return np.mean(loss)
    
    else:
        return loss

def binary_ce(y_true: np.ndarray, y_pred_proba: np.ndarray, mean: bool=True, epsilon: float=1e-8) -> np.ndarray:
    """
    Calculate classification loss using binary cross-entropy formula.

    ## Args:
        **y_true**: *np.ndarray*
        True labels data.

        **y_pred_proba**: *np.ndarray*
        Labels prediction probability.

        **mean**: *bool, default=True*
        Return loss mean or not.

        **epsilon**: *float*
        Small value for numerical stability.

    ## Returns:
        **np.ndarray**: *Labels prediction probability loss.*

    ## Raises:
        **None**

    ## Notes:
      Calculation is helped by numpy for reaching C-like speed.

    ## Usage Example:
    ```python
    >>> pred_proba = X @ coef + bias
    >>>
    >>> loss = binary_ce(y_true=y, y_pred_proba=pred_proba, mean=True, epsilon=1e-10)
    ```
    """
    y_pred_clip = np.clip(y_pred_proba, epsilon, 1 - epsilon)

    loss = -(y_true * np.log(y_pred_clip) + (1 - y_true) * np.log(1 - y_pred_clip))

    if mean:
        return np.mean(loss)
    
    else:
        return loss
    
def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate regression loss using mean squared error (MSE) formula.

    ## Args:
        **y_true**: *np.ndarray*
        True target data.

        **y_pred**: *np.ndarray*
        Target prediction.

    ## Returns:
        **float**: *Target prediction loss.*

    ## Raises:
        **None**

    ## Notes:
      Calculation is helped by numpy for reaching C-like speed.

    ## Usage Example:
    ```python
    >>> pred = X @ coef + bias
    >>>
    >>> loss = mean_squared_error(y_true=y, y_pred=pred)
    ```
    """
    return np.mean((y_true - y_pred)**2)

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate regression loss using mean absolute error (MAE) formula.

    ## Args:
        **y_true**: *np.ndarray*
        True target data.

        **y_pred**: *np.ndarray*
        Target prediction.

    ## Returns:
        **float**: *Target prediction loss.*

    ## Raises:
        **None**

    ## Notes:
      Calculation is helped by numpy for reaching C-like speed.

    ## Usage Example:
    ```python
    >>> pred = X @ coef + bias
    >>>
    >>> loss = mean_absolute_error(y_true=y, y_pred=pred)
    ```
    """
    return np.mean(np.abs(y_true - y_pred))

def root_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate regression loss using root mean squared error (RMSE) formula.

    ## Args:
        **y_true**: *np.ndarray*
        True target data.

        **y_pred**: *np.ndarray*
        Target prediction.

    ## Returns:
        **float**: *Target prediction loss.*

    ## Raises:
        **None**

    ## Notes:
      Calculation is helped by numpy for reaching C-like speed.

    ## Usage Example:
    ```python
    >>> pred = X @ coef + bias
    >>>
    >>> loss = root_squared_error(y_true=y, y_pred=pred)
    ```
    """
    return np.sqrt(np.mean((y_true - y_pred)**2))

def smoothl1_loss(y_true: np.ndarray, y_pred: np.ndarray, delta: float=0.5) -> float:
    """
    Calculate regression loss using smooth L1 (huber) loss formula.

    ## Args:
        **y_true**: *np.ndarray*
        True target data.

        **y_pred**: *np.ndarray*
        Target prediction.
        
        **delta**: *float*
        Function threshold between operation

    ## Returns:
        **float**: *Target prediction loss.*

    ## Raises:
        **None**

    ## Notes:
      Calculation is helped by numpy for reaching C-like speed.

    ## Usage Example:
    ```python
    >>> pred = X @ coef + bias
    >>>
    >>> loss = smoothl1_loss(y_true=y, y_pred=pred, delta=1.0)
    ```
    """
    diff = np.abs(y_true - y_pred)
    loss = np.where(diff < delta, 0.5 * diff**2 / delta, diff - 0.5 * delta)

    return np.mean(loss)
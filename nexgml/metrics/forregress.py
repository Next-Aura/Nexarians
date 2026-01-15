import numpy as np
from nexgml.guardians import iscontinuous
from nexgml.guardians import issafe_array, safe_array

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the R² (coefficient of determination) regression score function.

    ## Args:
        **y_true**: *np.ndarray*
        True target values.

        **y_pred**: *np.ndarray*
        Predicted target values.

    ## Returns:
        **float**: *R² score.*

    ## Raises:
        **ValueError**: *If label data (y_true) is not continious.*

    ## Notes:
      This function only for regressor models.

    ## Usage Example:
    ```python
    >>> pred = model.predict(X_test)
    >>> r2 = r2_score(y_true=y_test, y_pred=pred)
    >>>
    >>> print("Model's R² score:", r2)
    ```
    """
    if not iscontinuous(y_true):
        raise ValueError("R^2 score only calculate continious label loss.")
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    return 1 - (ss_res / ss_tot)

def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray, dtype=np.float32) -> float:
    """
    Calculate regression loss using root mean squared error (RMSE) formula.

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
    >>> loss = root_squared_error(y_true=y, y_pred=pred)
    ```
    """
    # Check array safety
    if not issafe_array(y_true) or not issafe_array(y_pred):
       raise ValueError("y_true or y_pred data is not safe for numerical operation."
                        "Please check the size or the value if there's NaN or infinity.")

    return safe_array(np.sqrt(np.mean((y_true - y_pred)**2, dtype=dtype)), dtype=dtype)

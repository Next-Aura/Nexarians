import numpy as np                           # Numpy for numerical computations
from nexgml.guardians import safe_array      # For numerical stability
from scipy.sparse import issparse, spmatrix  # For sparse data handling

def mse_deriv(X: np.ndarray | spmatrix, residual: np.ndarray, intercept: bool, dtype: np.float32=np.float32) -> tuple[np.ndarray, np.float32]:
    """
    Calculate Mean Squared Error (MSE) loss function derivative.

    ## Args:
      **X**: *np.ndarray* or *spmatrix*
      The data for formula calculation.

      **residual**: *np.ndarray*
      Residual data.

      **intercept**: *bool*
      Intercept flag, if true the function will also calculate grad w.r.t bias.

      **dtype**: *DTypeLike, default=np.float32*
        Data type output.

    ## Returns:
      **tuple**: *np.ndarray, np.float32*
      gradient w.r.t weight, gradient w.r.t bias.

    ## Raises:
      **None**

    ## Notes:
      Calculation is helped by numpy for reaching C-like speed.

    ## Usage Example:
    ```python
    >>> pred = X @ coef + bias
    >>> residual = pred - y
    >>> grad_w, grad_b = mse_deriv(X=X, residual=residual, intercept=True)
    ```
    """
    # X and residual as as array of float32
    if issparse(X) or issparse(residual):
      X, residual = X.astype(dtype), residual.astype(dtype)

    else:
      X, residual = np.asarray(X, dtype=dtype), np.asarray(residual, dtype=dtype)    # Initialize bias gradient
    grad_b = dtype(0.0)
    # Gradient w.r.t w calculation
    grad_w = safe_array(X.T @ (2 * residual) / X.shape[0])
    # Calculate bias gradient if intercept is used
    if bool(intercept):
        grad_b = safe_array(np.mean(2 * residual, dtype=dtype))

    return grad_w, grad_b

def rmse_deriv(X: np.ndarray | spmatrix, residual: np.ndarray, intercept: bool, dtype: np.float32=np.float32) -> tuple[np.ndarray, np.float32]:
    """
    Calculate Root Mean Squared Error (RMSE) loss function derivative.

    ## Args:
      **X**: *np.ndarray* or *spmatrix*
      The data for formula calculation.

      **residual**: *np.ndarray*
      Residual data.

      **intercept**: *bool*
      Intercept flag, if true the function will also calculate grad w.r.t bias.

      **dtype**: *DTypeLike, default=np.float32*
        Data type output.

    ## Returns:
      **tuple**: *np.ndarray, np.float32*
      gradient w.r.t weight, gradient w.r.t bias.

    ## Raises:
      **None**

    ## Notes:
      Calculation is helped by numpy for reaching C-like speed.

    ## Usage Example:
    ```python
    >>> pred = X @ coef + bias
    >>> residual = pred - y
    >>> grad_w, grad_b = rmse_deriv(X=X, residual=residual, intercept=True)
    ```
    """
    # X and residual as as array of float32
    if issparse(X) or issparse(residual):
      X, residual = X.astype(dtype), residual.astype(dtype)

    else:
      X, residual = np.asarray(X, dtype=dtype), np.asarray(residual, dtype=dtype)    # Initialize gradient w.r.t bias
    grad_b = dtype(0.0)
    # RMSE part
    rmse = np.sqrt(np.mean(residual**2, dtype=dtype))
    # Gradient w.r.t w calculation
    grad_w = safe_array((X.T @ (2 * residual)) / (X.shape[0] * rmse + 1e-10))
    # Calculate bias gradient if intercept is used
    if bool(intercept):
        grad_b = safe_array(np.mean(2 * residual, dtype=dtype) / (rmse + 1e-10))

    return grad_w, grad_b

def mae_deriv(X: np.ndarray | spmatrix, residual: np.ndarray, intercept: bool, dtype: np.float32=np.float32) -> tuple[np.ndarray, np.float32]:
    """
    Calculate Mean Absolute Error (MAE) loss function derivative.

    ## Args:
      **X**: *np.ndarray* or *spmatrix*
      The data for formula calculation.

      **residual**: *np.ndarray*
      Residual data.

      **intercept**: *bool*
      Intercept flag, if true the function will also calculate grad w.r.t bias.

      **dtype**: *DTypeLike, default=np.float32*
        Data type output.

    ## Returns:
      **tuple**: *np.ndarray, np.float32*
      gradient w.r.t weight, gradient w.r.t bias.

    ## Raises:
      **None**

    ## Notes:
      Calculation is helped by numpy for reaching C-like speed.

    ## Usage Example:
    ```python
    >>> pred = X @ coef + bias
    >>> residual = pred - y
    >>> grad_w, grad_b = mae_deriv(X=X, residual=residual, intercept=True)
    ```
    """
    # X and residual as as array of float32
    if issparse(X) or issparse(residual):
      X, residual = X.astype(dtype), residual.astype(dtype)

    else:
      X, residual = np.asarray(X, dtype=dtype), np.asarray(residual, dtype=dtype)    # Initialize gradient w.r.t bias
    grad_b = dtype(0.0)
    # Gradient w.r.t w calculation
    grad_w = safe_array(X.T @ np.sign(residual) / X.shape[0])
    # Calculate bias gradient if intercept is used
    if bool(intercept):
      grad_b = safe_array(np.mean(np.sign(residual), dtype=dtype))

    return grad_w, grad_b

def smoothl1_deriv(X: np.ndarray | spmatrix, residual: np.ndarray, intercept: bool, delta: float=0.5, dtype: np.float32=np.float32) -> tuple[np.ndarray, np.float32]:
    """
    Calculate Smooth L1 (Huber) loss function derivative.

    ## Args:
      **X**: *np.ndarray* or *spmatrix*
      The data for formula calculation.

      **residual**: *np.ndarray*
      Residual data.

      **intercept**: *bool*
      Intercept flag, if true the function will also calculate grad w.r.t bias.

      **delta**: *float*
      Threshold between 2 conditions in the calculation.

      **dtype**: *DTypeLike, default=np.float32*
        Data type output.

    ## Returns:
      **tuple**: *np.ndarray, np.float32*
      gradient w.r.t weight, gradient w.r.t bias.

    ## Raises:
      **None**

    ## Notes:
      Calculation is helped by numpy for reaching C-like speed.

    ## Usage Example:
    ```python
    >>> pred = X @ coef + bias
    >>> residual = pred - y
    >>> grad_w, grad_b = smoothl1_deriv(X=X, residual=residual, intercept=True, delta=0.8)
    ```
    """
    # X and residual as as array of float32
    if issparse(X) or issparse(residual):
      X, residual = X.astype(dtype), residual.astype(dtype)

    else:
      X, residual = np.asarray(X, dtype=dtype), np.asarray(residual, dtype=dtype)    
    
    delta = dtype(delta)
    # Initialize gradient w.r.t bias
    grad_b = dtype(0.0)
    # Gradient w.r.t w calculation
    grad_w = safe_array(X.T @ np.where(np.abs(residual) <= delta,
                            residual,
                            delta * np.sign(residual)
                            ) / X.shape[0])

    # Calculate bias gradient if intercept is used
    if bool(intercept):
        grad_b = safe_array(np.mean(
            np.where(np.abs(residual) <= delta,
                    residual,
                    delta * np.sign(residual))
                    , dtype=dtype))

    return grad_w, grad_b

def cce_deriv(X: np.ndarray | spmatrix, residual: np.ndarray, intercept: bool, n_classes: int, dtype: np.float32=np.float32) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate Categorical Cross-entropy (CCE) loss function derivative.

    ## Args:
      **X**: *np.ndarray* or *spmatrix*
      The data for formula calculation.

      **residual**: *np.ndarray*
      Residual data.

      **intercept**: *bool*
      Intercept flag, if true the function will also calculate grad w.r.t bias.

      **n_classes**: *int*
      Number of class in the data.

      **dtype**: *DTypeLike, default=np.float32*
        Data type output.

    ## Returns:
      **tuple**: *np.ndarray, np.ndarray*
      gradient w.r.t weight, gradient w.r.t bias.

    ## Raises:
      **None**

    ## Notes:
      Calculation is helped by numpy for reaching C-like speed.

    ## Usage Example:
    ```python
    >>> n_classes = len(np.unique(y))
    >>> pred = X @ coef + bias
    >>> residual = pred - y
    >>> grad_w, grad_b = cce_deriv(X=X, residual=residual, intercept=True, n_classes=n_classes)
    ```
    """
    # X and residual as as array of float32
    if issparse(X) or issparse(residual):
      X, residual = X.astype(dtype), residual.astype(dtype)

    else:
      X, residual = np.asarray(X, dtype=dtype), np.asarray(residual, dtype=dtype)
    # Intialize gradient w.r.t bias
    grad_b = np.zeros((n_classes), dtype=dtype)
    # Gradient w.r.t w calculation
    grad_w = safe_array((X.T @ residual) / X.shape[0])

    # Calculate bias gradient if intercept is used
    if bool(intercept):
        grad_b = safe_array(np.mean(residual, axis=0, dtype=dtype))

    return grad_w, grad_b

def lasso_deriv(a: np.ndarray, alpha: float, dtype: np.float32=np.float32) -> np.ndarray:
    """
    Calculate lasso (L1) penalty.

    ## Args:
        **a**: *np.ndarray* or *spmatrix*
        Argument that'll be regularized.

        **alpha**: *float*
        Penalty strength.

        **dtype**: *DTypeLike, default=np.float32*
        Data type output.

    ## Returns:
      **np.ndarray**: *Calculated penalty.*

    ## Raises:
      **None**

    ## Notes:
      Calculation is helped by numpy for reaching C-like speed.

    ## Usage Example:
    ```python
    >>> coef = 0.00025
    >>> alpha = 0.0001
    >>>
    >>> grad = lasso_deriv(a=coef, alpha=alpha)
    ```
    """
    grad = dtype(alpha) * np.sign(np.asarray(a, dtype=dtype))
    return grad

def ridge_deriv(a: np.ndarray, alpha: float, dtype: np.float32=np.float32) -> np.ndarray:
    """
    Calculate ridge (L2) penalty.

    ## Args:
        **a**: *np.ndarray*
        Argument that'll be regularized.

        **alpha**: *float*
        Penalty strength.

        **dtype**: *DTypeLike, default=np.float32*
        Data type output.

    ## Returns:
      **np.ndarray**: *Calculated penalty.*

    ## Raises:
      **None**

    ## Notes:
      Calculation is helped by numpy for reaching C-like speed.

    ## Usage Example:
    ```python
    >>> coef = 0.00025
    >>> alpha = 0.0001
    >>>
    >>> grad = ridge_deriv(a=coef, alpha=alpha)
    ```
    """
    grad = 2 * dtype(alpha) * np.asarray(a, dtype=dtype)
    return grad

def elasticnet_deriv(a: np.ndarray, alpha: float, dtype: np.float32=np.float32, l1_ratio: float=0.5) -> np.ndarray:
    """
    Calculate elastic net penalty.

    ## Args:
        **a**: *np.ndarray*
        Argument that'll be regularized.

        **alpha**: *float*
        Penalty strength.

        **dtype**: *DTypeLike, default=np.float32*
        Data type output.

        **l1_ratio**: *float*
        Penalties ratio between L1 and L2.

    ## Returns:
      **np.ndarray**: *Calculated penalty.*

    ## Raises:
      **None**

    ## Notes:
      Calculation is helped by numpy for reaching C-like speed.

    ## Usage Example:
    ```python
    >>> coef = 0.00025
    >>> alpha = 0.0001
    >>>
    >>> grad = elasticnet_deriv(a=coef, alpha=alpha, l1_ratio=0.2)
    ```
    """
    a = np.asarray(a, dtype=dtype)
    # L1 part
    l1 = dtype(l1_ratio) * np.sign(a)
    # L2 part
    l2 = 2 * (dtype(1 - l1_ratio) * a)
    # Total with alpha as regulation strength
    grad = dtype(alpha) * (l2 + l1)
    return grad

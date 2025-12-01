import numpy as np  # Numpy for numerical computations

def mse_deriv(X: np.ndarray, residual: np.ndarray, intercept: bool) -> tuple[np.ndarray, float]:
    """
    Calculate Mean Squared Error (MSE) loss function derivative.

    ## Args:
      **X**: *np.ndarray*
      The data for formula calculation.

      **residual**: *np.ndarray*
      Residual data.

      **intercept**: *bool*
      Intercept flag, if true the function will also calculate grad w.r.t bias.

    ## Returns:
      **tuple**: *np.ndarray, float*
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
    # Initialize bias gradient
    grad_b = 0.0
    # Gradient w.r.t w calculation
    grad_w = X.T @ (2 * residual) / X.shape[0]
    # Calculate bias gradient if intercept is used
    if intercept:
        grad_b = np.mean(2 * residual)

    return grad_w, grad_b

def rmse_deriv(X: np.ndarray, residual: np.ndarray, intercept: bool) -> tuple[np.ndarray, float]:
    """
    Calculate Root Mean Squared Error (RMSE) loss function derivative.

    ## Args:
      **X**: *np.ndarray*
      The data for formula calculation.

      **residual**: *np.ndarray*
      Residual data.

      **intercept**: *bool*
      Intercept flag, if true the function will also calculate grad w.r.t bias.

    ## Returns:
      **tuple**: *np.ndarray, float*
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
    # Initialize gradient w.r.t bias
    grad_b = 0.0
    # RMSE part
    rmse = np.sqrt(np.mean(residual**2))
    # Gradient w.r.t w calculation
    grad_w = (X.T @ (2 * residual)) / (X.shape[0] * rmse + 1e-10)
    # Calculate bias gradient if intercept is used
    if intercept:
        grad_b = np.mean(2 * residual) / (rmse + 1e-10)

    return grad_w, grad_b

def mae_deriv(X: np.ndarray, residual: np.ndarray, intercept: bool) -> tuple[np.ndarray, float]:
    """
    Calculate Mean Absolute Error (MAE) loss function derivative.

    ## Args:
      **X**: *np.ndarray*
      The data for formula calculation.

      **residual**: *np.ndarray*
      Residual data.

      **intercept**: *bool*
      Intercept flag, if true the function will also calculate grad w.r.t bias.

    ## Returns:
      **tuple**: *np.ndarray, float*
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
    # Initialize gradient w.r.t bias
    grad_b = 0.0
    # Gradient w.r.t w calculation
    grad_w = X.T @ np.sign(residual) / X.shape[0]
    # Calculate bias gradient if intercept is used
    if intercept:
      grad_b = np.mean(np.sign(residual))

    return grad_w, grad_b

def smoothl1_deriv(X: np.ndarray, residual: np.ndarray, intercept: bool, delta: float=0.5) -> tuple[np.ndarray, float]:
    """
    Calculate Smooth L1 (Huber) loss function derivative.

    ## Args:
      **X**: *np.ndarray*
      The data for formula calculation.

      **residual**: *np.ndarray*
      Residual data.

      **intercept**: *bool*
      Intercept flag, if true the function will also calculate grad w.r.t bias.

      **delta**: *float*
      Threshold between 2 conditions in the calculation.

    ## Returns:
      **tuple**: *np.ndarray, float*
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
    # Initialize gradient w.r.t bias
    grad_b = 0.0
    # Gradient w.r.t w calculation
    grad_w = X.T @ np.where(np.abs(residual) <= delta, 
                            residual, 
                            delta * np.sign(residual)
                            ) / X.shape[0]
    
    # Calculate bias gradient if intercept is used
    if intercept:
        grad_b = np.mean(
            np.where(np.abs(residual) <= delta, 
                    residual, 
                    delta * np.sign(residual))
                    )
        
    return grad_w, grad_b

def cce_deriv(X: np.ndarray, residual: np.ndarray, intercept: bool, n_classes: int) -> tuple[np.ndarray, float]:
    """
    Calculate Categorical Cross-entropy (CCE) loss function derivative.

    ## Args:
      **X**: *np.ndarray*
      The data for formula calculation.

      **residual**: *np.ndarray*
      Residual data.

      **intercept**: *bool*
      Intercept flag, if true the function will also calculate grad w.r.t bias.

      **n_classes**: *int*
      Number of class in the data.

    ## Returns:
      **tuple**: *np.ndarray, float*
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
    # Intialize gradient w.r.t bias
    grad_b = np.zeros(n_classes)
    # Gradient w.r.t w calculation
    grad_w = (X.T @ residual) / X.shape[0]

    # Calculate bias gradient if intercept is used
    if intercept:
        grad_b = np.mean(residual, axis=0)

    return grad_w, grad_b

def lasso_deriv(a: np.ndarray, alpha: float) -> np.ndarray:
    """
    Calculate lasso (L1) penalty.

    ## Args:
        **a**: *np.ndarray*
        Argument that'll be regulazed.

        **alpha**: *float*
        Penalty strength.

    ## Returns:
      **np.ndarray**: *Calculated penalty.*

    ## Returns:
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
    grad = alpha * np.sign(a)
    return grad

def ridge_deriv(a: np.ndarray, alpha: float) -> np.ndarray:
    """
    Calculate ridge (L2) penalty.

    ## Args:
        **a**: *np.ndarray*
        Argument that'll be regulazed.

        **alpha**: *float*
        Penalty strength.

    ## Returns:
      **np.ndarray**: *Calculated penalty.*

    ## Returns:
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
    grad = 2 * alpha * a
    return grad

def elasticnet_deriv(a: np.ndarray, alpha: float, l1_ratio: float) -> np.ndarray:
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
      **np.ndarray**: *Calculated penalty.*

    ## Returns:
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
    # L1 part
    l1 = l1_ratio * np.sign(a)
    # L2 part
    l2 = 2 * ((1 - l1_ratio) * a)
    # Total with alpha as regulation strength
    grad = alpha * (l2 + l1)
    return grad
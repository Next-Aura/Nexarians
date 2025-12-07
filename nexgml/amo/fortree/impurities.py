import numpy as np                       # Numpy for numerical computations
from nexgml.guardians import safe_array  # For numerical stability

def squared_error(y: np.ndarray) -> float:
    """
    Calculate the variance of the given labels (MSE).

    ## Args:
        **y**: *np.ndarray*
        Array of target values.

    ## Returns:
        **float**: *Variance of the labels. Returns 0.0 if labels are empty.*
        
    ## Raises:
        **None**

    ## Notes:
      Calculation is helped by numpy for reaching C-like speed.

    ## Usage Example:
    ```python
    >>> y = np.ndarray([1, 2, 8, 1, 7, 2])
    >>>
    >>> var = squared_error(y)
    >>> print("Variance: ", var)
    >>> # print: 'Variance:  8.25'
    ```
    """
    # Check label's size
    if y.size == 0:
        # If the size is 0, then return 0.0
        return 0.0
    
    # Calculate label mean
    mean = np.mean(y)
    # Calculate label variance (MSE)
    return safe_array(np.mean((y - mean) ** 2))

def friedman_squared_error(y: np.ndarray) -> float:
    """
    Calculate the variance of the given labels (MSE).

    ## Args:
        **y**: *np.ndarray*
        Array of target values.

    ## Returns:
        **float**: *Variance of the labels. Returns 0.0 if labels are empty.*
        
    ## Raises:
        **None**

    ## Notes:
      Calculation is helped by numpy for reaching C-like speed.

    ## Usage Example:
    ```python
    >>> y = np.array([1, 2, 9, 1, 3, 2])
    >>>
    >>> var = friedman_squared_error(y)
    >>> print("Variance: ", var)
    >>> # print: 'Variance:  9.2'
    ```
    """
    n = y.size

    if n <= 1:
        return 0.0
    
    mean = safe_array(np.mean((y - np.mean(y))**2))

    return mean * (n / (n - 1))

def absolute_error(y: np.ndarray) -> float:
    """
    Calculate the mean absolute error of the given labels.

    ## Args:
        **y**: *np.ndarray*
        Array of target values.

    ## Returns:
        **float**: *Mean absolute error of the labels. Returns 0.0 if labels are empty.*
        
    ## Raises:
        **None**

    ## Notes:
      Calculation is helped by numpy for reaching C-like speed.

    ## Usage Example:
    ```python
    >>> y = np.array([1, 2, 9, 1, 3, 2])
    >>>
    >>> var = absolute_error(y)
    >>> print("Variance: ", var)
    >>> # print: 'Variance:  2.0'
    ```
    """
    # Check labels array size
    if y.size == 0:
        # If the size is 0, then return 0.0
        return 0.0
    
    # Calculate labels mean
    mean = np.mean(y)
    # Calculate label variance (absolute error)
    return np.mean(np.abs(y - mean))

def poisson_deviance(y: np.ndarray) -> float:
    """
    Calculate the Poisson deviance of the given labels.

    ## Args:
        **y**: *np.ndarray*
        Array of target values.

    ## Returns:
        **float**: *Poisson deviance of the labels.*

    ## Raises:
        **ValueError**: *If target values are negative.*

    ## Notes:
      Calculation is helped by numpy for reaching C-like speed.

    ## Usage Example:
    ```python
    >>> y = np.array([1, 2, 9, 1, 3, 2])
    >>>
    >>> var = poisson_deviance(y)
    >>> print("Variance: ", var)
    >>> # print: 'Variance:  12.13685117648822'
    ```
    """
    # Check labels array size
    if y.size == 0:
        # If the size is 0, then return 0.0
        return 0.0
    
    # Check if there's no labels that less than 0
    if np.any(y < 0):
        # If it's exist throw an error
        raise ValueError("Poisson deviance requires non-negative target values.")
    
    # Calculate labels mean
    mean_y = np.mean(y)

    # If labels mean is less than 0, return 0.0
    if mean_y <= 0:
        return 0.0

    # Calculate labels variance (poisson deviance)
    return safe_array(2.0 * np.sum(y * np.log(np.maximum(y, 1e-9) / mean_y) - (y - mean_y)))

def gini_impurity(y: np.ndarray) -> float:
    """
    Calculate the Gini impurity for a set of labels.

    Gini impurity measures the impurity of a node in a decision tree.
    It is defined as 1 - sum(p_i^2) where p_i is the proportion of samples
    of class i in the node.

    ## Args:
        **y**: *np.ndarray*
        Array of class labels.

    ## Returns:
        **float**: *The Gini impurity value.*

    ## Raises:
        **None**

    ## Notes:
      Calculation is helped by numpy for reaching C-like speed.

    ## Usage Example:
    ```python
    >>> labels = np.array([1, 2, 9, 1, 3, 2])
    >>>
    >>> impurity = gini_impurity(labels)
    >>> print("Impurity: ", impurity)
    >>> # print: 'Impurity:  0.7222222222222222'
    ```
    """
    if len(y) == 0:
        return 0.0

    y = y.astype(np.int32)
    max_label = y.max() if len(y) > 0 else 0
    counts = np.bincount(y, minlength=max_label + 1)
    probs = counts / len(y)
    gini = 1.0 - np.sum(probs ** 2)
    return gini

def log_loss_impurity(y: np.ndarray) -> float:
    """
    Calculate the log loss (cross-entropy) for a set of labels.

    Log loss measures the impurity of a node in a decision tree.
    It is defined as -sum(p_i * log(p_i)) where p_i is the proportion of samples
    of class i in the node.

    ## Args:
        **labels**: *np.ndarray* 
        Array of class labels.

    ## Returns:
        **float**: *The log loss value.*

    ## Raises:
        **None**

    ## Notes:
      Calculation is helped by numpy for reaching C-like speed.

    ## Usage Example:
    ```python
    >>> labels = np.array([1, 2, 9, 1, 3, 2])
    >>>
    >>> impurity = log_loss_impurity(labels)
    >>> print("Impurity: ", impurity)
    >>> # print: 'Impurity:  1.329661348854758'
    ```
    """
    if len(y) == 0:
        return 0.0

    y = y.astype(np.int32)
    max_label = y.max() if len(y) > 0 else 0
    counts = np.bincount(y, minlength=max_label + 1)
    probs = counts / len(y)

    log_loss_val = 0.0
    for p in probs:
        if p > 0:
            log_loss_val -= p * np.log(p)

    return safe_array(log_loss_val)

def entropy_impurity(y: np.ndarray) -> float:
    """
    Calculate the entropy for a set of labels.

    Entropy measures the impurity of a node in a decision tree.
    It is defined as -sum(p_i * log2(p_i)) where p_i is the proportion of samples
    of class i in the node.

    ## Args:
        **labels**: *np.ndarray*
        Array of class labels.

    ## Returns:
        **float**: *The entropy value.*

    ## Raises:
        **None**

    ## Notes:
      Calculation is helped by numpy for reaching C-like speed.

    ## Usage Example:
    ```python
    >>> labels = np.array([1, 2, 9, 1, 3, 2])
    >>>
    >>> impurity = entropy_impurity(labels)
    >>> print("Impurity: ", impurity)
    >>> # print: 'Impurity:  1.9182958340544893'
    ```
    """
    if len(y) == 0:
        return 0.0

    y = y.astype(np.int32)
    max_label = y.max() if len(y) > 0 else 0
    counts = np.bincount(y, minlength=max_label + 1)
    probs = counts / len(y)

    entropy_val = 0.0
    for p in probs:
        if p > 0:
            entropy_val -= p * np.log2(p)
    return entropy_val
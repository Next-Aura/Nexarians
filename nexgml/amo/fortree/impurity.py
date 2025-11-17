import numpy as np  # Numpy for numerical computations

def squared_error(labels: np.ndarray) -> float:
    """
    Calculate the variance of the given labels (MSE).

    ## Args:
        **labels**: *np.ndarray*
        Array of target values.

    ## Returns:
        **float**: *Variance of the labels. Returns 0.0 if labels are empty.*
        
    ## Raises:
        **None**
    """
    # Check label's size
    if labels.size == 0:
        # If the size is 0, then return 0.0
        return 0.0
    
    # Calculate label mean
    mean = np.mean(labels)
    # Calculate label variance (MSE)
    return np.mean((labels - mean) ** 2)

def friedman_squared_error(labels: np.ndarray) -> float:
    n = labels.size

    if n <= 1:
        return 0.0
    
    mean = np.mean((labels - np.mean(labels))**2)

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
    return 2.0 * np.sum(y * np.log(np.maximum(y, 1e-9) / mean_y) - (y - mean_y))

def gini_impurity(labels: np.ndarray) -> float:
    """
    Calculate the Gini impurity for a set of labels.

    Gini impurity measures the impurity of a node in a decision tree.
    It is defined as 1 - sum(p_i^2) where p_i is the proportion of samples
    of class i in the node.

    ## Args:
        **labels**: *np.ndarray*
        Array of class labels.

    ## Returns:
        **float**: *The Gini impurity value.*

    ## Raises:
        **None**
    """
    if len(labels) == 0:
        return 0.0

    labels = labels.astype(np.int32)
    max_label = labels.max() if len(labels) > 0 else 0
    counts = np.bincount(labels, minlength=max_label + 1)
    probs = counts / len(labels)
    gini = 1.0 - np.sum(probs ** 2)
    return gini

def log_loss_impurity(labels: np.ndarray) -> float:
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
    """
    if len(labels) == 0:
        return 0.0

    labels = labels.astype(np.int32)
    max_label = labels.max() if len(labels) > 0 else 0
    counts = np.bincount(labels, minlength=max_label + 1)
    probs = counts / len(labels)

    log_loss_val = 0.0
    for p in probs:
        if p > 0:
            log_loss_val -= p * np.log(p)

    return log_loss_val

def entropy_impurity(labels: np.ndarray) -> float:
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
    """
    if len(labels) == 0:
        return 0.0

    labels = labels.astype(np.int32)
    max_label = labels.max() if len(labels) > 0 else 0
    counts = np.bincount(labels, minlength=max_label + 1)
    probs = counts / len(labels)

    entropy_val = 0.0
    for p in probs:
        if p > 0:
            entropy_val -= p * np.log2(p)
    return entropy_val
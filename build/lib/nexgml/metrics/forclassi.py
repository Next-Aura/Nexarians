import numpy as np

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the mean accuracy on the given test data and labels.

    ## Args:
        **y_true**: *np.ndarray* or *spmatrix*
        True target labels.

        **y_pred**: *np.ndarray*
        Predicted target labels.

    ## Returns:
        **float**: *Mean accuracy score.*

    ## Raises:
        **None**

    ## Notes:
      This function only for classifier models.

    ## Usage Example:
    ```python
    >>> pred = model.predict(X_test)
    >>> acc = accuracy_score(y_true=y_test, y_pred=pred)
    >>>
    >>> print("Model's accuracy:", acc)
    ```
    """
    # Compare prediction with true labels and compute mean
    return np.mean(y_pred == y_true)

def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the precision score for binary classification.

    ## Args:
        **y_true**: *np.ndarray* or *spmatrix*
        True target labels.

        **y_pred**: *np.ndarray*
        Predicted target labels.

    ## Returns:
        **float**: *Precision score.*

    ## Raises:
        **None**

    ## Notes:
      This function only for classifier models.

    ## Usage Example:
    ```python
    >>> pred = model.predict(X_test)
    >>> precis = precision_score(y_true=y_test, y_pred=pred)
    >>>
    >>> print("Model's precision:", precis)
    ```
    """
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    predicted_positives = np.sum(y_pred == 1)
    
    if predicted_positives == 0:
        return 0.0
    
    return true_positives / predicted_positives

def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the recall score for binary classification.

    ## Args:
        **y_true**: *np.ndarray* or *spmatrix*
        True target labels.

        **y_pred**: *np.ndarray*
        Predicted target labels.

    ## Returns:
        **float**: *Recall score.*

    ## Raises:
        **None**

    ## Notes:
      This function only for classifier models.

    ## Usage Example:
    ```python
    >>> pred = model.predict(X_test)
    >>> recall = recall_score(y_true=y_test, y_pred=pred)
    >>>
    >>> print("Model's recall:", recall)
    ```
    """
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    actual_positives = np.sum(y_true == 1)
    
    if actual_positives == 0:
        return 0.0
    
    return true_positives / actual_positives

def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the F1 score for binary classification.

    ## Args:
        **y_true**: *np.ndarray* or *spmatrix*
        True target labels.

        **y_pred**: *np.ndarray*
        Predicted target labels.

    ## Returns:
        **float**: *F1 score.*

    ## Raises:
        **None**

    ## Notes:
      This function only for classifier models.

    ## Usage Example:
    ```python
    >>> pred = model.predict(X_test)
    >>> f1 = f1_score(y_true=y_test, y_pred=pred)
    >>>
    >>> print("Model's f1 score:", f1)
    ```
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    if (precision + recall) == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)
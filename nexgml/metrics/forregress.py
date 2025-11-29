import numpy as np

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
        **None**
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return 1 - (ss_res / ss_tot)
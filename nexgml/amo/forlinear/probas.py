import numpy as np                       # Numpy for numerical computations
from nexgml.guardians import safe_array  # For numerical stability

def softmax(z: np.ndarray, dtype: np.float32=np.float32) -> np.ndarray:
    """
    Calculate the softmax probability of the given logits.

    ## Args:
        **z**: *np.ndarray*
        Raw logits.

        **dtype**: *np.float32*
        Data type output.

    ## Returns:
        **np.ndarray**: *Probability of the given logits.*

    ## Raises:
        **None**

    ## Notes:
      Calculation is helped by numpy for reaching C-like speed.

    ## Usage Example:
    ```python
    >>> logits = np.mean([0.2, 0.2, 0.5, 0.1])
    >>>
    >>> proba = softmax(logits)
    >>> print("Proba: ", proba)
    >>> # print: 'Proba:  [0.23503441 0.23503441 0.31726326 0.21266793]'
    ```
    """
    z = np.asarray(z, dtype=dtype)
    if z.ndim == 1:
        z = z.reshape(1, -1)
        squeeze = True

    else:
        squeeze = False

    z_max = np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z - z_max)
    exp_z_sum = np.sum(exp_z, axis=1, keepdims=True, dtype=dtype)
    exp_z_sum = np.where(exp_z_sum == 0, 1, exp_z_sum)
    out = exp_z / exp_z_sum

    return safe_array(out[0], 1e-15, 1 - 1e-15, dtype=dtype) if squeeze else safe_array(out, 1e-15, 1 - 1e-15, dtype=dtype)

def sigmoid(z: np.ndarray, dtype: np.float32=np.float32) -> np.ndarray:
    """
    Calculate the sigmoid probability of the given logits.

    ## Args:
        **z**: *np.ndarray*
        Raw logits.

        **dtype**: *np.float32*
        Data type output.

    ## Returns:
        **np.ndarray**: *Probability of the given logits.*

    ## Raises:
        **None**

    ## Notes:
      Calculation is helped by numpy for reaching C-like speed.

    ## Usage Example:
    ```python
    >>> logits = np.array([0.725, 0.278])
    >>>
    >>> proba = sigmoid(logits)
    >>> print("Proba: ", proba)
    >>> # print: 'Proba:  [0.6737071  0.56905583]'
    ```
    """
    try:
        from scipy.special import expit
        return dtype(expit(z))
    
    except Exception:
        z_maxi = np.clip(z, -500, 500, dtype=dtype)
        return dtype(1) / (dtype(1) + np.exp(-z_maxi))
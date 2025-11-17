import numpy as np  # Numpy for numerical computations

def softmax(z: np.ndarray) -> np.ndarray:
    """
    Calculate the softmax probability of the given logits.

    ## Args:
        **z**: *np.ndarray*
        Raw logits.

    ## Returns:
        **np.ndarray**: *Probability of the given logits.*

    ## Raises:
        **None**
    """
    z = np.asarray(z)
    if z.ndim == 1:
        z = z.reshape(1, -1)
        squeeze = True

    else:
        squeeze = False

    z_max = np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z - z_max)
    exp_z_sum = np.sum(exp_z, axis=1, keepdims=True)
    exp_z_sum = np.where(exp_z_sum == 0, 1, exp_z_sum)
    out = exp_z / exp_z_sum

    return out[0] if squeeze else out

def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Calculate the sigmoid probability of the given logits.

    ## Args:
        **z**: *np.ndarray*
        Raw logits.

    ## Returns:
        **np.ndarray**: *Probability of the given logits.*

    ## Raises:
        **None**
    """
    try:
        from scipy.special import expit
        return expit(z)
    
    except Exception:
        z_maxi = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_maxi))

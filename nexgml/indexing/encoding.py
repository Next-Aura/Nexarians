import numpy as np                    # Numpy for numerical computations
from typing import Optional           # More specific type hints

def one_hot_labeling(y: np.ndarray, classes: Optional[np.ndarray]) -> np.ndarray:
    """
    Label one-hot encoding

    ## Args
        **y**: *np.ndarray*
        Labels data.

        **classes**: *Optional[np.ndarray]*
        Unique classes from labels data.

    ## Returns
        **np.ndarray**: *one-hot encoded label.*

    ## Raises
        **None**

    ## Notes:
      This function has python loop that may cause latency.

    ## Usage Example:
    ```python
    >>> y = [1, 1, 1, 3, 3, 1]
    >>> classes = np.unique(y)
    >>> one_hot = one_hot_labeling(y=y, classes=classes)
    >>>
    >>> print("One-hot label:", one_hot)
    ```
    """
    if classes is None:
        classes = np.unique(y)

    y_one_hot = np.zeros((y.shape[0], len(classes)), dtype=int)
    for i, cls in enumerate(classes):
        y_one_hot[:, i] = (y == cls).astype(int)
        
    return y_one_hot

def integer_labeling(y: np.ndarray, classes: Optional[np.ndarray], to_integer_from: str='one-hot') -> np.ndarray:
    """
    Label integer encoding

    ## Args:
        **y**: *np.ndarray*
        Labels data.

        **classes**: *Optional[np.ndarray]*
        Unique classes from labels data.

        **to_integer_from**: *str*
        Encode to integer from given argument dtype.

    ## Returns:
        **np.ndarray**: *Array of indices.*

    ## Raises:
        **ValueError**: *If 'to_integer_from' argument is invalid.*

    ## Notes:
      This function has python loop that may cause latency.

    ## Usage Example:
    ```python
    >>> y = [[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0]]
    >>> classes = np.unique(y)
    >>> label = integer_labeling(y=y, classes=classes)
    >>>
    >>> print("Integer label:", label)
    ```
    """
    if classes is None:
        classes = np.unique(y)

    if to_integer_from == 'one-hot':
        return np.argmax(y, axis=1)

    elif to_integer_from == 'labels':
        class_to_int = {cls: i for i, cls in enumerate(classes)}
        y_integer = np.array([class_to_int[cls] for cls in y])
        return y_integer

    else:
        raise ValueError(f"Invalid to_integer_from argument, {to_integer_from}.")
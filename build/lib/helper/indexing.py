import numpy as np                    # Numpy for numerical computations
from math import log2, sqrt           # For math operations
from functools import lru_cache       # For caching, prevent re-calculate same argument
from typing import Optional, Literal  # More specific type hints

# ========== JIT INITIALIZATION ==========
try:
    from numba import jit

except ImportError:
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# ========== INDEXING HELPER ==========
class Indexing:
    """
    Helper for indexing or label encoding task
    """
    @lru_cache
    @staticmethod
    def standard_indexing(n: int, maxi: Literal['sqrt', 'log2'] | float | int) -> int:
        """
        Get slicing data index.

        ## Args:
          **n**: *int*
          number of argument that want to be sliced.

          **maxi**: *Literal['sqrt', 'log2'], float, int*
          slicing method.
        
        ## Returns
          **int**: *Index of sliced data*

        ## Raises
          **ValueError**: *If invalid maxi argument is given*
        """
        if maxi is None:
            max_ = n

        elif isinstance(maxi, int):
                max_ = max(1, min(n, maxi))

        elif isinstance(maxi, str):
            if maxi == 'sqrt':
              max_ = max(1, int(sqrt(n)))

            elif maxi == 'log2':
                max_ = max(1, int(log2(n)))

            else:
                raise ValueError(f"Invalid maxi argument, {maxi}.")
                
        elif isinstance(maxi, float):
            max_ = max(1, min(n, int(np.round(n * maxi))))

        else:
            raise ValueError(f"Invalid maxi argument, {maxi}.")
        
        return max_
    
    @staticmethod
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
        """
        if classes is None:
            classes = np.unique(y)

        y_one_hot = np.zeros((y.shape[0], len(classes)), dtype=int)
        jit(nopython=True)
        for i, cls in enumerate(classes):
            y_one_hot[:, i] = (y == cls).astype(int)
            
        return y_one_hot
    
    @staticmethod
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
import numpy as np                    # Numpy for numerical computations
from math import log2, sqrt           # For math operations
from functools import lru_cache       # For caching, prevent re-calculate same argument
from typing import Literal            # More specific type hints

@lru_cache
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
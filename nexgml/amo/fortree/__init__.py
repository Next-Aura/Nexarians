"""*ForTree* module provides common machine learning calculation in tree models.
  ## Provides:
    - **impurity**: *Focused on label impurity computations.*
  
  ## See also:
    - **forlinear**
  
  ## Note:
    **All the helpers implemented in python programming language.**"""

from .impurities import (squared_error,
                       friedman_squared_error,
                       absolute_error,
                       gini_impurity,
                       log_loss_impurity,
                       poisson_deviance,
                       entropy_impurity)

__all__ = [
    'squared_error',
    'friedman_squared_error',
    'absolute_error',
    'gini_impurity',
    'log_loss_impurity',
    'poisson_deviance',
    'entropy_impurity'
]
"""*ForLinear* module provides common calculation in machine learning.
  ## Provides:
    - **losses**: *Focused on losses computations.*
    - **probas**: *Focused on probability computations.*
    - **fortree**: *sub-amo that focus on tree models computations.*
    - **penalties**: *Focused on penalties computations.*
    - **derivs**: *Focused on linear function derivative computations.*
  
  ## See also:
    - **fortree**
  
  ## Notes:
    **All the helpers implemented in python programming language.**"""

from .losses import (mean_squared_error, 
                     mean_absolute_error, 
                     root_squared_error, 
                     smoothl1_loss, 
                     categorical_ce, 
                     binary_ce)

from .probas import (softmax,
                     sigmoid)

from .penalties import (lasso,
                        ridge,
                        elasticnet)

from .derivs import (mse_deriv,
                     mae_deriv,
                     rmse_deriv,
                     smoothl1_deriv,
                     lasso_deriv,
                     ridge_deriv,
                     elasticnet_deriv,
                     cce_deriv)

__all__ = [
    'mean_squared_error',
    'mean_absolute_error', 
    'root_squared_error', 
    'smoothl1_loss', 
    'categorical_ce', 
    'binary_ce',
    'softmax',
    'sigmoid',
    'lasso',
    'ridge',
    'elasticnet',
    'mse_deriv',
    'mae_deriv',
    'rmse_deriv',
    'smoothl1_deriv',
    'lasso_deriv',
    'ridge_deriv',
    'elasticnet_deriv',
    'cce_deriv',
]
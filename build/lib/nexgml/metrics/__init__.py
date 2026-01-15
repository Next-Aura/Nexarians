"""*Metrics* module provided helper for model metric utilities.

  ## Provides:
    - **forregress**: *Focused on regression task model metrics.*
    - **forclassi**: *Focused on classification task model metrics.*
  
  ## See also:
    - **amo (Advanced Math Operations)**
  
  ## Note:
    **All the helpers implemented in python programming language.**"""

from .forregress import (r2_score,
                         root_mean_squared_error)
from .forclassi import (accuracy_score,
                       precision_score,
                       recall_score,
                       f1_score)

__all__ = [
    'r2_score',
    'accuracy_score',
    'precision_score',
    'recall_score',
    'f1_score',
    'root_mean_squared_error'
]
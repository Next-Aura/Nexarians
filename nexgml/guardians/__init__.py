"""*Guardians* module provided helper for numerical stability.

  ## Provides:
    - **arr**: *Focused on array numerical stabilities.*
  
  ## See also:
    - **amo (Advanced Math Operations)**
  
  ## Note:
    **All the helpers implemented in python programming language.**"""

from .arr import (safe_array, hasinf, hasnan)

__all__ = [
    'safe_array',
    'hasinf',
    'hasnan'
]
"""*Guardians* module provided helper for numerical stability.

  ## Provides:
    - **arr**: *Focused on array numerical stabilities.*
    - **detect**: *Focused on specific data detection.* 
  
  ## See also:
    - **amo (Advanced Math Operations)**
  
  ## Note:
    **All the helpers implemented in python programming language.**"""

from .arr import (safe_array, issafe_array)
from .detect import (hasinf, hasnan, iscontinuous, isdiscrete)

__all__ = [
    'safe_array',
    'hasinf',
    'hasnan',
    'iscontinuous',
    'isdiscrete',
    'issafe_array'
]
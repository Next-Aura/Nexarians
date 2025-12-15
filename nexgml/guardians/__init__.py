"""*Guardians* module provided helper for numerical stability.

  ## Provides:
    - **arr**: *Focused on array numerical stabilities.*
    - **detect**: *Focused on specific data detection.* 
  
  ## See also:
    - **amo (Advanced Math Operations)**
  
  ## Note:
    **All the helpers implemented in python programming language.**"""

from .arr import (safe_array, issafe_array)
from .detect import (hasinf, hasnan, iscontinious, isdiscrete)

__all__ = [
    'safe_array',
    'hasinf',
    'hasnan',
    'iscontinious',
    'isdiscrete',
    'issafe_array'
]
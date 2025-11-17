"""*Indexing* module provided helper for indexing utilities.
  ## Provides:
  - **encoding**: *Focused on encoding utilities like tranforming labels format.*
  - **indexing**: *Focused on indexing utilities like data slicing.*
  
  ## See also:
  - **amo (Advanced Math Operations)**
  
  ## Note:
  **All the helpers implemented in python programming language.**"""

  
from .indexing import standard_indexing
from .encoding import (one_hot_labeling,
                       integer_labeling)

__all__ = [
    'standard_indexing',
    'one_hot_labeling',
    'integer_labeling'
]
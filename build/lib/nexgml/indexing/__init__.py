"""*Indexing* module provided helper for indexing utilities.
  
  ## Provides:
    - **encoding**: *Focused on encoding utilities like tranforming labels format.*
    - **indexing**: *Focused on indexing utilities like data slicing.*
    - **detecting**: *Focused on detecting data format like one-hot encoding detection.*
  
  ## See also:
    - **amo (Advanced Math Operations)**
  
  ## Note:
    **All the helpers implemented in python programming language.**"""

  
from .indexing import standard_indexing
from .encoding import (one_hot_labeling,
                       integer_labeling)
from .detecting import isone_hot, isbinary

__all__ = [
    'standard_indexing',
    'one_hot_labeling',
    'integer_labeling',
    'isone_hot',
    'isbinary',
]
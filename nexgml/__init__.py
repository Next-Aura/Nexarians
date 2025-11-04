"""NexGML — Next Generation Machine Learning

A collection of educational ML implementations focusing on gradient-based methods,
tree models, and utility helpers.
"""
__version__ = "0.1.3"

from .helper.amo import AMO, ForTree
from .helper.indexing import Indexing
import gradient_supported
import tree_models

__all__ = [
    "AMO", 
    "ForTree", 
    "__version__", 
    "Indexing",
    "gradient_supported",
    "tree_models"]
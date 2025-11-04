"""NexGML — Next Generation Machine Learning

A collection of educational ML implementations focusing on gradient-based methods,
tree models, and utility helpers.
"""
__version__ = "0.1.2"

from .helper.amo import AMO, ForTree
from .helper.indexing import Indexing

__all__ = ["AMO", "ForTree", "__version__", "Indexing"]
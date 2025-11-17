"""# NexGML — Next Generation Machine Learning
A collection of educational ML implementations.
## Description:
  **NexGML (Next Generation Machine Learning) made from a teenager curiousness about AI,
  NexGML is implementation of that teenager knowledge about AI. NexGML also has a purpose
  for helping people to learn about machine learning and use them in some cases.**
  
## Provides:
  - **gradient_supported**
  - **tree_models**
  - **indexing**
  - **amo (Advanced Math Operations)**

## See also:
  - **scikit-learn**
  - **xgboost**

## Supports:
  **You can also be part of NexGML with join our community discord, Nexarians Base Camp,
  the discord server will provide AI related news, off-topic channel, and more!.**

### Version:
  **1.0.0**
"""

__version__ = "1.0.0"

import sys
from nexgml.modelkit.nexg import ModelLike
from nexgml.model_configs.Config import AdvancedConfig

__all__ = [
    'ModelLike', 
    "AdvancedConfig", 
    "__version__",
    ]
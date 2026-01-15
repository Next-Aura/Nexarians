# NexGML Helper: `Indexing`

## Overview

The `Indexing` module provides specialized, optimized helper functions crucial for data preparation in machine learning, focusing on **label encoding** and **data indexing/slicing strategies**.

This module is essential for ensuring training data is in the correct format (e.g., one-hot encoding for multi-class classification) and for implementing advanced sampling or feature selection techniques that rely on sophisticated index generation.

## Installation & Import

This module is part of the internal `NexGML` helper library. You can import the functions as follows:

```python
import numpy as np
from nexgml.indexing import standard_indexing, one_hot_labeling...
```

## `Indexing` API Reference

This class provides static methods for robust and optimized data labeling and indexing operations.

-----

### `one_hot_labeling(y, classes, dtype=np.int32)`

Converts an array of categorical or numerical labels into a **one-hot encoded** matrix, required for models using Softmax and Categorical Cross-Entropy (CCE).

  * **Parameters**:
      * `y` (`np.ndarray`): 1D array of target labels (e.g., `[0, 1, 0, 2]`).
      * `classes` (`Optional[np.ndarray]`): Array of unique class labels. If `None`, it is determined from `y`.
      * `dtype` (`DTypeLike`, default=`np.int32`): Data type output.
  * **Returns**:
      * (`np.ndarray`): One-hot encoded matrix (samples × number of classes).

### `integer_labeling(y, classes, to_integer_from='one-hot', dtype=np.int32)`

Converts encoded labels back into a 1D array of integer indices or original labels.

  * **Parameters**:
      * `y` (`np.ndarray`): Labels data. Can be a one-hot matrix or an array of original labels.
      * `classes` (`Optional[np.ndarray]`): Unique classes used for mapping. If `None`, it is determined from `y`.
      * `to_integer_from` (`str`, default=`'one-hot'`):
          * `'one-hot'`: Converts a one-hot matrix (`y`) into integer indices (0, 1, 2, ...).
          * `'labels'`: Converts a label array (`y`) into contiguous integer indices (0, 1, 2, ...).
      * `dtype` (`DTypeLike`, default=`np.int32`): Data type output.
  * **Returns**:
      * (`np.ndarray`): Array of integer indices.
  * **Raises**:
      * `ValueError`: If `to_integer_from` argument is invalid.

### `standard_indexing(n, maxi)`

Determines the optimal maximum index or slice size for sub-sampling, typically used in feature selection or model splitting criteria.

  * **Key Feature**: This method is decorated with **`@lru_cache`** to memoize results for identical input parameters, significantly improving performance when called repeatedly in tree-based algorithms or hyperparameter tuning.

  * **Parameters**:

      * `n` (`int`): The total number of arguments/samples to be sliced.
      * `maxi` (`Literal['sqrt', 'log2'] | float | int`): The slicing method:
          * `int`: Uses the provided integer value (capped at `n`).
          * `float`: Uses a fraction of `n` (e.g., `0.5` means $0.5 \times n$).
          * `'sqrt'`: Uses $\lfloor \sqrt{n} \rfloor$.
          * `'log2'`: Uses $\lfloor \log_2(n) \rfloor$.

  * **Returns**:

      * (`int`): The calculated index or slice size.

  * **Raises**:

      * `ValueError`: If `maxi` argument is invalid.

...
## Usage Examples

```python
import numpy as np
from nexgml.indexing import one_hot_labeling, integer_labeling, standard_indexing

# --- Label Encoding Example ---
print("--- Label Encoding ---")
raw_labels = np.array(['cat', 'dog', 'cat', 'bird', 'dog'])
unique_classes = np.unique(raw_labels) # ['bird', 'cat', 'dog']

# 1. One-Hot Encoding
y_ohe = one_hot_labeling(raw_labels, unique_classes)
print(f"One-Hot Encoded:\n{y_ohe}")
# Output: [[0 1 0], [0 0 1], [0 1 0], [1 0 0], [0 0 1]]

# 2. Integer Encoding (from One-Hot)
y_int_from_ohe = integer_labeling(y_ohe)
print(f"Integer Indices (from OHE): {y_int_from_ohe}") 
# Output: [1 2 1 0 2] (index 0='bird', 1='cat', 2='dog')

# 3. Integer Encoding (from Labels)
y_int_from_labels = integer_labeling(raw_labels, unique_classes, to_integer_from='labels')
print(f"Integer Indices (from Labels): {y_int_from_labels}") 
# Output: [1 2 1 0 2]

# --- Indexing / Slicing Example ---
print("\n--- Slicing Strategy ---")
total_features = 1000

# Calculate max features based on sqrt rule (common in Random Forests)
max_sqrt = standard_indexing(total_features, 'sqrt')
print(f"Max features (sqrt of 1000): {max_sqrt}") 
# Output: 31

# Calculate max features based on 5% rule
max_float = standard_indexing(total_features, 0.05)
print(f"Max features (5% of 1000): {max_float}") 
# Output: 50
```

## Performance Note

The `standard_indexing` method utilizes the **`@lru_cache`** decorator from Python's standard library. This ensures that the slicing logic (especially the costly `sqrt` and `log2` calculations) is only executed once for any given pair of `(n, maxi)` arguments. Subsequent calls with the same arguments will return the cached result instantly, which is a major performance benefit when building complex ensembles or performing repeated cross-validation.
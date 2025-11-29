# NexGML Helper: `Guardians`

## Overview

The `Guardians` module provides specialized utilities for numerical stability in machine learning, focusing on **safe array processing** to handle potential issues like NaN values, infinities, and extreme outliers that could destabilize computations.

This module is essential for preprocessing data arrays before feeding them into models, ensuring robustness against numerical instabilities that might arise from floating-point operations or data anomalies.

## Installation & Import

This module is part of the internal `NexGML` helper library. You can import the functions as follows:

```python
import numpy as np
from nexgml.guardians import safe_array
```

## `Guardians` API Reference

This module provides static methods for robust and safe numerical array operations.

-----

### `safe_array(arr, max_value=1e10, min_value=-1e10)`

Safely convert array to finite numbers within specified bounds.

  * **Parameters**:
      * `arr` (`np.ndarray`): Input array to be processed.
      * `max_value` (`float`, default=`1e10`): Maximum allowable value in the array.
      * `min_value` (`float`, default=`-1e10`): Minimum allowable value in the array.
  * **Returns**:
      * (`np.ndarray`): Processed array with values clipped to the specified bounds.
  * **Raises**:
      * `RuntimeWarning`: Warns if any values were clipped due to overflow.

## Usage Examples

```python
import numpy as np
from nexgml.guardians import safe_array

# Example with problematic array
problematic_array = np.array([1.0, np.nan, np.inf, -np.inf, 1e15, -1e15])
print("Original array:", problematic_array)
# Output: [  1.   nan   inf -inf 1e+15 -1e+15]

# Process with safe_array
safe_arr = safe_array(problematic_array)
print("Safe array:", safe_arr)
# Output: [ 1.00000000e+00  0.00000000e+00  1.00000000e+10 -1.00000000e+10  1.00000000e+10 -1.00000000e+10]
# Note: RuntimeWarning may be issued if clipping occurs

# Custom bounds
custom_safe_arr = safe_array(problematic_array, max_value=1000.0, min_value=-1000.0)
print("Custom bounds safe array:", custom_safe_arr)
# Output: [ 1.00000000e+00  0.00000000e+00  1.00000000e+03 -1.00000000e+03  1.00000000e+03 -1.00000000e+03]
```

## Performance Note

all the functions along with NumPy computation. This results in C-like speed for all calculations, which is critical when these functions are called millions of times inside a model's training loop.
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

or

```python
import numpy as np
from nexgml import guardians as grd
```

## `Guardians` API Reference

This module provides static methods for robust and safe numerical array operations.

-----

### `safe_array(arr, min_value=-1e10, max_value=1e10, warn_me=True, clip=True, dtype=np.float32)`

Safely convert array to finite numbers within specified bounds.

  * **Parameters**:
      * `arr` (`np.ndarray`): Input array to be processed.
      * `min_value` (`float`, default=`-1e10`): Minimum allowable value in the array.
      * `max_value` (`float`, default=`1e10`): Maximum allowable value in the array.
      * `warn_me` (`bool`, default=`True`): If True, the function would throw a warning if there's a NaN or infinity value detected.
      * `clip` (`bool`, default=`True`): If True, the function would clip array value if it greater than max_value or less than min_value.
      * `dtype` (`DTypeLike`, default=`np.float32`): Data type output.
  * **Returns**:
      * (`np.ndarray`): Processed array with values clipped to the specified bounds.
  * **Raises**:
      * `RuntimeWarning`: Warns if any values were clipped due to overflow.

### `issafe_array(arr)`

Check if an array is safe for numerical operations.

  * **Parameters**:
      * `arr` (`np.ndarray`): Input array to be processed.
  * **Returns**:
      * (`bool`): Array's safety condition.

### `hasinf(arr)`

Check if there's an infinity value in an object.

  * **Parameters**:
      * `arr` (`np.ndarray, list, spmatrix`): Input to be processed.
  * **Returns**:
      * (`bool`): Condition if there's an infinity value.

### `hasnan(arr)`

Check if there's an infinity value in an object.

  * **Parameters**:
      * `arr` (`np.ndarray, list, spmatrix`): Input to be processed.
  * **Returns**:
      * (`bool`): Condition if there's a NaN.

### `iscontinious(a)`

Check if an array contain continious value.

  * **Parameters**:
      * `a` (`np.ndarray, list, spmatrix`): Input to be processed.
  * **Returns**:
      * (`bool`): Contidion if there's a continious value.

### `isdiscrete(a)`

Check if an array contain discrete value.

  * **Parameters**:
      * `a` (`np.ndarray, list, spmatrix`): Input to be processed.
  * **Returns**:
      * (`bool`): Contidion if there's a discrete value.

## Usage Examples

```python
import numpy as np
from nexgml.guardians import safe_array, hasinf, hasnan, isdiscrete, iscontinious

# Example with problematic array
problematic_array = np.array([1.0, np.nan, np.inf, -np.inf, 1e15, -1e15])
print("Original array:", problematic_array)
# Output: [  1.   nan   inf -inf 1e+15 -1e+15]

# Process with safe_array
safe_arr = safe_array(problematic_array)
has_inf = hasinf(problematic_array)
has_nan = hasnan(problematic_array)
is_cont = iscontinious(problematic_array)
is_disc = isdiscrete(problematic_array)
print("Safe array:", safe_arr)
print("Has infinity:", has_inf)
print("Has nan:", has_nan)
print("Is Continious:", is_cont)
print("Is Discrete:", is_disc)
# Output: [ 1.00000000e+00  0.00000000e+00  1.00000000e+10 -1.00000000e+10  1.00000000e+10 -1.00000000e+10]
# Note: RuntimeWarning may be issued if clipping occurs

# Custom bounds
custom_safe_arr = safe_array(problematic_array, max_value=1000.0, min_value=-1000.0)
print("Custom bounds safe array:", custom_safe_arr)
# Output: [ 1.00000000e+00  0.00000000e+00  1.00000000e+03 -1.00000000e+03  1.00000000e+03 -1.00000000e+03]
```

## Performance Note

All the functions along with NumPy computation. This results in C-like speed for all calculations, which is critical when these functions are called millions of times inside a model's training loop.
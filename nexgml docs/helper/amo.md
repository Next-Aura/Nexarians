# NexGML Helper: `AMO`

## Overview

The `AMO` module provides a collection of optimized, low-level mathematical operations essential for building machine learning algorithms. These helpers are designed for high performance and numerical stability.

The module is split into two primary static classes:

  * **`AMO` (Advanced Math Operations)**: Contains common activation functions (like `softmax`, `sigmoid`) and loss functions (like `categorical_ce`, `mean_squared_error`).
  * **`ForTree`**: Contains impurity measures and criteria (like `gini_impurity`, `squared_error`) specifically required for building decision tree models.

For maximum speed, both classes and their methods are JIT-compiled using `numba`.

## Installation & Import

This module is part of the internal `NexGML` helper library. You can import the classes as follows:

```python
import numpy as np
from nexgml import AMO
from nexgml import ForTree
```

## `AMO` API Reference

This class provides static methods for common activation and loss calculations.

-----

### `AMO.softmax(z)`

Calculates the numerically stable softmax probability for a given set of logits.

  * **Parameters**:
      * `z` (`np.ndarray`): Raw logits, can be 1D or 2D.
  * **Returns**:
      * (`np.ndarray`): Probability scores (summing to 1 across the class axis).

### `AMO.sigmoid(z)`

Calculates the sigmoid (logistic) probability. It attempts to use `scipy.special.expit` for high precision and falls back to a clipped `numpy` implementation if `scipy` is unavailable.

  * **Parameters**:
      * `z` (`np.ndarray`): Raw logits.
  * **Returns**:
      * (`np.ndarray`): Probabilities between 0 and 1.

### `AMO.categorical_ce(y_true, y_pred_proba, mean=True)`

Calculates the categorical cross-entropy loss.

**Key Feature**: This function includes built-in **automatic class weighting** to counteract imbalanced datasets. Weights are calculated inversely proportional to class frequency.

  * **Parameters**:
      * `y_true` (`np.ndarray`): True labels, expected in **one-hot encoded** format.
      * `y_pred_proba` (`np.ndarray`): Predicted class probabilities from the model.
      * `mean` (`bool`, default=`True`): If `True`, returns the mean loss over the batch. If `False`, returns the loss for each sample.
  * **Returns**:
      * (`np.ndarray` or `float`): The calculated loss.

### `AMO.binary_ce(y_true, y_pred_proba, mean=True)`

Calculates the binary cross-entropy loss for binary classification.

  * **Parameters**:
      * `y_true` (`np.ndarray`): True binary labels (0 or 1).
      * `y_pred_proba` (`np.ndarray`): Predicted probabilities (between 0 and 1).
      * `mean` (`bool`, default=`True`): If `True`, returns the mean loss.
  * **Returns**:
      * (`np.ndarray` or `float`): The calculated loss.

### `AMO.mean_squared_error(y_true, y_pred)`

Calculates the Mean Squared Error (MSE) regression loss.

  * **Parameters**:
      * `y_true` (`np.ndarray`): True continuous target values.
      * `y_pred` (`np.ndarray`): Predicted continuous values.
  * **Returns**:
      * (`float`): The MSE loss.

### `AMO.mean_absolute_error(y_true, y_pred)`

Calculates the Mean Absolute Error (MAE) regression loss.

  * **Parameters**:
      * `y_true` (`np.ndarray`): True continuous target values.
      * `y_pred` (`np.ndarray`): Predicted continuous values.
  * **Returns**:
      * (`float`): The MAE loss.

### `AMO.root_squared_error(y_true, y_pred)`

Calculates the Root Mean Squared Error (RMSE) regression loss.

  * **Parameters**:
      * `y_true` (`np.ndarray`): True continuous target values.
      * `y_pred` (`np.ndarray`): Predicted continuous values.
  * **Returns**:
      * (`float`): The RMSE loss.

### `AMO.smoothl1_loss(y_true, y_pred, delta)`

Calculates the Smooth L1 (Huber) Loss, which is less sensitive to outliers than MSE.

  * **Parameters**:
      * `y_true` (`np.ndarray`): True continuous target values.
      * `y_pred` (`np.ndarray`): Predicted continuous values.
      * `delta` (`float`): The threshold point where the function transitions from quadratic to linear.
  * **Returns**:
      * (`float`): The Smooth L1 loss.

## `ForTree` API Reference

This class provides static methods for calculating node impurity and criteria, essential for splitting nodes in decision tree models.

-----

### Regression Tree Criteria

### `ForTree.squared_error(labels)`

Calculates the variance of the labels, used as the splitting criterion for standard regression trees (equivalent to MSE).

  * **Parameters**:
      * `labels` (`np.ndarray`): Continuous target values in the current node.
  * **Returns**:
      * (`float`): The variance (impurity) of the node.

### `ForTree.friedman_squared_error(labels)`

Calculates an improved version of MSE for tree splitting, often used in Gradient Boosting.

  * **Parameters**:
      * `labels` (`np.ndarray`): Continuous target values in the current node.
  * **Returns**:
      * (`float`): The impurity of the node.

### `ForTree.absolute_error(labels)`

Calculates the mean absolute error from the mean of the labels, used as a robust splitting criterion for regression trees (less sensitive to outliers).

  * **Parameters**:
      * `labels` (`np.ndarray`): Continuous target values in the current node.
  * **Returns**:
      * (`float`): The impurity of the node.

### `ForTree.poisson_deviance(labels)`

Calculates the Poisson deviance, used as a splitting criterion for regression tasks involving count data.

  * **Parameters**:
      * `labels` (`np.ndarray`): Non-negative count target values.
  * **Returns**:
      * (`float`): The impurity of the node.
  * **Raises**:
      * `ValueError`: If any label is negative.

### Classification Tree Criteria

### `ForTree.gini_impurity(labels)`

Calculates the Gini impurity, a standard criterion for classification trees (like CART). Measures the probability of misclassifying a randomly chosen element.

  * **Parameters**:
      * `labels` (`np.ndarray`): Integer class labels in the current node.
  * **Returns**:
      * (`float`): The Gini impurity (0 = pure, 0.5 = max impurity for 2 classes).

### `ForTree.log_loss_impurity(labels)`

Calculates the Cross-Entropy (Log Loss) impurity for a set of labels.

  * **Parameters**:
      * `labels` (`np.ndarray`): Integer class labels in the current node.
  * **Returns**:
      * (`float`): The log loss impurity.

### `ForTree.entropy_impurity(labels)`

Calculates the Entropy impurity (using `log2`), used as the criterion for ID3 and C4.5 classification trees.

  * **Parameters**:
      * `labels` (`np.ndarray`): Integer class labels in the current node.
  * **Returns**:
      * (`float`): The entropy (0 = pure, 1 = max impurity for 2 classes).

## Usage Examples

```python
import numpy as np
from nexgml import AMO, ForTree

# --- AMO Example ---
print("--- AMO ---")
logits = np.array([1.0, 3.0, 0.5])
probs = AMO.softmax(logits)
print(f"Softmax: {probs}")
# Output: Softmax: [[0.1063317 0.784663  0.1090053]]

# Imbalanced labels (class 0 is rare)
y_true_ohe = np.array([[0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]])
y_pred_probs = np.array([
    [0.1, 0.8, 0.1], 
    [0.9, 0.05, 0.05], 
    [0.2, 0.7, 0.1], 
    [0.1, 0.6, 0.3]
])

# AMO.categorical_ce will automatically weigh class 0 higher
loss = AMO.categorical_ce(y_true_ohe, y_pred_probs)
print(f"Weighted CCE Loss: {loss:.6f}")


# --- ForTree Example ---
print("\n--- ForTree ---")
# Node labels for classification
class_labels = np.array([0, 1, 1, 0, 1, 2, 1])
gini = ForTree.gini_impurity(class_labels)
entropy = ForTree.entropy_impurity(class_labels)
print(f"Gini Impurity: {gini:.6f}")
print(f"Entropy: {entropy:.6f}")

# Node labels for regression
reg_labels = np.array([10.5, 11.2, 9.8, 10.1, 12.0])
mse_crit = ForTree.squared_error(reg_labels)
print(f"Regression Criterion (MSE): {mse_crit:.6f}")
```

## Performance Note

The `AMO` and `ForTree` classes, along with their static methods, are decorated with `numba.jit(nopython=True)`. This compiles the Python code to optimized machine code on the first execution. This results in C-like speed for all calculations, which is critical when these functions are called millions of times inside a model's training loop.
# Metrics - Evaluation Metrics for Classification and Regression

## Overview

The `Metrics` module provides a collection of evaluation metrics for assessing the performance of machine learning models. These metrics are essential for measuring how well classification and regression models perform on given datasets.

The module is split into two different parts:

  * **`ForClassi`**: Sub-Metrics that contains common classification metrics (like `accuracy_score`, `precision_score`, `recall_score`, `f1_score`).
  * **`ForRegress`**: Sub-Metrics that contains common regression metrics (like `r2_score`).

## Installation & Import

This module is part of the internal `NexGML` helper library. You can import the functions as follows:

```python
from nexgml.metrics import accuracy_score, precision_score, recall_score, f1_score
from nexgml.metrics import r2_score
```

`Or`

```python
from nexgml.metrics import forclassi as fc
from nexgml.metrics import forregress as fr
```

## `ForClassi` API Reference

ForClassi module provides NumPy backend operations for common classification evaluation metrics.

-----

### `accuracy_score(y_true, y_pred)`

Calculates the mean accuracy on the given test data and labels.

  * **Parameters**:
      * `y_true` (`np.ndarray` or `spmatrix`): True target labels.
      * `y_pred` (`np.ndarray`): Predicted target labels.
  * **Returns**:
      * (`float`): Mean accuracy score.
  * **Raises**:
      * `None`

### `precision_score(y_true, y_pred)`

Calculates the precision score for binary classification.

  * **Parameters**:
      * `y_true` (`np.ndarray` or `spmatrix`): True target labels.
      * `y_pred` (`np.ndarray`): Predicted target labels.
  * **Returns**:
      * (`float`): Precision score.
  * **Raises**:
      * `None`

### `recall_score(y_true, y_pred)`

Calculates the recall score for binary classification.

  * **Parameters**:
      * `y_true` (`np.ndarray` or `spmatrix`): True target labels.
      * `y_pred` (`np.ndarray`): Predicted target labels.
  * **Returns**:
      * (`float`): Recall score.
  * **Raises**:
      * `None`

### `f1_score(y_true, y_pred)`

Calculates the F1 score for binary classification.

  * **Parameters**:
      * `y_true` (`np.ndarray` or `spmatrix`): True target labels.
      * `y_pred` (`np.ndarray`): Predicted target labels.
  * **Returns**:
      * (`float`): F1 score.
  * **Raises**:
      * `None`

## `ForRegress` API Reference

ForRegress module provides NumPy backend operations for common regression evaluation metrics.

-----

### `r2_score(y_true, y_pred)`

Calculates the R² (coefficient of determination) regression score function.

  * **Parameters**:
      * `y_true` (`np.ndarray`): True target values.
      * `y_pred` (`np.ndarray`): Predicted target values.
  * **Returns**:
      * (`float`): R² score.
  * **Raises**:
      * `None`

## Usage Examples

```python
import numpy as np
from nexgml.metrics.forclassi import accuracy_score, precision_score, recall_score, f1_score
from nexgml.metrics.forregress import r2_score

# --- Classification Example ---
y_true_class = np.array([0, 1, 1, 0, 1])
y_pred_class = np.array([0, 1, 0, 0, 1])

acc = accuracy_score(y_true_class, y_pred_class)
prec = precision_score(y_true_class, y_pred_class)
rec = recall_score(y_true_class, y_pred_class)
f1 = f1_score(y_true_class, y_pred_class)

print(f"Accuracy: {acc:.6f}")
print(f"Precision: {prec:.6f}")
print(f"Recall: {rec:.6f}")
print(f"F1 Score: {f1:.6f}")

# --- Regression Example ---
y_true_reg = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred_reg = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

r2 = r2_score(y_true_reg, y_pred_reg)
print(f"R² Score: {r2:.6f}")
```

## Performance Note

The `ForClassi` and `ForRegress` modules utilize NumPy computations for efficient metric calculations, ensuring fast evaluation even for large datasets.

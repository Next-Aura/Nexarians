# L1R – L1 Regressor

## Overview

L1R (L1 Regressor), also known as Lasso Regression, is a linear regression model that uses L1 regularization to prevent overfitting and perform feature selection (driving some coefficients to zero). It optimizes using Coordinate Descent, an iterative method that updates one coefficient at a time. It supports dense, sparse, and pandas inputs.

Perfect for teaching, quick prototyping, or when you need sparse models without heavy dependencies.

## Installation & Requirements

```bash
pip install numpy scipy pandas
```

```python
from nexgml.gradient_supported import L1Regressor
```

## Mathematical Formulation

### Prediction Function
$$
\hat{y} = Xw + b
$$

- $X$ – feature matrix  
- $w$ – weight vector  
- $b$ – bias (if `fit_intercept=True`)

### Loss Functions

- **MSE (Mean Squared Error) with L1 Regularization**: $\frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2 + \alpha \sum |w_i|$

### Regularization

- **L1 (Lasso)**: $\alpha \sum |w_i|$ (promotes sparsity)

### Optimization (Coordinate Descent)

Iteratively update each $w_j$:

$$
w_j = \text{soft_threshold}\left( \frac{1}{N} X_j^T (y - X_{-j} w_{-j}), \alpha \right) / ||X_j||^2
$$

Using soft-threshold: $\text{sign}(z) \max(|z| - \gamma, 0)$

Converges when coefficient changes < `tol`.

## Key Features
- **Regularization**: L1 (Lasso) for feature selection  
- **Input**: dense `np.ndarray`, `pd.DataFrame`, **or** SciPy sparse matrices  
- **Optimization**: Coordinate Descent  
- **Early stopping** on convergence (`tol`)  
- **Verbose levels** (0/1/2)  
- **Intercept**: Optional via `fit_intercept`  

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_iter` | `int` | `100` | Max iterations for Coordinate Descent |
| `alpha` | `float` | `1e-4` | Regularization strength |
| `fit_intercept` | `bool` | `True` | Add bias term |
| `tol` | `float` | `1e-4` | Convergence tolerance |
| `early_stopping` | `bool` | `True` | Enable early stop on convergence |
| `verbose` | `int` | `0` | 0 = silent, 1 = ~5 % progress, 2 = every iteration |

## Model Attributes (post-fit)

| Attribute | Type | Description |
|-----------|------|-------------|
| `weights` | `np.ndarray` | Learned coefficients |
| `b` | `float \| np.ndarray` | Bias (intercept); scalar or array for multi-output |
| `n_outputs_` | `int` | Number of output dimensions |
| `loss_history` | `List[float]` | Residual per iteration |

## API Reference

### `L1Regressor.__init__(...)`
Creates the model with the hyper-parameters above.

### `fit(X_train, y_train)`
Fits using Coordinate Descent.

- **Raises** `ValueError` for NaN/Inf or shape mismatch

### `predict(X_test)`
Returns $\hat{y}$ for new samples.

- **Raises** `ValueError` if model not fitted or NaN/Inf

### `score(X_test, y_test)`
Returns R² score.

### `get_params(deep)`
Returns model paramters

### `set_params(**params)`
Returns model's attribute that ready to set

## Usage Examples

### 1. Default (with intercept)
```python
import numpy as np
from sklearn.datasets import make_regression
from nexgml.gradient_supported import L1Regressor

X, y = make_regression(n_samples=200, n_features=10, noise=0.2, random_state=42)

model = L1Regressor(alpha=0.05, max_iter=200, verbose=1)
model.fit(X, y)

print(f"Loss: {model.loss_history[-1]:.6f}")
print(f"Weights (mean): {model.weights.mean():.6f}, bias: {model.b:.6f}")
```

### 2. No intercept + pandas
```python
import pandas as pd

X_df = pd.DataFrame(X)
y_series = pd.Series(y)

model = L1Regressor(fit_intercept=False, alpha=0.01, tol=1e-5)
model.fit(X_df, y_series)
```

### 3. Sparse data
```python
from scipy.sparse import csr_matrix

X_sp = csr_matrix(np.random.randn(500, 200))
y_sp = np.random.randn(500)

model = L1Regressor(alpha=0.001, early_stopping=False, verbose=2)
model.fit(X_sp, y_sp)
```

## Best Practices

1. **Scale features** (`StandardScaler`) → improves convergence and sparsity.  
2. Tune `alpha` via cross-validation: higher values increase sparsity.  
3. Use for high-dimensional data; L1 selects features automatically.  
4. For sparse data, leverage SciPy matrices for efficiency.  
5. Monitor `loss_history` to check convergence; adjust `max_iter` if needed.  
6. Plot loss curve:

```python
import matplotlib.pyplot as plt
plt.plot(model.loss_history)
plt.title('Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
```

## Error Handling

- **NaN/Inf** in `X` or `y` → `ValueError`  
- **Shape mismatch** → `ValueError`  

## Performance Notes

| Aspect | L1R |
|--------|------|
| **Speed** | Coordinate Descent – efficient for high-dim (O(n p) per iter) |
| **Memory** | Sparse-friendly (`csr_matrix`, `csc_matrix`) |
| **Scalability** | Good for sparse/high-dim; iterative for large n |

## Comparison with scikit-learn `Lasso`

| Feature | L1R | scikit-learn `Lasso` |
|---------|------|-----------------------------------|
| **Regularization** | ✅ L1 only | ✅ L1 only |
| **Solver** | ✅ Coordinate Descent | ✅ Coordinate Descent |
| **Sparse Input Support** | ✅ Full (`csr`, `csc`) | ✅ Full (`csr`, `csc`) |
| **Intercept** | ✅ fit_intercept | ✅ fit_intercept |
| **Pandas Support** | ✅ Direct (to_numpy) | ✅ Indirect |
| **Early Stopping** | ✅ Built-in (`tol`) | ✅ Partial (via `tol`) |
| **Verbose Levels** | ✅ 0/1/2 | ✅ Basic |
| **Loss History** | ✅ `loss_history` | ❌ Not exposed |
| **Customizability** | ✅ Full CD loop | ❌ Limited (black-box) |

> **Note**: `Lasso` is more optimized and handles warm starts. L1R excels in **transparency**, **teaching**, and **simple implementation**.

## License

Part of the **NexGML** package – MIT (or as specified in the repository).
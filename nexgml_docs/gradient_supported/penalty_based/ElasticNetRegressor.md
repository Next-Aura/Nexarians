# ENR – ElasticNet Regressor

## Overview

ENR (ElasticNet Regressor) is a linear regression model that combines L1 (Lasso) and L2 (Ridge) regularization to balance sparsity and smoothness. It optimizes using Coordinate Descent, updating one coefficient at a time. Supports dense, sparse, and pandas inputs.

Ideal for teaching, prototyping, or when you need a flexible regressor that can select features and handle correlated data.

## Installation & Requirements

```bash
pip install numpy scipy pandas
```

```python
from nexgml.gradient_supported import ElasticNetRegressor
```

## Mathematical Formulation

### Prediction Function
$$
\hat{y} = Xw + b
$$

- $X$ – feature matrix  
- $w$ – weight vector  
- $b$ – bias (if `fit_intercept=True`)

### Loss Function

- **MSE (Mean Squared Error) with ElasticNet Regularization**:  
  $$
  \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2 + \alpha \left[ \text{l1\_ratio} \sum |w_i| + (1 - \text{l1\_ratio}) \sum w_i^2 \right]
  $$

### Regularization

- **ElasticNet**: Mixes L1 and L2 penalties via `l1_ratio`  
  - L1: $\alpha \cdot \text{l1\_ratio} \sum |w_i|$  
  - L2: $\alpha \cdot (1 - \text{l1\_ratio}) \sum w_i^2$

### Optimization (Coordinate Descent)

Iteratively update each $w_j$:

$$
w_j = \frac{\text{soft\_threshold}(\rho, \lambda_1)}{z + \lambda_2}
$$

Where:
- $\rho$ is the correlation between $X_j$ and the residual
- $z$ is the squared norm of $X_j$
- $\lambda_1 = \alpha \cdot \text{l1\_ratio}$
- $\lambda_2 = \alpha \cdot (1 - \text{l1\_ratio})$

Soft-threshold: $\text{sign}(\rho) \max(|\rho| - \lambda_1, 0)$

Converges when coefficient changes < `tol`.

## Key Features
- **Regularization**: ElasticNet (L1 + L2, tunable via `l1_ratio`)
- **Input**: dense `np.ndarray`, `pd.DataFrame`, **or** SciPy sparse matrices
- **Optimization**: Coordinate Descent
- **Early stopping** on convergence (`tol`)
- **Verbose levels** (0/1/2)
- **Intercept**: Optional via `fit_intercept`

## Parameters

| Parameter      | Type    | Default | Description                                 |
|----------------|---------|---------|---------------------------------------------|
| `max_iter`     | `int`   | `100`   | Max iterations for Coordinate Descent       |
| `alpha`        | `float` | `1e-4`  | Regularization strength                     |
| `l1_ratio`     | `float` | `0.5`   | Mix between L1 (sparsity) and L2 (smooth)   |
| `fit_intercept`| `bool`  | `True`  | Add bias term                               |
| `tol`          | `float` | `1e-4`  | Convergence tolerance                       |
| `early_stopping`| `bool` | `True`  | Enable early stop on convergence            |
| `verbose`      | `int`   | `0`     | 0 = silent, 1 = ~5 % progress, 2 = every iteration |

## Model Attributes (post-fit)

| Attribute      | Type             | Description                                 |
|----------------|------------------|---------------------------------------------|
| `weights`      | `np.ndarray`     | Learned coefficients                        |
| `b`            | `float \| np.ndarray` | Bias (intercept); scalar or array for multi-output |
| `n_outputs_`   | `int`            | Number of output dimensions                 |
| `loss_history` | `List[float]`    | Residual per iteration                      |

## API Reference

### `ElasticNetRegressor.__init__(...)`
Creates the model with the hyper-parameters above.

### `fit(X_train, y_train)`
Fits using Coordinate Descent.

- **Raises** `ValueError` for NaN/Inf or shape mismatch
- **Raises** `RuntimeWarning` if there's a NaN value that clipped

### `predict(X_test)`
Returns $\hat{y}$ for new samples.

- **Raises** `ValueError` if model not fitted or NaN/Inf

### `score(X_test, y_test)`
Returns R² score.

## Usage Examples

### 1. Default (with intercept)
```python
import numpy as np
from sklearn.datasets import make_regression
from nexgml.gradient_supported import ElasticNetRegressor

X, y = make_regression(n_samples=200, n_features=10, noise=0.2, random_state=42)

model = ElasticNetRegressor(alpha=0.05, l1_ratio=0.7, max_iter=200, verbose=1)
model.fit(X, y)

print(f"Loss: {model.loss_history[-1]:.6f}")
print(f"Weights (mean): {model.weights.mean():.6f}, bias: {model.b.squeeze():.6f}")
```

### 2. No intercept + pandas
```python
import pandas as pd

X_df = pd.DataFrame(X)
y_series = pd.Series(y)

model = ElasticNetRegressor(fit_intercept=False, alpha=0.01, l1_ratio=0.3, tol=1e-5)
model.fit(X_df, y_series)
```

### 3. Sparse data
```python
from scipy.sparse import csr_matrix

X_sp = csr_matrix(np.random.randn(500, 200))
y_sp = np.random.randn(500)

model = ElasticNetRegressor(alpha=0.001, l1_ratio=0.5, early_stopping=False, verbose=2)
model.fit(X_sp, y_sp)
```

## Best Practices

1. **Scale features** (`StandardScaler`) → improves convergence and sparsity.
2. Tune `alpha` and `l1_ratio` via cross-validation: higher `alpha` increases regularization, higher `l1_ratio` increases sparsity.
3. Use for high-dimensional or correlated data; ElasticNet balances feature selection and stability.
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

| Aspect      | ENR                                      |
|-------------|------------------------------------------|
| **Speed**   | Coordinate Descent – moderate for high-dim, slower than L1/L2 only |
| **Memory**  | Sparse-friendly (`csr_matrix`, `csc_matrix`) |
| **Scalability** | Good for sparse/high-dim; iterative for large n |

## Comparison with scikit-learn `ElasticNet`

| Feature                | ENR | scikit-learn `ElasticNet` |
|------------------------|-----|---------------------------|
| **Regularization**     | ✅ L1 + L2 (tunable) | ✅ L1 + L2 (tunable) |
| **Solver**             | ✅ Coordinate Descent | ✅ Coordinate Descent |
| **Sparse Input Support** | ✅ Full (`csr`, `csc`) | ✅ Full (`csr`, `csc`) |
| **Intercept**          | ✅ fit_intercept | ✅ fit_intercept |
| **Pandas Support**     | ✅ Direct (to_numpy) | ✅ Indirect |
| **Early Stopping**     | ✅ Built-in (`tol`) | ✅ Partial (via `tol`) |
| **Verbose Levels**     | ✅ 0/1/2 | ✅ Basic |
| **Loss History**       | ✅ `loss_history` | ❌ Not exposed |
| **Customizability**    | ✅ Full CD loop | ❌ Limited (black-box) |

> **Note**: scikit-learn is more optimized and handles warm starts. ENR excels in **transparency**, **teaching**, and **simple implementation**.

## License

Part of the **NexGML** package – MIT (or as specified in the repository).
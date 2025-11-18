# L2R – L2 Regressor

## Overview

L2R (L2 Regressor), also known as Ridge Regression, is a linear regression model that uses L2 regularization (Tikhonov regularization) to prevent overfitting and handle multicollinearity in the data. It finds the optimal weights using the closed-form solution (Normal Equation) with a penalty term. It supports dense, sparse, and pandas inputs.

Perfect for teaching, quick prototyping, or when you need a stable, interpretable regressor without heavy dependencies.

## Installation & Requirements

```bash
pip install numpy scipy pandas
```

```python
from nexgml.gradient_supported import L2Regressor
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

- **MSE (Mean Squared Error) with L2 Regularization**: $\frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2 + \alpha \sum w_i^2$

### Regularization

- **L2 (Ridge)**: $\alpha \sum w_i^2$ (added to MSE loss; intercept not regularized)

### Closed-Form Solution
$$
w = (X^T X + \alpha I)^{-1} X^T y
$$
(Adjusted for intercept by augmenting $X$ with column of ones; $\alpha$ not applied to intercept row.)

Uses `np.linalg.solve` or fallback to pseudoinverse if singular.

## Key Features
- **Regularization**: L2 (Ridge)  
- **Input**: dense `np.ndarray`, `pd.DataFrame`, **or** SciPy sparse matrices  
- **Optimization**: Closed-form Normal Equation  
- **Intercept**: Optional via `fit_intercept`  

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | `float` | `1e-4` | Regularization strength |
| `fit_intercept` | `bool` | `True` | Add bias term |

## Model Attributes (post-fit)

| Attribute | Type | Description |
|-----------|------|-------------|
| `weights` | `np.ndarray` | Learned coefficients |
| `b` | `np.ndarray` | Bias (intercept) |
| `loss_history` | `List[float]` | Loss history (e.g., final residual) |

## API Reference

### `L2Regressor.__init__(...)`
Creates the model with the hyper-parameters above.

### `fit(X_train, y_train)`
Fits using closed-form solution.

- **Raises** `ValueError` for NaN/Inf or shape mismatch  
- **Raises** `np.linalg.LinAlgError` if matrix singular (fallback to pinv)

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
from nexgml.gradient_supported import L2Regressor

X, y = make_regression(n_samples=200, n_features=10, noise=0.2, random_state=42)

model = L2Regressor(alpha=0.05)
model.fit(X, y)

print(f"Weights (mean): {model.weights.mean():.6f}, bias: {model.b[0][0]:.6f}")
```

### 2. No intercept + pandas
```python
import pandas as pd

X_df = pd.DataFrame(X)
y_series = pd.Series(y)

model = L2Regressor(fit_intercept=False, alpha=0.01)
model.fit(X_df, y_series)
```

### 3. Sparse data
```python
from scipy.sparse import csr_matrix

X_sp = csr_matrix(np.random.randn(500, 200))
y_sp = np.random.randn(500)

model = L2Regressor(alpha=0.001)
model.fit(X_sp, y_sp)
```

## Best Practices

1. **Scale features** (`StandardScaler`) → improves stability with regularization.  
2. Tune `alpha` via cross-validation: small for low regularization, large for high.  
3. Use for multicollinear data; L2 shrinks coefficients without zeroing them.  
4. For sparse data, leverage SciPy matrices for efficiency.  

## Error Handling

- **NaN/Inf** in `X` or `y` → `ValueError`  
- **Shape mismatch** → `ValueError`  
- **Singular matrix** → `np.linalg.LinAlgError` (handled with pinv)  

## Performance Notes

| Aspect | L2R |
|--------|------|
| **Speed** | Closed-form – fast for small-medium data (O(n^3) inversion) |
| **Memory** | Sparse-friendly (`csr_matrix`, `csc_matrix`) |
| **Scalability** | Limited for very large n/p due to inversion; use iterative solvers for big data |

## Comparison with scikit-learn `Ridge`

| Feature | L2R | scikit-learn `Ridge` |
|---------|------|-----------------------------------|
| **Regularization** | ✅ L2 only | ✅ L2 only |
| **Solver** | ✅ Closed-form (solve/pinv) | ✅ Cholesky / SVD / SAG / etc. |
| **Sparse Input Support** | ✅ Full (`csr`, `csc`) | ✅ Full (`csr`, `csc`) |
| **Intercept** | ✅ fit_intercept | ✅ fit_intercept |
| **Pandas Support** | ✅ Direct (to_numpy) | ✅ Indirect |
| **Loss History** | ✅ `loss_history` | ❌ Not exposed |
| **Customizability** | ✅ Simple matrix ops | ❌ Limited (black-box solvers) |

> **Note**: `Ridge` offers more solvers for large data. L2R excels in **simplicity**, **teaching**, and **closed-form transparency**.

## License

Part of the **NexGML** package – MIT (or as specified in the repository).
# ENC – ElasticNet Classifier

## Overview

ENC (ElasticNet Classifier) is a linear classification model that minimizes a least-squares loss with ElasticNet regularization (L1 + L2) using Coordinate Descent. It handles multi-class via one-hot encoding, treating it as multiple regression problems with sparsity and smoothness-inducing penalty. It supports dense, sparse, and pandas inputs.

Perfect for teaching, quick prototyping, or when you need flexible, interpretable classifiers that balance feature selection and stability.

## Installation & Requirements

```bash
pip install numpy scipy pandas
```

```python
from nexgml.gradient_supported import ElasticNetClassifier
```

## Mathematical Formulation

### Prediction Function
$$
z = Xw + b
$$

- $X$ – feature matrix  
- $w$ – weight matrix (features × classes)  
- $b$ – bias vector (if `fit_intercept=True`)
- Predict class as $\arg\max_c z_c$

### Loss Functions

- **MSE (Mean Squared Error) with ElasticNet Regularization**:  
  $$
  \frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C} (y_{i,c} - z_{i,c})^2 + \alpha \left[ \text{l1\_ratio} \sum |w_{i,j}| + (1 - \text{l1\_ratio}) \sum w_{i,j}^2 \right]
  $$
  (where $y$ is one-hot encoded)

### Regularization

- **ElasticNet**: Mixes L1 and L2 penalties via `l1_ratio`  
  - L1: $\alpha \cdot \text{l1\_ratio} \sum |w_{i,j}|$  
  - L2: $\alpha \cdot (1 - \text{l1\_ratio}) \sum w_{i,j}^2$

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
- **Multi-class**: Via one-hot least squares
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
| `weights`      | `np.ndarray`     | Learned coefficients (features × classes)   |
| `b`            | `float \| np.ndarray` | Bias (intercept); scalar or array for multi-class |
| `classes`      | `np.ndarray`     | Unique class labels                         |
| `n_classes`    | `int`            | Number of unique classes                    |
| `loss_history` | `List[float]`    | Residual per iteration                      |

## API Reference

### `ElasticNetClassifier.__init__(...)`
Creates the model with the hyper-parameters above.

### `fit(X_train, y_train)`
Fits using Coordinate Descent.

- **Raises** `ValueError` for NaN/Inf or shape mismatch
- **Raises** `RuntimeWarning` if there's a NaN value that clipped

### `predict(X_test)`
Returns predicted class labels (argmax of scores).

- **Raises** `ValueError` if model not fitted or NaN/Inf

### `score(X_test, y_test)`
Returns mean accuracy.

## Usage Examples

### 1. Default (with intercept)
```python
import numpy as np
from sklearn.datasets import make_classification
from nexgml.gradient_supported import ElasticNetClassifier

X, y = make_classification(n_samples=200, n_features=10, n_classes=3, n_informative=5, random_state=42)

model = ElasticNetClassifier(alpha=0.05, l1_ratio=0.7, max_iter=200, verbose=1)
model.fit(X, y)

print(f"Loss: {model.loss_history[-1]:.6f}")
print(f"Weights (mean): {model.weights.mean():.6f}, bias (mean): {np.mean(model.b):.6f}")
```

### 2. No intercept + pandas
```python
import pandas as pd

X_df = pd.DataFrame(X)
y_series = pd.Series(y)

model = ElasticNetClassifier(fit_intercept=False, alpha=0.01, l1_ratio=0.3, tol=1e-5)
model.fit(X_df, y_series)
```

### 3. Sparse data
```python
from scipy.sparse import csr_matrix

X_sp = csr_matrix(np.random.randn(500, 200))
y_sp = np.random.randint(0, 3, 500)  # 3 classes

model = ElasticNetClassifier(alpha=0.001, l1_ratio=0.5, early_stopping=False, verbose=2)
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

| Aspect      | ENC                                      |
|-------------|------------------------------------------|
| **Speed**   | Coordinate Descent – moderate for high-dim, slower than L1/L2 only |
| **Memory**  | Sparse-friendly (`csr_matrix`, `csc_matrix`) |
| **Scalability** | Good for sparse/high-dim; iterative for large n |

## Comparison with scikit-learn `SGDClassifier` (with elasticnet penalty)

| Feature | ENC | scikit-learn `SGDClassifier` (elasticnet) |
|---------|-----|-------------------------------------------|
| **Regularization** | ✅ L1 + L2 (tunable) | ✅ L1 + L2 (tunable) |
| **Loss Functions** | ✅ MSE (least-squares) | ✅ Log / Hinge / Squared Error / etc. |
| **Solver** | ✅ Coordinate Descent | ✅ SGD (mini-batch) |
| **Sparse Input Support** | ✅ Full (`csr`, `csc`) | ✅ Full (`csr`, `csc`) |
| **Early Stopping** | ✅ Built-in (`tol`) | ✅ Built-in (`tol`, `n_iter_no_change`) |
| **Verbose Levels** | ✅ 0/1/2 | ✅ Basic (via `verbose`) |
| **Multi-class** | ✅ One-hot least squares | ✅ OVR / Multinomial |
| **Loss History** | ✅ `loss_history` | ❌ Not exposed |
| **Customizability** | ✅ Full CD loop | ❌ Limited (black-box solver) |

> **Note**: `SGDClassifier` is faster for large datasets with stochastic updates. ENC excels in **coordinate descent transparency**, **elasticnet for classification**, **teaching**, and **custom logic**.

## License

Part of the **NexGML** package – MIT (or as specified in the repository).
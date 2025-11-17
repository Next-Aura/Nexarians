# L1C – L1 Classifier

## Overview

L1C (L1 Classifier) is a linear classification model that minimizes a least-squares loss with L1 regularization using Coordinate Descent. It handles multi-class via one-hot encoding, treating it as multiple regression problems with sparsity-inducing penalty. It supports dense, sparse, and pandas inputs.

Perfect for teaching, quick prototyping, or when you need sparse, interpretable classifiers without heavy dependencies.

## Installation & Requirements

```bash
pip install numpy scipy pandas
```

```python
from nexgml.gradient_supported import L1Classifier
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

- **MSE (Mean Squared Error) with L1 Regularization**: $\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C} (y_{i,c} - z_{i,c})^2 + \alpha \sum |w_{i,j}|$  
(where $y$ is one-hot encoded)

### Regularization

- **L1 (Lasso)**: $\alpha \sum |w_{i,j}|$ (promotes sparsity)

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
- **Multi-class**: Via one-hot least squares  
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
| `weights` | `np.ndarray` | Learned coefficients (features × classes) |
| `b` | `float \| np.ndarray` | Bias (intercept); scalar or array for multi-class |
| `classes` | `np.ndarray` | Unique class labels |
| `n_classes` | `int` | Number of unique classes |
| `loss_history` | `List[float]` | Residual per iteration |

## API Reference

### `L1Classifier.__init__(...)`
Creates the model with the hyper-parameters above.

### `fit(X_train, y_train)`
Fits using Coordinate Descent.

- **Raises** `ValueError` for NaN/Inf or shape mismatch

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
from nexgml.gradient_supported import L1Classifier

X, y = make_classification(n_samples=200, n_features=10, n_classes=3, n_informative=5, random_state=42)

model = L1Classifier(alpha=0.05, max_iter=200, verbose=1)
model.fit(X, y)

print(f"Loss: {model.loss_history[-1]:.6f}")
print(f"Weights (mean): {model.weights.mean():.6f}, bias (mean): {np.mean(model.b):.6f}")
```

### 2. No intercept + pandas
```python
import pandas as pd

X_df = pd.DataFrame(X)
y_series = pd.Series(y)

model = L1Classifier(fit_intercept=False, alpha=0.01, tol=1e-5)
model.fit(X_df, y_series)
```

### 3. Sparse data
```python
from scipy.sparse import csr_matrix

X_sp = csr_matrix(np.random.randn(500, 200))
y_sp = np.random.randint(0, 3, 500)  # 3 classes

model = L1Classifier(alpha=0.001, early_stopping=False, verbose=2)
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

| Aspect | L1C |
|--------|------|
| **Speed** | Coordinate Descent – efficient for high-dim (O(n p) per iter) |
| **Memory** | Sparse-friendly (`csr_matrix`, `csc_matrix`) |
| **Scalability** | Good for sparse/high-dim; iterative for large n |

## Comparison with scikit-learn `SGDClassifier` (with l1 penalty)

| Feature | L1C | scikit-learn `SGDClassifier` (l1) |
|---------|------|-----------------------------------|
| **Regularization** | ✅ L1 only | ✅ L1 / L2 / ElasticNet / None |
| **Loss Functions** | ✅ MSE (least-squares) | ✅ Log / Hinge / Squared Error / etc. |
| **Solver** | ✅ Coordinate Descent | ✅ SGD (mini-batch) |
| **Sparse Input Support** | ✅ Full (`csr`, `csc`) | ✅ Full (`csr`, `csc`) |
| **Early Stopping** | ✅ Built-in (`tol`) | ✅ Built-in (`tol`, `n_iter_no_change`) |
| **Verbose Levels** | ✅ 0/1/2 | ✅ Basic (via `verbose`) |
| **Multi-class** | ✅ One-hot least squares | ✅ OVR / Multinomial |
| **Loss History** | ✅ `loss_history` | ❌ Not exposed |
| **Customizability** | ✅ Full CD loop | ❌ Limited (black-box solver) |

> **Note**: `SGDClassifier` is faster for large datasets with stochastic updates. L1C excels in **coordinate descent sparsity**, **least-squares for classification**, **teaching**, and **custom logic**.

## License

Part of the **NexGML** package – MIT (or as specified in the repository).
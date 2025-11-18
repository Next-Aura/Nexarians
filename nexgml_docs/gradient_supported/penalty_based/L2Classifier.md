# L2C – L2 Classifier

## Overview

L2C (L2 Classifier), also known as Ridge Classifier, is a linear classification model that minimizes a least-squares loss with L2 regularization using a closed-form solution (Normal Equation). It handles multi-class via one-hot encoding, treating it as multiple regression problems. It supports dense, sparse, and pandas inputs.

Perfect for teaching, quick prototyping, or when you need a stable, interpretable classifier without heavy dependencies.

## Installation & Requirements

```bash
pip install numpy scipy pandas
```

```python
from nexgml.gradient_supported import L2Classifier
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

- **MSE (Mean Squared Error) with L2 Regularization**: $\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C} (y_{i,c} - z_{i,c})^2 + \alpha \sum w_{i,j}^2$  
(where $y$ is one-hot encoded)

### Regularization

- **L2 (Ridge)**: $\alpha \sum w_{i,j}^2$ (intercept not regularized)

### Closed-Form Solution
$$
W = (X^T X + \alpha I)^{-1} X^T Y
$$
(Adjusted for intercept by augmenting $X$; $Y$ one-hot; $\alpha$ not on intercept.)

Uses `np.linalg.solve` or fallback to pseudoinverse if singular.

## Key Features
- **Regularization**: L2 (Ridge)  
- **Input**: dense `np.ndarray`, `pd.DataFrame`, **or** SciPy sparse matrices  
- **Optimization**: Closed-form Normal Equation  
- **Multi-class**: Via one-hot least squares  
- **Intercept**: Optional via `fit_intercept`  

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | `float` | `1e-4` | Regularization strength |
| `fit_intercept` | `bool` | `True` | Add bias term |

## Model Attributes (post-fit)

| Attribute | Type | Description |
|-----------|------|-------------|
| `weights` | `np.ndarray` | Learned coefficients (features × classes) |
| `b` | `np.ndarray` | Bias vector |
| `classes` | `np.ndarray` | Unique class labels |
| `n_classes` | `int` | Number of unique classes |
| `loss_history` | `List[float]` | Loss history (e.g., final residual) |

## API Reference

### `L2Classifier.__init__(...)`
Creates the model with the hyper-parameters above.

### `fit(X_train, y_train)`
Fits using closed-form solution.

- **Raises** `ValueError` for NaN/Inf or shape mismatch  
- **Raises** `np.linalg.LinAlgError` if matrix singular (fallback to pinv)

### `predict(X_test)`
Returns predicted class labels (argmax of scores).

- **Raises** `ValueError` if model not fitted or NaN/Inf

### `score(X_test, y_test)`
Returns mean accuracy.

### `get_params(deep)`
Returns model paramters

### `set_params(**params)`
Returns model's attribute that ready to set

## Usage Examples

### 1. Default (with intercept)
```python
import numpy as np
from sklearn.datasets import make_classification
from nexgml.gradient_supported import L2Classifier

X, y = make_classification(n_samples=200, n_features=10, n_classes=3, n_informative=5, random_state=42)

model = L2Classifier(alpha=0.05)
model.fit(X, y)

print(f"Weights (mean): {model.weights.mean():.6f}, bias (mean): {np.mean(model.b):.6f}")
```

### 2. No intercept + pandas
```python
import pandas as pd

X_df = pd.DataFrame(X)
y_df = pd.DataFrame(y)  # or pd.Series(y)

model = L2Classifier(fit_intercept=False, alpha=0.01)
model.fit(X_df, y_df)
```

### 3. Sparse data
```python
from scipy.sparse import csr_matrix

X_sp = csr_matrix(np.random.randn(500, 200))
y_sp = np.random.randint(0, 3, 500)  # 3 classes

model = L2Classifier(alpha=0.001)
model.fit(X_sp, y_sp)
```

## Best Practices

1. **Scale features** (`StandardScaler`) → improves stability with regularization.  
2. Tune `alpha` via cross-validation: small for low regularization, large for high.  
3. Use for multicollinear data; L2 shrinks coefficients without zeroing them.  
4. For sparse data, leverage SciPy matrices for efficiency.  
5. Suitable for multi-class; efficient for moderate-sized data.

## Error Handling

- **NaN/Inf** in `X` or `y` → `ValueError`  
- **Shape mismatch** → `ValueError`  
- **Singular matrix** → `np.linalg.LinAlgError` (handled with pinv)  

## Performance Notes

| Aspect | L2C |
|--------|------|
| **Speed** | Closed-form – fast for small-medium data (O(n^3) inversion) |
| **Memory** | Sparse-friendly (`csr_matrix`, `csc_matrix`) |
| **Scalability** | Limited for very large n/p due to inversion; use iterative solvers for big data |

## Comparison with scikit-learn `RidgeClassifier`

| Feature | L2C | scikit-learn `RidgeClassifier` |
|---------|------|-----------------------------------|
| **Regularization** | ✅ L2 only | ✅ L2 only |
| **Solver** | ✅ Closed-form (solve/pinv) | ✅ LSQR / SAG / etc. |
| **Sparse Input Support** | ✅ Full (`csr`, `csc`) | ✅ Full (`csr`, `csc`) |
| **Intercept** | ✅ fit_intercept | ✅ fit_intercept |
| **Multi-class** | ✅ One-hot least squares | ✅ One-vs-rest (binary) or multi-output |
| **Pandas Support** | ✅ Direct (to_numpy) | ✅ Indirect |
| **Loss History** | ✅ `loss_history` | ❌ Not exposed |
| **Customizability** | ✅ Simple matrix ops | ❌ Limited (black-box solvers) |

> **Note**: `RidgeClassifier` offers more solvers for large data. L2C excels in **simplicity**, **teaching**, and **closed-form transparency**.

## License

Part of the **NexGML** package – MIT (or as specified in the repository).
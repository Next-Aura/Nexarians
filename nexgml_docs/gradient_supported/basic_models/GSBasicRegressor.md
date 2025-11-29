# GSBR – Gradient Supported Basic Regressor

## Overview

GSBR (Gradient Supported Basic Regressor) is a lightweight, custom linear regression model implemented in Python. It supports optimization via **gradient descent**, includes regularization options such as **L1 (Lasso)**, **L2 (Ridge)**, and **ElasticNet**, and also learning rate schedulers (**constant**, **invscaling**, **plateau**, **adaptive**) to prevent overfitting. The model can minimize **MSE**, **RMSE**, or **MAE** loss functions. It works with both dense and sparse matrices, offers early stopping, data shuffling, and multi-level verbose logging.

Perfect for teaching, quick prototyping, or when you need a simple, interpretable regressor without heavy dependencies.

## Installation & Requirements

```bash
pip install numpy scipy
```

```python
from nexgml.gradient_supported import BasicRegressor
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

- **MSE (Mean Squared Error)**: $\frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2$
- **RMSE (Root Mean Squared Error)**: $\sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2}$
- **MAE (Mean Absolute Error)**: $\frac{1}{N}\sum_{i=1}^{N}|y_i - \hat{y}_i|$

### Regularization
Added to the loss to control model complexity:

- **L1 (Lasso)**: $\alpha \sum |w_i|$
- **L2 (Ridge)**: $\alpha \sum w_i^2$
- **ElasticNet**: $\alpha \bigl[ l1\_ratio \sum |w_i| + (1-l1\_ratio)\sum w_i^2 \bigr]$

Total loss = *base loss* + *regularization term*.

### Gradients (example for MSE)
$$
\frac{\partial L}{\partial w} = \frac{2}{N}X^{T}(Xw + b - y) + \text{regularization gradient}
$$
$$
\frac{\partial L}{\partial b}= \frac{2}{N}\sum (Xw + b - y)
$$

MAE uses the **sign** function; RMSE normalises by the current RMSE value.

## Key Features
- **Regularization**: L1 / L2 / ElasticNet / None  
- **Losses**: MSE, RMSE, MAE
- **Input**: dense `np.ndarray` **or** SciPy sparse matrices  
- **Optimization**: gradient descent with learning-rate control
- **LR Schedulers**: constant / invscaling / plateau / adaptive
- **Early stopping** on loss convergence (`tol`)  
- **Shuffling** + `random_state` for reproducibility  
- **Verbose levels** (0/1/2)  

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_iter` | `int` | `1000` | Max gradient-descent steps |
| `learning_rate` | `float` | `0.01` | Step size |
| `penalty` | `Literal['l1','l2','elasticnet'] \| None` | `'l2'` | Regularization type |
| `alpha` | `float` | `0.0001` | Regularization strength |
| `l1_ratio` | `float` | `0.5` | ElasticNet mix (0 = L2, 1 = L1) |
| `loss` | `Literal['mse','rmse','mae']` | `'mse'` | Loss function |
| `fit_intercept` | `bool` | `True` | Add bias term |
| `tol` | `float` | `0.0001` | Early-stop tolerance |
| `shuffle` | `bool` | `True` | Shuffle data each epoch |
| `random_state` | `int \| None` | `None` | Seed for shuffling |
| `early_stopping` | `bool` | `True` | Enable early stop |
| `verbose` | `int` | `0` | 0 = silent, 1 = ~5 % progress, 2 = every epoch |
| `verbosity` | `Literal['light', 'heavy']` | `light` | light = standard log information, heavy = more detail log information |
| `lr_scheduler` | `Literal['constant','invscaling','plateau', 'adative']` | `'invscaling'` | Type of learning rate scheduler |
| `power_t` | `float` | `0.25` | Exponent for invscaling |
| `patience` | `int` | `5` | Epochs to wait for plateau |
| `factor` | `float` | `0.5` | LR reduction factor for plateau |
| `stoic_iter` | `int` | `10` | Warm-up epochs before early stop/scheduler |
| `epsilon` | `float` | `1e-15` | Small value for numerical stability |

## Model Attributes (post-fit)

| Attribute | Type | Description |
|-----------|------|-------------|
| `weights` | `np.ndarray` | Learned coefficients |
| `b` | `float` | Bias (intercept) |
| `loss_history` | `List[float]` | Loss per iteration |

## API Reference

### `BasicRegressor.__init__(...)`
Creates the model with the hyper-parameters above.

### `fit(X_train, y_train)`
Trains via gradient descent.

- **Raises** `ValueError` for NaN/Inf or shape mismatch  
- **Raises** `OverflowError` if weights/bias become NaN/Inf

### `predict(X_test)`
Returns $\hat{y}$ for new samples.

- **Raises** `ValueError` if model not fitted

### `score(X_test, y_test)`
Returns R² score.

## Usage Examples

### 1. L2 + MSE (default)
```python
import numpy as np
from sklearn.datasets import make_regression
from nexgml.gradient_supported import BasicRegressor

X, y = make_regression(n_samples=200, n_features=10, noise=0.2, random_state=42)

model = BasicRegressor(max_iter=1500, learning_rate=0.02,
                       penalty='l2', alpha=0.05, verbose=1)
model.fit(X, y)

print(f"Loss: {model.loss_history[-1]:.6f}")
print(f"Weights (mean): {model.weights.mean():.6f}, bias: {model.b:.6f}")
```

### 2. ElasticNet + MAE + scaling
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

model = BasicRegressor(penalty='elasticnet', alpha=0.01, l1_ratio=0.7,
                       loss='mae', max_iter=3000, learning_rate=0.005,
                       shuffle=True, random_state=123, verbose=2)
model.fit(X_sc, y)
```

### 3. Sparse data (no regularisation)
```python
from scipy.sparse import csr_matrix

X_sp = csr_matrix(np.random.randn(500, 200))
y_sp = np.random.randn(500)

model = BasicRegressor(penalty=None, max_iter=800, learning_rate=0.03)
model.fit(X_sp, y_sp)
```

## Best Practices

1. **Scale features** (`StandardScaler`) → faster convergence.  
2. Start with `learning_rate ∈ [0.001, 0.1]`; monitor `loss_history`.  
3. Use `early_stopping=True` + `tol=1e-4`.  
4. For high-dimensional data, keep `penalty='l1'` or `'elasticnet'`.  
5. Plot loss curve:

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
- **Numerical overflow** during training → `OverflowError` (stops early)  

## Performance Notes

| Aspect | GSBR |
|--------|------|
| **Speed** | Batch GD – good for ≤ 10 k samples |
| **Memory** | Sparse-friendly (`csr_matrix`, `csc_matrix`) |
| **Scalability** | Extend to mini-batch for > 100 k rows |

## Comparison with scikit-learn

| Feature | GSBR | scikit-learn |
|---------|------|--------------|
| Regularization | L1/L2/ElasticNet/None | Yes |
| Loss options | MSE/RMSE/MAE | MSE only |
| Solver | Gradient descent | Analytical (OLS, SVD) |
| Early stopping | Yes | Limited |
| Shuffling / seed | Yes | No |
| Verbose levels | 0/1/2 | Basic |

## License

Part of the **NexGML** package – MIT (or as specified in the repository).
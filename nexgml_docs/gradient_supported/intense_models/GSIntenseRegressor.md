# GSIR – Gradient Supported Intense Regressor

## Overview

GSIR (Gradient Supported Intense Regressor) is an advanced, custom linear regression model implemented in Python. It supports optimization via **mini-batch gradient descent** and includes regularization options such as **L1 (Lasso)**, **L2 (Ridge)**, and **ElasticNet** to prevent overfitting. The model can minimize **MSE**, **MAE**, or **Smooth L1 (Huber)** loss functions and offers multiple optimizers (**MBGD**, **Adam**, **AdamW**) and learning rate schedulers (**constant**, **invscaling**, **plateau**). It works with both dense and sparse matrices, offers early stopping, data shuffling, and multi-level verbose logging.

Perfect for teaching, quick prototyping, or when you need a flexible, interpretable regressor with advanced optimization without heavy dependencies.

## Installation & Requirements

```bash
pip install numpy scipy
```

```python
from nexgml.gradient_supported import IntenseRegressor
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
- **Smooth L1 (Huber)**: $\frac{1}{N}\sum_{i=1}^{N} \begin{cases} 
  0.5 (y_i - \hat{y}_i)^2 & |y_i - \hat{y}_i| \leq \delta \\ 
  \delta |y_i - \hat{y}_i| - 0.5 \delta^2 & \text{otherwise}
  \end{cases}$

### Regularization
Added to the loss to control model complexity (applied in loss or update step for AdamW):

- **L1 (Lasso)**: $\alpha \sum |w_i|$
- **L2 (Ridge)**: $\alpha \sum w_i^2$
- **ElasticNet**: $\alpha \bigl[ l1\_ratio \sum |w_i| + (1-l1\_ratio)\sum w_i^2 \bigr]$

Total loss = *base loss* + *regularization term*.

### Gradients (example for MSE)
$$
\frac{\partial L}{\partial w} = \frac{2}{N}X^{T}(Xw + b - y) + \text{regularization gradient (applied in update)}
$$
$$
\frac{\partial L}{\partial b}= \frac{2}{N}\sum (Xw + b - y)
$$

MAE uses the **sign** function; RMSE normalises by the current RMSE value; Smooth L1 uses conditional gradient.

## Key Features
- **Regularization**: L1 / L2 / ElasticNet / None  
- **Losses**: MSE / RMSE / MAE / Smooth L1 (Huber)  
- **Input**: dense `np.ndarray` **or** SciPy sparse matrices  
- **Optimization**: mini-batch gradient descent with MBGD / Adam / AdamW  
- **LR Schedulers**: constant / invscaling / plateau  
- **Early stopping** on loss convergence (`tol`)  
- **Shuffling** + `random_state` for reproducibility  
- **Verbose levels** (0/1/2)  
- **Mini-batch** processing for efficiency  

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_iter` | `int` | `1000` | Max training epochs |
| `learning_rate` | `float` | `0.01` | Initial step size |
| `penalty` | `Literal['l1','l2','elasticnet'] \| None` | `'l2'` | Regularization type |
| `alpha` | `float` | `0.001` | Regularization strength |
| `l1_ratio` | `float` | `0.5` | ElasticNet mix (0 = L2, 1 = L1) |
| `loss` | `Literal['mse','mae','smoothl1']` | `'mse'` | Loss function |
| `fit_intercept` | `bool` | `True` | Add bias term |
| `tol` | `float` | `0.0001` | Early-stop tolerance |
| `shuffle` | `bool` | `True` | Shuffle data each epoch |
| `random_state` | `int \| None` | `None` | Seed for shuffling |
| `early_stopping` | `bool` | `True` | Enable early stop |
| `verbose` | `int` | `0` | 0 = silent, 1 = ~5 % progress, 2 = every epoch + LR updates |
| `verbosity` | `Literal['light', 'heavy']` | `'light'` | light = standard log information, heavy = more detail log information |
| `lr_scheduler` | `Literal['constant','invscaling','plateau']` | `'invscaling'` | Type of learning rate scheduler |
| `optimizer` | `Literal['mbgd','adam','adamw']` | `'mbgd'` | Optimizer type |
| `batch_size` | `int` | `16` | Mini-batch size |
| `power_t` | `float` | `0.25` | Exponent for invscaling |
| `patience` | `int` | `5` | Epochs to wait for plateau |
| `factor` | `float` | `0.5` | LR reduction factor for plateau |
| `delta` | `float` | `1.0` | Threshold for Smooth L1 loss |
| `stoic_iter` | `int` | `10` | Warm-up epochs before early stop/scheduler |
| `epsilon` | `float` | `1e-15` | Small value for numerical stability |
| `adalr_window` | `int` | `5` | Loss window for adaptive learning rate |
| `w_init_scale` | `float` | `0.01` | Weight initialization scale |

## Model Attributes (post-fit)

| Attribute | Type | Description |
|-----------|------|-------------|
| `weights` | `np.ndarray` | Learned coefficients |
| `b` | `float` | Bias (intercept) |
| `loss_history` | `List[float]` | Loss per epoch |

## API Reference

### `IntenseRegressor.__init__(...)`
Creates the model with the hyper-parameters above.

### `fit(X_train, y_train)`
Trains via mini-batch gradient descent.

- **Raises** `ValueError` for NaN/Inf, shape mismatch, invalid params  
- **Raises** `OverflowError` if weights/bias become NaN/Inf
- **Raises** `RuntimeWarning` if there's a NaN value that clipped

### `predict(X_test)`
Returns $\hat{y}$ for new samples.

- **Raises** `ValueError` if model not fitted

### `score(X_test, y_test)`
Returns R² score.

## Usage Examples

### 1. MBGD + invscaling (default)
```python
import numpy as np
from sklearn.datasets import make_regression
from nexgml.gradient_supported import IntenseRegressor

X, y = make_regression(n_samples=500, n_features=20, noise=0.2, random_state=42)

model = IntenseRegressor(max_iter=1500, learning_rate=0.02,
                         penalty='l2', alpha=0.05, batch_size=32, verbose=1)
model.fit(X, y)

print(f"Loss: {model.loss_history[-1]:.6f}")
print(f"Weights (mean): {model.weights.mean():.6f}, bias: {model.b:.6f}")
```

### 2. Adam + plateau + scaling
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

model = IntenseRegressor(optimizer='adam', lr_scheduler='plateau', penalty='elasticnet', alpha=0.01, l1_ratio=0.7,
                         loss='smoothl1', delta=1.0, max_iter=3000, learning_rate=0.005, batch_size=64,
                         patience=10, factor=0.1, shuffle=True, random_state=123, verbose=2)
model.fit(X_sc, y)
```

### 3. AdamW + L2 + sparse data
```python
from scipy.sparse import csr_matrix

X_sp = csr_matrix(np.random.randn(1000, 500))
y_sp = np.random.randn(1000)

model = IntenseRegressor(optimizer='adamw', penalty='l2', alpha=0.001, max_iter=800, learning_rate=0.03,
                         batch_size=128, lr_scheduler='constant', stoic_iter=20)
model.fit(X_sp, y_sp)
```

## Best Practices

1. **Scale features** (`StandardScaler`) → faster convergence.  
2. Start with `learning_rate ∈ [0.001, 0.1]`; monitor `loss_history`.  
3. Use `early_stopping=True` + `tol=1e-4` and tune `stoic_iter` for warm-up.  
4. For high-dimensional data, use `penalty='l1'` or `'elasticnet'`.  
5. Choose `optimizer='adam'` or `'adamw'` for better convergence on complex data; tune `batch_size` based on memory.  
6. For robust regression, use `loss='smoothl1'` and tune `delta`.  
7. Plot loss curve:

```python
import matplotlib.pyplot as plt
plt.plot(model.loss_history)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
```

## Error Handling

- **NaN/Inf** in `X` or `y` → `ValueError`  
- **Shape mismatch** → `ValueError`  
- **Numerical overflow** during training → `OverflowError` (stops early)  
- **Invalid params** (e.g., AdamW with non-L2) → `ValueError`

## Performance Notes

| Aspect | GSIR |
|--------|------|
| **Speed** | Mini-batch GD – good for 10k+ samples |
| **Memory** | Sparse-friendly (`csr_matrix`, `csc_matrix`) |
| **Scalability** | Mini-batch + advanced optimizers for large datasets |

## Comparison with scikit-learn `SGDRegressor`

| Feature | GSIR | scikit-learn `SGDRegressor` |
|---------|------|-----------------------------------|
| **Regularization** | ✅ L1 / L2 / ElasticNet / None | ✅ L1 / L2 / ElasticNet / None |
| **Loss Functions** | ✅ MSE / RMSE / MAE / Smooth L1 | ✅ MSE / Huber / Epsilon Insensitive (partial overlap) |
| **Solver** | ✅ MBGD / Adam / AdamW | ✅ SGD (mini-batch) |
| **Sparse Input Support** | ✅ Full (`csr`, `csc`) | ✅ Full (`csr`, `csc`) |
| **Early Stopping** | ✅ Built-in (`tol`, `stoic_iter`) | ✅ Built-in (`tol`, `n_iter_no_change`) |
| **Data Shuffling** | ✅ Per-epoch + `random_state` | ✅ Built-in (per-iteration) |
| **Verbose Levels** | ✅ 0/1/2 (progress + LR updates) | ✅ Basic (via `verbose`) |
| **Learning Rate Control** | ✅ Initial + schedulers (constant/invscaling/plateau) | ✅ Initial + schedulers (constant/invscaling/adaptive/optimal) |
| **Loss History** | ✅ `loss_history` attribute | ❌ Not exposed |
| **Customizability** | ✅ Full access to mini-batch/optimizer loop | ❌ Limited (black-box solver) |
| **Mini-Batch Support** | ✅ Built-in (`batch_size`) | ✅ Built-in (`partial_fit` for online) |

> **Note**: `SGDRegressor` is faster and more robust for very large datasets due to optimized SGD. GSIR excels in **flexibility**, **advanced optimizers like Adam/AdamW**, **teaching**, and **custom gradient logic**.

## License

Part of the **NexGML** package – MIT (or as specified in the repository).
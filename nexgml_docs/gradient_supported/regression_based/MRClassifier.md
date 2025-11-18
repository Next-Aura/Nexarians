# MRC – Mini-batch Regression Classifier

## Overview

MRC (Mini-batch Regression Classifier) is an advanced linear classification model implemented in Python. It uses **mini-batch gradient descent** with **softmax** for multi-class classification, minimizing regression-style loss functions (**MSE**, **RMSE**, **MAE**, **Smooth L1**) between one-hot labels and predicted probabilities. It includes regularization options (**L1**, **L2**, **ElasticNet**) and learning rate schedulers (**constant**, **invscaling**, **plateau**). It works with both dense and sparse matrices, offers early stopping, data shuffling, and multi-level verbose logging.

Perfect for teaching, quick prototyping, or when you need a flexible classifier using regression losses with mini-batch efficiency without heavy dependencies.

## Installation & Requirements

```bash
pip install numpy scipy
```

```python
from nexgml.gradient_supported import MRClassifier
```

## Mathematical Formulation

### Prediction Function
$$
z = Xw + b
$$
$$
\hat{p} = \text{softmax}(z)
$$

- $X$ – feature matrix  
- $w$ – weight matrix (features × classes)  
- $b$ – bias vector (if `fit_intercept=True`)  
- $\hat{p}$ – predicted class probabilities

### Loss Functions

Regression losses applied to one-hot labels $y$ and probabilities $\hat{p}$:

- **MSE**: $\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C} (y_{i,c} - \hat{p}_{i,c})^2$
- **RMSE**: $\sqrt{\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C} (y_{i,c} - \hat{p}_{i,c})^2}$
- **MAE**: $\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C} |y_{i,c} - \hat{p}_{i,c}|$
- **Smooth L1**: $\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C} \begin{cases} 
  0.5 (y_{i,c} - \hat{p}_{i,c})^2 & |y_{i,c} - \hat{p}_{i,c}| \leq \delta \\ 
  \delta |y_{i,c} - \hat{p}_{i,c}| - 0.5 \delta^2 & \text{otherwise}
\end{cases}$

### Regularization
Added to the loss to control model complexity:

- **L1 (Lasso)**: $\alpha \sum |w_{i,j}|$
- **L2 (Ridge)**: $\alpha \sum w_{i,j}^2$
- **ElasticNet**: $\alpha \bigl[ l1\_ratio \sum |w_{i,j}| + (1-l1\_ratio)\sum w_{i,j}^2 \bigr]$

Total loss = *base loss* + *regularization term*.

### Gradients (example for MSE)
$$
\frac{\partial L}{\partial w} = \frac{2}{N}X^{T}(\hat{p} - y) + \text{regularization gradient}
$$
$$
\frac{\partial L}{\partial b}= \frac{2}{N}\sum (\hat{p} - y)
$$

MAE uses **sign**; RMSE normalizes by current RMSE; Smooth L1 conditional.

## Key Features
- **Regularization**: L1 / L2 / ElasticNet / None  
- **Losses**: MSE / RMSE / MAE / Smooth L1  
- **Input**: dense `np.ndarray` **or** SciPy sparse matrices  
- **Optimization**: mini-batch gradient descent  
- **LR Schedulers**: constant / invscaling / plateau  
- **Early stopping** on loss convergence (`tol`)  
- **Shuffling** + `random_state` for reproducibility  
- **Verbose levels** (0/1/2)  
- **Multi-class** support via softmax  
- **Mini-batch** processing for efficiency  

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_iter` | `int` | `1000` | Max training epochs |
| `learning_rate` | `float` | `0.01` | Initial step size |
| `penalty` | `Literal['l1','l2','elasticnet'] \| None` | `'l2'` | Regularization type |
| `alpha` | `float` | `0.0001` | Regularization strength |
| `l1_ratio` | `float` | `0.5` | ElasticNet mix (0 = L2, 1 = L1) |
| `loss` | `Literal['mse','rmse','mae','smoothl1']` | `'mse'` | Loss function |
| `fit_intercept` | `bool` | `True` | Add bias term |
| `tol` | `float` | `0.0001` | Early-stop tolerance |
| `shuffle` | `bool` | `True` | Shuffle data each epoch |
| `random_state` | `int \| None` | `None` | Seed for shuffling |
| `early_stopping` | `bool` | `True` | Enable early stop |
| `verbose` | `int` | `0` | 0 = silent, 1 = ~5 % progress, 2 = every epoch |
| `lr_scheduler` | `Literal['constant','invscaling','plateau']` | `'invscaling'` | Learning rate scheduler |
| `batch_size` | `int` | `16` | Mini-batch size |
| `power_t` | `float` | `0.25` | Exponent for invscaling |
| `patience` | `int` | `5` | Epochs to wait for plateau |
| `factor` | `float` | `0.5` | LR reduction factor for plateau |
| `delta` | `float` | `1.0` | Threshold for Smooth L1 loss |
| `stoic_iter` | `int` | `10` | Warm-up epochs before early stop/scheduler |

## Model Attributes (post-fit)

| Attribute | Type | Description |
|-----------|------|-------------|
| `weights` | `np.ndarray` | Learned coefficients (features × classes) |
| `b` | `np.ndarray` | Bias vector |
| `loss_history` | `List[float]` | Loss per epoch |
| `classes` | `np.ndarray` | Unique class labels |
| `n_classes` | `int` | Number of unique classes |

## API Reference

### `MRClassifier.__init__(...)`
Creates the model with the hyper-parameters above.

### `fit(X_train, y_train)`
Trains via mini-batch gradient descent.

- **Raises** `ValueError` for NaN/Inf, shape mismatch, invalid params, or <2 classes  
- **Raises** `OverflowError` if weights/bias become NaN/Inf

### `predict_proba(X_test)`
Returns predicted class probabilities $\hat{p}$ for new samples.

- **Raises** `ValueError` if model not fitted

### `predict(X_test)`
Returns predicted class labels (argmax of probabilities).

- **Raises** `ValueError` if model not fitted

### `score(X_test, y_test)`
Returns mean accuracy.

### `get_params(deep)`
Returns model paramters

### `set_params(**params)`
Returns model's attribute that ready to set

## Usage Examples

### 1. MSE + invscaling (default)
```python
import numpy as np
from sklearn.datasets import make_classification
from nexgml.gradient_supported import MRClassifier

X, y = make_classification(n_samples=500, n_features=20, n_classes=4, n_informative=10, random_state=42)

model = MRClassifier(max_iter=1500, learning_rate=0.02,
                     penalty='l2', alpha=0.05, loss='mse', batch_size=32, verbose=1)
model.fit(X, y)

print(f"Loss: {model.loss_history[-1]:.6f}")
print(f"Weights (mean): {model.weights.mean():.6f}, bias (mean): {np.mean(model.b):.6f}")
```

### 2. MAE + plateau + scaling
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

model = MRClassifier(loss='mae', penalty='elasticnet', alpha=0.01, l1_ratio=0.7,
                     lr_scheduler='plateau', max_iter=3000, learning_rate=0.005, batch_size=64,
                     patience=10, factor=0.1, shuffle=True, random_state=123, verbose=2)
model.fit(X_sc, y)
```

### 3. Smooth L1 + sparse data (no regularization)
```python
from scipy.sparse import csr_matrix

X_sp = csr_matrix(np.random.randn(1000, 500))
y_sp = np.random.randint(0, 5, 1000)  # 5 classes

model = MRClassifier(penalty=None, loss='smoothl1', delta=0.5, max_iter=800, learning_rate=0.03,
                     batch_size=128, lr_scheduler='constant', stoic_iter=20)
model.fit(X_sp, y_sp)
```

## Best Practices

1. **Scale features** (`StandardScaler`) → faster convergence.  
2. Start with `learning_rate ∈ [0.001, 0.1]`; monitor `loss_history`.  
3. Use `early_stopping=True` + `tol=1e-4` and tune `stoic_iter` for warm-up.  
4. For high-dimensional data, use `penalty='l1'` or `'elasticnet'`.  
5. Tune `batch_size` based on memory: smaller for noisy updates, larger for stability.  
6. Choose loss based on task: 'mse' for standard, 'mae'/'smoothl1' for robustness.  
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
- **Fewer than 2 classes** → `ValueError`  
- **Invalid params** → `ValueError`

## Performance Notes

| Aspect | MRC |
|--------|------|
| **Speed** | Mini-batch GD – good for 10k+ samples |
| **Memory** | Sparse-friendly (`csr_matrix`, `csc_matrix`) |
| **Scalability** | Mini-batch for larger datasets |

## Comparison with scikit-learn `SGDClassifier`

| Feature | MRC | scikit-learn `SGDClassifier` |
|---------|------|-----------------------------------|
| **Regularization** | ✅ L1 / L2 / ElasticNet / None | ✅ L1 / L2 / ElasticNet / None |
| **Loss Functions** | ✅ MSE / RMSE / MAE / Smooth L1 | ✅ Log / Hinge / Squared Hinge / etc. (includes squared_error, huber) |
| **Solver** | ✅ Mini-batch GD | ✅ SGD (mini-batch=1) |
| **Sparse Input Support** | ✅ Full (`csr`, `csc`) | ✅ Full (`csr`, `csc`) |
| **Early Stopping** | ✅ Built-in (`tol`, `stoic_iter`) | ✅ Built-in (`tol`, `n_iter_no_change`) |
| **Data Shuffling** | ✅ Per-epoch + `random_state` | ✅ Built-in (per-iteration) |
| **Verbose Levels** | ✅ 0/1/2 (progress + LR) | ✅ Basic (via `verbose`) |
| **Learning Rate Control** | ✅ Initial + schedulers (constant/invscaling/plateau) | ✅ Initial + schedulers (constant/invscaling/adaptive/optimal) |
| **Loss History** | ✅ `loss_history` attribute | ❌ Not exposed |
| **Customizability** | ✅ Full mini-batch loop access | ❌ Limited (black-box solver) |
| **Mini-Batch Support** | ✅ Built-in (`batch_size`) | ✅ Partial (via `partial_fit`) |

> **Note**: `SGDClassifier` is faster for very large datasets with stochastic updates. MRC excels in **mini-batch optimization**, **regression losses for classification**, **teaching**, and **custom logic**.

## License

Part of the **NexGML** package – MIT (or as specified in the repository).
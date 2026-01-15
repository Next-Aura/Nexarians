# GSBC – Gradient Supported Basic Classifier

## Overview

GSBC (Gradient Supported Basic Classifier) is a lightweight, custom linear classification model implemented in Python. It supports optimization via **gradient descent** with **softmax** for multi-class classification, includes regularization options such as **L1 (Lasso)**, **L2 (Ridge)**, and **ElasticNet**, and also learning rate schedulers (**constant**, **invscaling**, **plateau**) to prevent overfitting. The model minimizes **categorical cross-entropy** loss. It works with both dense and sparse matrices, offers early stopping, data shuffling, and multi-level verbose logging.

Perfect for teaching, quick prototyping, or when you need a simple, interpretable classifier without heavy dependencies.

## Installation & Requirements

```bash
pip install numpy scipy
```

```python
from nexgml.gradient_supported import BasicClassifier
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

- **Categorical Cross-Entropy**: $-\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C} y_{i,c} \log(\hat{p}_{i,c})$  
(where $y$ is one-hot encoded, $C$ is number of classes)

### Regularization
Added to the loss to control model complexity:

- **L1 (Lasso)**: $\alpha \sum |w_{i,j}|$
- **L2 (Ridge)**: $\alpha \sum w_{i,j}^2$
- **ElasticNet**: $\alpha \bigl[ l1\_ratio \sum |w_{i,j}| + (1-l1\_ratio)\sum w_{i,j}^2 \bigr]$

Total loss = *base loss* + *regularization term*.

### Gradients (example for Cross-Entropy)
$$
\frac{\partial L}{\partial w} = \frac{1}{N}X^{T}(\hat{p} - y) + \text{regularization gradient}
$$
$$
\frac{\partial L}{\partial b}= \frac{1}{N}\sum (\hat{p} - y)
$$

Supports class weighting for imbalanced data.

## Key Features
- **Regularization**: L1 / L2 / ElasticNet / None  
- **Loss**: Categorical Cross-Entropy  
- **Input**: dense `np.ndarray` **or** SciPy sparse matrices  
- **Optimization**: gradient descent with learning-rate control
- **LR Schedulers**: constant / invscaling / plateau 
- **Early stopping** on loss convergence (`tol`)  
- **Shuffling** + `random_state` for reproducibility  
- **Verbose levels** (0/1/2)  
- **Multi-class** support via softmax  
- **Class weighting** for imbalanced datasets  

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_iter` | `int` | `1000` | Max gradient-descent steps |
| `learning_rate` | `float` | `0.05` | Step size |
| `penalty` | `Literal['l1','l2','elasticnet'] \| None` | `'l2'` | Regularization type |
| `alpha` | `float` | `0.0001` | Regularization strength |
| `l1_ratio` | `float` | `0.5` | ElasticNet mix (0 = L2, 1 = L1) |
| `fit_intercept` | `bool` | `True` | Add bias term |
| `tol` | `float` | `0.0001` | Early-stop tolerance |
| `shuffle` | `bool` | `True` | Shuffle data each epoch |
| `random_state` | `int \|None` | `None` | Seed for shuffling |
| `early_stopping` | `bool` | `True` | Enable early stop |
| `verbose` | `int` | `0` | 0 = silent, 1 = ~5 % progress, 2 = every epoch |
| `verbosity` | `Literal['light', 'heavy']` | `'light'` | light = standard log information, heavy = more detail log information |
| `lr_scheduler` | `Literal['constant','invscaling','plateau']` | `'invscaling'` | Type of learning rate scheduler |
| `power_t` | `float` | `0.25` | Exponent for invscaling |
| `patience` | `int` | `5` | Epochs to wait for plateau |
| `factor` | `float` | `0.5` | LR reduction factor for plateau |
| `stoic_iter` | `int` | `10` | Warm-up epochs before early stop/scheduler |
| `weightning` | `bool` | `True` | If True, apply class weighting to handle imbalanced classes |
| `epsilon` | `float` | `1e-15` | Small value for numerical stability |
| `adalr_window` | `int` | `5` | Loss window for adaptive learning rate |
| `w_init_scale` | `float` | `0.01` | Weight initialization scale |

## Model Attributes (post-fit)

| Attribute | Type | Description |
|-----------|------|-------------|
| `weights` | `np.ndarray` | Learned coefficients (features × classes) |
| `b` | `np.ndarray` | Bias vector |
| `loss_history` | `List[float]` | Loss per iteration |
| `classes` | `np.ndarray` | Unique class labels |
| `n_classes` | `int` | Number of unique classes |

## API Reference

### `BasicClassifier.__init__(...)`
Creates the model with the hyper-parameters above.

### `fit(X_train, y_train)`
Trains via gradient descent.

- **Raises** `ValueError` for NaN/Inf or shape mismatch  
- **Raises** `OverflowError` if weights/bias become NaN/Inf
- **Raises** `RuntimeWarning` if there's a NaN value that clipped

### `predict_proba(X_test)`
Returns predicted class probabilities $\hat{p}$ for new samples.

- **Raises** `ValueError` if model not fitted

### `predict(X_test)`
Returns predicted class labels (argmax of probabilities).

- **Raises** `ValueError` if model not fitted

### `score(X_test, y_test)`
Returns mean accuracy.

## Usage Examples

### 1. L2 (default)
```python
import numpy as np
from sklearn.datasets import make_classification
from nexgml.gradient_supported import BasicClassifier

X, y = make_classification(n_samples=200, n_features=10, n_classes=3, n_informative=5, random_state=42)

model = BasicClassifier(max_iter=1500, learning_rate=0.02,
                        penalty='l2', alpha=0.05, verbose=1)
model.fit(X, y)

print(f"Loss: {model.loss_history[-1]:.6f}")
print(f"Weights (mean): {model.weights.mean():.6f}, bias (mean): {np.mean(model.b):.6f}")
```

### 2. ElasticNet + scaling
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

model = BasicClassifier(penalty='elasticnet', alpha=0.01, l1_ratio=0.7,
                        max_iter=3000, learning_rate=0.005,
                        shuffle=True, random_state=123, verbose=2)
model.fit(X_sc, y)
```

### 3. Sparse data (no regularisation)
```python
from scipy.sparse import csr_matrix

X_sp = csr_matrix(np.random.randn(500, 200))
y_sp = np.random.randint(0, 3, 500)  # 3 classes

model = BasicClassifier(penalty=None, max_iter=800, learning_rate=0.03)
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
- **Fewer than 2 classes** → `ValueError`

## Performance Notes

| Aspect | GSBC |
|--------|------|
| **Speed** | Batch GD – good for ≤ 10 k samples |
| **Memory** | Sparse-friendly (`csr_matrix`, `csc_matrix`) |
| **Scalability** | Extend to mini-batch for > 100 k rows |

## Comparison with scikit-learn `LogisticRegression`

| Feature | GSBC | scikit-learn `LogisticRegression` |
|---------|------|-----------------------------------|
| **Regularization** | ✅ L1 / L2 / ElasticNet / None | ✅ L1 / L2 / ElasticNet / None |
| **Loss Function** | ✅ Cross-Entropy | ✅ Cross-Entropy |
| **Solver** | ✅ Gradient Descent | ✅ LBFGS, SAG, SAGA, Newton-CG, liblinear |
| **Sparse Input Support** | ✅ Full (`csr`, `csc`) | ✅ Full (`csr`, `csc`) |
| **Early Stopping** | ✅ Built-in (`tol`) | ❌ Only with `SAG/SAGA` + `warm_start` |
| **Data Shuffling** | ✅ Per-epoch + `random_state` | ❌ Only in `SAG/SAGA` |
| **Verbose Levels** | ✅ 0/1/2 (progress every epoch or ~5%) | ✅ Basic (via `verbose`) |
| **Class Weighting** | ✅ Automatic balancing | ✅ Manual via `class_weight` |
| **Learning Rate Control** | ✅ Manual `learning_rate` | ❌ Only in `SAG/SAGA` (adaptive) |
| **Loss History** | ✅ `loss_history` attribute | ❌ Not exposed |
| **Customizability** | ✅ Full access to GD loop | ❌ Limited (black-box solvers) |

> **Note**: `LogisticRegression` is faster and more robust for large datasets due to optimized solvers. GSBC excels in **transparency**, **teaching**, and **custom gradient logic**.

## License

Part of the **NexGML** package – MIT (or as specified in the repository).
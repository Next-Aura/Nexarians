# FBR – Forest Backend Regressor

## Overview

FBR (Forest Backend Regressor) is a lightweight, custom random forest model for regression tasks implemented in Python. It builds an ensemble of decision trees by randomly sampling data (bootstrapping) and features, then aggregates predictions via averaging to improve accuracy and reduce overfitting. It supports various tree hyperparameters and works with both dense and sparse matrices.

Perfect for teaching, quick prototyping, or when you need a robust, interpretable ensemble regressor without heavy dependencies.

## Installation & Requirements

```bash
pip install numpy scipy
```

```python
from nexgml.tree_models import ForestBackendRegressor
```

## Mathematical Formulation

### Prediction Function

Each tree predicts a value by traversing to a leaf. The forest aggregates via averaging:

- For a sample $x$, get predictions $\hat{y}_1, \hat{y}_2, ..., \hat{y}_T$ from $T$ trees.
- Final $\hat{y} = \frac{1}{T} \sum_{t=1}^T \hat{y}_t$

### Impurity Functions (Criteria)

Inherited from trees: **Squared Error**, **Friedman MSE**, **Absolute Error**, or **Poisson**.

### Ensemble Mechanism

- **Bagging**: Bootstrap sampling for each tree.
- **Feature Randomness**: Subset of features per tree/split.
- Splits minimize weighted child impurity, as in single trees.

## Key Features
- **Ensemble Size**: n_estimators trees  
- **Sampling**: bootstrap / max_samples / max_features  
- **Tree Params**: max_depth / min_samples_leaf / etc.  
- **Criteria**: Squared Error / Friedman MSE / Absolute Error / Poisson  
- **Input**: dense `np.ndarray` **or** SciPy sparse matrices  
- **Double Sampling**: Optional extra subsampling per tree  
- **Verbose**: Progress logging  

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_estimators` | `int` | `10` | Number of trees in forest |
| `bootstrap` | `bool` | `True` | Use bootstrap sampling for trees |
| `double_sampling` | `bool` | `False` | Apply extra sampling inside each tree |
| `max_depth` | `int \| None` | `6` | Max depth per tree |
| `min_samples_leaf` | `int \| None` | `1` | Min samples per leaf |
| `criterion` | `Literal['squared_error','friedman_mse','absolute_error','poisson']` | `'squared_error'` | Impurity measure |
| `max_features` | `Optional[Literal['sqrt','log2']] \| int \| float \| None` | `'sqrt'` | Features per tree/split |
| `max_samples` | `Optional[Literal['sqrt','log2']] \| int \| float \| None` | `'sqrt'` | Samples per tree (if bootstrap) |
| `random_state` | `int \| None` | `None` | Seed for randomness |
| `min_samples_split` | `int \| None` | `2` | Min samples to split node |
| `min_impurity_decrease` | `float \| None` | `0.0` | Min impurity decrease for split |
| `verbose` | `int` | `0` | 0 = silent, 1 = progress |

## Model Attributes (post-fit)

| Attribute | Type | Description |
|-----------|------|-------------|
| `trees` | `List[TreeBackendRegressor]` | List of fitted trees |
| `feature_indices` | `List[np.ndarray]` | Features used per tree |

## API Reference

### `ForestBackendRegressor.__init__(...)`
Creates the model with the hyper-parameters above.

### `fit(X_train, y_train)`
Trains the forest by fitting each tree.

- **Raises** `ValueError` for invalid params

### `predict(X_test)`
Returns predicted values (average of tree predictions).

- **Raises** `ValueError` if model not fitted

## Usage Examples

### 1. Default (Squared Error + bootstrap)
```python
import numpy as np
from sklearn.datasets import make_regression
from nexgml.tree_models import ForestBackendRegressor

X, y = make_regression(n_samples=200, n_features=10, noise=0.2, random_state=42)

model = ForestBackendRegressor(n_estimators=20, max_depth=5, min_samples_leaf=3,
                               criterion='squared_error', max_features='sqrt', max_samples='sqrt',
                               random_state=42, verbose=1)
model.fit(X, y)

print(f"Predictions: {model.predict(X[:5])}")
```

### 2. Friedman MSE + no bootstrap
```python
model = ForestBackendRegressor(n_estimators=15, bootstrap=False, double_sampling=True,
                               criterion='friedman_mse', max_depth=7, min_samples_split=4,
                               max_samples=0.8, min_impurity_decrease=0.01, random_state=123)
model.fit(X, y)
```

### 3. Sparse data + Poisson
```python
from scipy.sparse import csr_matrix

X_sp = csr_matrix(np.random.randn(500, 200))
y_sp = np.random.poisson(lam=2, size=500)  # Poisson-distributed targets

model = ForestBackendRegressor(n_estimators=10, criterion='poisson', max_depth=4,
                               min_samples_leaf=10, verbose=1)
model.fit(X_sp, y_sp)
```

## Best Practices

1. **No scaling needed**, but encode categoricals.  
2. Increase `n_estimators` for better performance (trade-off: time).  
3. Use `bootstrap=True` and `max_features='sqrt'` for diversity.  
4. Tune tree params like `max_depth` to control complexity.  
5. Set `double_sampling=True` for extra randomness.  
6. Choose criterion based on data: 'poisson' for counts, 'absolute_error' for robustness.

## Error Handling

- **Invalid params** (e.g., n_estimators ≤0, invalid criterion) → `ValueError`  
- **Not fitted** → `ValueError`  

## Performance Notes

| Aspect | FBR |
|--------|------|
| **Speed** | Parallelizable trees – good for 10k+ samples |
| **Memory** | Sparse-friendly (`csr_matrix`, `csc_matrix`) |
| **Scalability** | Ensemble for larger data; OOB error possible extension |

## Comparison with scikit-learn `RandomForestRegressor`

| Feature | FBR | scikit-learn `RandomForestRegressor` |
|---------|------|-----------------------------------|
| **Ensemble Size** | ✅ n_estimators | ✅ n_estimators |
| **Bootstrap** | ✅ With max_samples | ✅ With max_samples |
| **Criteria** | ✅ Squared Error / Friedman MSE / Absolute Error / Poisson | ✅ Squared Error / Absolute Error / Poisson (no Friedman MSE directly) |
| **Feature Subsampling** | ✅ max_features | ✅ max_features |
| **Tree Pruning** | ✅ max_depth / min_samples_leaf / etc. | ✅ Same + more |
| **Sparse Input Support** | ✅ Full (`csr`, `csc`) | ✅ Full (`csr`, `csc`) |
| **Double Sampling** | ✅ Custom option | ❌ Not built-in |
| **Verbose** | ✅ Basic progress | ✅ Basic (n_jobs aware) |
| **Customizability** | ✅ Access to individual trees/indices | ✅ Access to estimators |
| **Parallel Training** | ❌ Sequential | ✅ Multi-threaded (n_jobs) |

> **Note**: `RandomForestRegressor` is faster with parallelization and more features. FBR excels in **transparency**, **teaching**, and **simple ensemble logic**.

## License

Part of the **NexGML** package – MIT (or as specified in the repository).
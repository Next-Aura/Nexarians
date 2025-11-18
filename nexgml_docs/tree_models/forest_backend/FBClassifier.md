# FBC – Forest Backend Classifier

## Overview

FBC (Forest Backend Classifier) is a lightweight, custom random forest model for classification tasks implemented in Python. It builds an ensemble of decision trees by randomly sampling data (bootstrapping) and features, then aggregates predictions via majority vote to improve accuracy and reduce overfitting. It supports various tree hyperparameters and works with both dense and sparse matrices.

Perfect for teaching, quick prototyping, or when you need a robust, interpretable ensemble classifier without heavy dependencies.

## Installation & Requirements

```bash
pip install numpy scipy
```

```python
from nexgml.tree_models import ForestBackendClassifier
```

## Mathematical Formulation

### Prediction Function

Each tree predicts a class by traversing to a leaf. The forest aggregates via majority vote:

- For a sample $x$, get predictions $\hat{y}_1, \hat{y}_2, ..., \hat{y}_T$ from $T$ trees.
- Final $\hat{y} = \arg\max_k \sum_{t=1}^T \mathbb{1}(\hat{y}_t = k)$

Probabilities: proportion of trees voting for each class.

### Impurity Functions (Criteria)

Inherited from trees: **Gini**, **Entropy**, or **Log Loss**.

### Ensemble Mechanism

- **Bagging**: Bootstrap sampling (with replacement) for each tree.
- **Feature Randomness**: Subset of features per tree/split.
- Splits minimize weighted child impurity, as in single trees.

## Key Features
- **Ensemble Size**: n_estimators trees  
- **Sampling**: bootstrap / max_samples / max_features  
- **Tree Params**: max_depth / min_samples_leaf / etc.  
- **Criteria**: Gini / Entropy / Log Loss  
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
| `min_samples_leaf` | `int \| None` | `5` | Min samples per leaf |
| `criterion` | `Literal['gini','entropy','log_loss']` | `'gini'` | Impurity measure |
| `max_features` | `Optional[Literal['sqrt','log2']] \| int \| float \| None` | `'sqrt'` | Features per tree/split |
| `max_samples` | `Optional[Literal['sqrt','log2']] \| int \| float \| None` | `'sqrt'` | Samples per tree (if bootstrap) |
| `random_state` | `int \| None` | `None` | Seed for randomness |
| `min_samples_split` | `int \| None` | `2` | Min samples to split node |
| `min_impurity_decrease` | `float \| None` | `0.0` | Min impurity decrease for split |
| `verbose` | `int` | `0` | 0 = silent, 1 = progress |

## Model Attributes (post-fit)

| Attribute | Type | Description |
|-----------|------|-------------|
| `trees` | `List[TreeBackendClassifier]` | List of fitted trees |
| `feature_indices` | `List[np.ndarray]` | Features used per tree |

## API Reference

### `ForestBackendClassifier.__init__(...)`
Creates the model with the hyper-parameters above.

### `fit(X_train, y_train)`
Trains the forest by fitting each tree.

- **Raises** `ValueError` for invalid params

### `predict(X_test)`
Returns predicted class labels (majority vote).

- **Raises** `ValueError` if model not fitted

### `predict_proba(X_test)`
Returns class probabilities (vote proportions).

- **Raises** `ValueError` if model not fitted

### `score(X_test, y_test)`
Returns mean accuracy.

### `get_params(deep)`
Returns model paramters

### `set_params(**params)`
Returns model's attribute that ready to set

## Usage Examples

### 1. Default (Gini + bootstrap)
```python
import numpy as np
from sklearn.datasets import make_classification
from nexgml.tree_models import ForestBackendClassifier

X, y = make_classification(n_samples=200, n_features=10, n_classes=3, n_informative=5, random_state=42)

model = ForestBackendClassifier(n_estimators=20, max_depth=5, min_samples_leaf=3,
                                criterion='gini', max_features='sqrt', max_samples='sqrt',
                                random_state=42, verbose=1)
model.fit(X, y)

print(f"Predictions: {model.predict(X[:5])}")
```

### 2. Entropy + no bootstrap
```python
model = ForestBackendClassifier(n_estimators=15, bootstrap=False, double_sampling=True,
                                criterion='entropy', max_depth=7, min_samples_split=4,
                                max_samples=0.8, min_impurity_decrease=0.01, random_state=123)
model.fit(X, y)
```

### 3. Sparse data + log_loss
```python
from scipy.sparse import csr_matrix

X_sp = csr_matrix(np.random.randn(500, 200))
y_sp = np.random.randint(0, 3, 500)  # 3 classes

model = ForestBackendClassifier(n_estimators=10, criterion='log_loss', max_depth=4,
                                min_samples_leaf=10, verbose=1)
model.fit(X_sp, y_sp)
```

## Best Practices

1. **No scaling needed**, but encode categoricals.  
2. Increase `n_estimators` for better performance (trade-off: time).  
3. Use `bootstrap=True` and `max_features='sqrt'` for diversity.  
4. Tune tree params like `max_depth` to control complexity.  
5. Set `double_sampling=True` for extra randomness.  
6. Use `predict_proba` for confidence scores.  

## Error Handling

- **Invalid params** (e.g., n_estimators ≤0, invalid criterion) → `ValueError`  
- **Not fitted** → `ValueError`  

## Performance Notes

| Aspect | FBC |
|--------|------|
| **Speed** | Parallelizable trees – good for 10k+ samples |
| **Memory** | Sparse-friendly (`csr_matrix`, `csc_matrix`) |
| **Scalability** | Ensemble for larger data; OOB error possible extension |

## Comparison with scikit-learn `RandomForestClassifier`

| Feature | FBC | scikit-learn `RandomForestClassifier` |
|---------|------|-----------------------------------|
| **Ensemble Size** | ✅ n_estimators | ✅ n_estimators |
| **Bootstrap** | ✅ With max_samples | ✅ With max_samples |
| **Criteria** | ✅ Gini / Entropy / Log Loss | ✅ Gini / Entropy / Log Loss |
| **Feature Subsampling** | ✅ max_features | ✅ max_features |
| **Tree Pruning** | ✅ max_depth / min_samples_leaf / etc. | ✅ Same + more |
| **Sparse Input Support** | ✅ Full (`csr`, `csc`) | ✅ Full (`csr`, `csc`) |
| **Double Sampling** | ✅ Custom option | ❌ Not built-in |
| **Verbose** | ✅ Basic progress | ✅ Basic (n_jobs aware) |
| **Predict Proba** | ✅ Vote proportions | ✅ Vote proportions |
| **Customizability** | ✅ Access to individual trees/indices | ✅ Access to estimators |
| **Parallel Training** | ❌ Sequential | ✅ Multi-threaded (n_jobs) |

> **Note**: `RandomForestClassifier` is faster with parallelization and more features. FBC excels in **transparency**, **teaching**, and **simple ensemble logic**.

## License

Part of the **NexGML** package – MIT (or as specified in the repository).
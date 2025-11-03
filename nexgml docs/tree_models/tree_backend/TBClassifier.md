# TBC – Tree Backend Classifier

## Overview

TBC (Tree Backend Classifier) is a lightweight, custom decision tree model for classification tasks implemented in Python. It builds a tree by recursively splitting nodes to minimize a given impurity criterion (**Gini**, **Entropy**, or **Log Loss**) and supports various pruning and stopping parameters to prevent overfitting. It works with both dense and sparse matrices and includes options for feature and sample subsampling for randomness.

Perfect for teaching, quick prototyping, or when you need a simple, interpretable classifier without heavy dependencies.

## Installation & Requirements

```bash
pip install numpy scipy
```

```python
from nexgml.tree_models import TreeBackendClassifier
```

## Mathematical Formulation

### Prediction Function

Predictions are made by traversing the tree from the root to a leaf based on feature thresholds:

- Start at root node.
- At each internal node, compare sample feature value to threshold: go left if ≤ threshold, right otherwise.
- Reach leaf node and return majority class label.

### Impurity Functions (Criteria)

- **Gini**: $1 - \sum_{k=1}^{K} p_k^2$ (where $p_k$ is proportion of class $k$)
- **Entropy**: $-\sum_{k=1}^{K} p_k \log_2(p_k)$
- **Log Loss**: $-\sum_{k=1}^{K} p_k \log(p_k)$ (similar to entropy but natural log)

### Splitting Rule

At each node, select feature and threshold that minimize weighted impurity of child nodes:

$$
I = \frac{N_l}{N} I_l + \frac{N_r}{N} I_r
$$

(where $N_l$, $N_r$ are left/right child sizes, $I_l$, $I_r$ are their impurities)

Splits only if impurity decrease ≥ `min_impurity_decrease` and meets other constraints.

## Key Features
- **Criteria**: Gini / Entropy / Log Loss  
- **Pruning/Stopping**: max_depth / min_samples_leaf / min_samples_split / min_impurity_decrease  
- **Input**: dense `np.ndarray` **or** SciPy sparse matrices  
- **Randomness**: max_features / max_samples / random_state for subsampling  
- **Interpretability**: Tree structure as nested dict  

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_depth` | `int \| None` | `6` | Maximum tree depth |
| `min_samples_leaf` | `int \| None` | `5` | Min samples per leaf node |
| `criterion` | `Literal['gini','entropy','log_loss']` | `'gini'` | Impurity measure for splits |
| `max_features` | `Optional[Literal['sqrt','log2']] \| int \| float \| None` | `None` | Features to consider per split |
| `max_samples` | `Optional[Literal['sqrt','log2']] \| int \| float \| None` | `None` | Samples to draw for training |
| `random_state` | `int \| None` | `None` | Seed for randomness |
| `min_samples_split` | `int \| None` | `2` | Min samples to split a node |
| `min_impurity_decrease` | `float \| None` | `0.0` | Min impurity decrease for split |

## Model Attributes (post-fit)

| Attribute | Type | Description |
|-----------|------|-------------|
| `tree` | `dict` | Nested dict representing tree structure |

## API Reference

### `TreeBackendClassifier.__init__(...)`
Creates the model with the hyper-parameters above.

### `fit(X_train, y_train)`
Builds the tree recursively.

- **Raises** `ValueError` for empty data, shape mismatch, or invalid params

### `predict(X_test)`
Returns predicted class labels by traversing the tree.

- **Raises** `ValueError` if model not fitted

### `score(X_test, y_test)`
Returns mean accuracy.

## Usage Examples

### 1. Gini (default)
```python
import numpy as np
from sklearn.datasets import make_classification
from nexgml.tree_models import TreeBackendClassifier

X, y = make_classification(n_samples=200, n_features=10, n_classes=3, n_informative=5, random_state=42)

model = TreeBackendClassifier(max_depth=5, min_samples_leaf=3,
                              criterion='gini', max_features='sqrt', random_state=42)
model.fit(X, y)

print(f"Accuracy: {model.score(X, y):.4f}")
```

### 2. Entropy + subsampling
```python
model = TreeBackendClassifier(criterion='entropy', max_depth=7, min_samples_split=4,
                              max_samples=0.8, min_impurity_decrease=0.01, random_state=123)
model.fit(X, y)
```

### 3. Sparse data + log_loss
```python
from scipy.sparse import csr_matrix

X_sp = csr_matrix(np.random.randn(500, 200))
y_sp = np.random.randint(0, 3, 500)  # 3 classes

model = TreeBackendClassifier(criterion='log_loss', max_depth=4, min_samples_leaf=10)
model.fit(X_sp, y_sp)
```

## Best Practices

1. **No scaling needed** for features, but encode categorical variables if present.  
2. Tune `max_depth` and `min_samples_leaf` to balance bias-variance; start with small values to avoid overfitting.  
3. Use `max_features='sqrt'` for randomness and faster training on high-dimensional data.  
4. Set `min_impurity_decrease > 0` for simpler trees.  
5. Inspect tree structure (`model.tree`) for interpretability.  
6. For larger datasets, use subsampling with `max_samples < 1.0`.  

## Error Handling

- **Empty/mismatch data** → `ValueError`  
- **Invalid params** (e.g., criterion, depths/leaves/splits ≤0, split < 2*leaf) → `ValueError`  

## Performance Notes

| Aspect | TBC |
|--------|------|
| **Speed** | Recursive splits – good for ≤ 10k samples |
| **Memory** | Sparse-friendly (`csr_matrix`, `csc_matrix`) |
| **Scalability** | Limited depth for large data; extend to ensembles for better performance |

## Comparison with scikit-learn `DecisionTreeClassifier`

| Feature | TBC | scikit-learn `DecisionTreeClassifier` |
|---------|------|-----------------------------------|
| **Criteria** | ✅ Gini / Entropy / Log Loss | ✅ Gini / Entropy / Log Loss |
| **Pruning/Stopping** | ✅ max_depth / min_samples_leaf / min_samples_split / min_impurity_decrease | ✅ Same + more (min_weight_fraction_leaf, etc.) |
| **Randomness** | ✅ max_features / max_samples / random_state | ✅ max_features / random_state (no max_samples) |
| **Sparse Input Support** | ✅ Full (`csr`, `csc`) | ✅ Full (`csr`, `csc`) |
| **Tree Structure Access** | ✅ Nested dict | ✅ Internal (via export_text/graphviz) |
| **Customizability** | ✅ Full recursive build access | ❌ Limited (black-box splitter) |

> **Note**: `DecisionTreeClassifier` is faster and more optimized for large datasets. TBC excels in **transparency**, **teaching**, and **simple implementation**.

## License

Part of the **NexGML** package – MIT (or as specified in the repository).
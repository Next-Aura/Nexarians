import numpy as np
from scipy.sparse import issparse
import pandas as pd
from typing import Literal, Optional

from .childs.KGBC_support import Support
from ..indexing import standard_indexing, one_hot_labeling
from ..amo import softmax

class KiloGBC:
    def __init__(self,
                 n_estimators=100,
                 learning_rate=0.1,
                 alpha=1e-4,
                 l1_ratio=0.5,
                 fit_intercept=True,
                 bootstrap=True,
                 max_samples: Optional[Literal['sqrt', 'log2']] | int | float | None='sqrt',
                 max_features: Optional[Literal['sqrt', 'log2']] | int | float | None='sqrt',
                 clip_value=10.0,
                 verbose=0,
                 random_state=None):
        
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.support_alpha = float(alpha)
        self.support_l1_ratio = float(l1_ratio)
        self.support_intercept = bool(fit_intercept)
        self.bootstrap = bool(bootstrap)
        self.max_samples = max_samples
        self.max_features = max_features
        self.clip_value = float(clip_value)
        self.verbose = int(verbose)
        self.random_state = None if random_state is None else int(random_state)

        self.classes_ = None
        self.n_classes_ = None
        self.supports_ = []
        self.initial_logits_ = None

    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy(dtype=np.float64)

        elif issparse(X):
            X = X.toarray().astype(np.float64)

        else:
            X = np.array(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array for X, got shape {X.shape}")
        
        y = np.asarray(y)
        n_samples, n_features = X.shape
        classes = np.unique(y)
        self.classes_ = classes
        self.n_classes_ = len(classes)
        y_one_hot = one_hot_labeling(y, classes)
        current_logits = np.zeros((n_samples, self.n_classes_), dtype=np.float64)
        self.initial_logits_ = current_logits.copy()
        self.supports_ = []

        for m in range(self.n_estimators):
            p_current = softmax(current_logits)
            residuals = y_one_hot - p_current

            idx = standard_indexing(n_samples, self.max_samples)
            if self.bootstrap:
                idx = np.random.choice(n_samples, idx, replace=True)
            
            else:
                idx = np.random.choice(n_samples, idx, replace=False)

            X_sub = X[idx]
            resid_sub = residuals[idx]

            feature_indices = standard_indexing(n_features, self.max_features)
            feature_indices = np.random.choice(n_features, feature_indices, replace=False)
            X_sub_feat = X_sub[:, feature_indices]

            support = Support(alpha=self.support_alpha,
                              l1_ratio=self.support_l1_ratio,
                              fit_intercept=self.support_intercept)
            
            support.fit(X_sub_feat, resid_sub)
            self.supports_.append(support)
            support.feature_indices_ = feature_indices
            pred_logits = support.predict(X[:, feature_indices])

            assert pred_logits.shape == (n_samples, self.n_classes_), f"pred_logits shape mismatch: {pred_logits.shape}"

            pred_logits = np.clip(pred_logits, -self.clip_value, self.clip_value)
            current_logits += (self.learning_rate * pred_logits)

            if self.verbose and ((m % max(1, self.n_estimators // 20)) == 0 or m < 5):
                print(f"|=Train model {m + 1}/{self.n_estimators}=|")

        self.initial_logits_ = np.zeros((1, self.n_classes_), dtype=np.float64)

        return self

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy(dtype=np.float64)
        elif issparse(X):
            X = X.toarray().astype(np.float64)
        else:
            X = np.array(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array for X, got shape {X.shape}")
        
        n_samples = X.shape[0]
        logits = np.zeros((n_samples, self.n_classes_), dtype=np.float64)

        for support in self.supports_:
            pred = support.predict(X[:, support.feature_indices_])
            pred = np.clip(pred, -self.clip_value, self.clip_value)
            logits += (self.learning_rate * pred)

        probs = softmax(logits)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        idxs = np.argmax(probs, axis=1)
        return self.classes_[idxs]
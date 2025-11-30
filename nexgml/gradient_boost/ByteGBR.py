import numpy as np
from scipy.sparse import issparse
import pandas as pd
from typing import Literal, Optional

from .childs.BGBR_support import Support
from ..indexing import standard_indexing

class ByteGBR:
    def __init__(self,
                 n_estimators=100,
                 learning_rate=0.1,
                 alpha=1e-4,
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
        self.support_intercept = bool(fit_intercept)
        self.bootstrap = bool(bootstrap)
        self.max_samples = max_samples
        self.max_features = max_features
        self.clip_value = float(clip_value)
        self.verbose = int(verbose)
        self.random_state = None if random_state is None else int(random_state)

        self.supports_ = []
        self.initial_pred_ = None

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

        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n_samples, n_features = X.shape
        current_pred = np.zeros((n_samples, 1), dtype=np.float64)
        self.initial_pred_ = current_pred
        self.supports_ = []

        for m in range(self.n_estimators):
            residuals = y - current_pred

            idx = standard_indexing(n_samples, self.max_samples)
            if self.bootstrap:
                idx = np.random.choice(n_samples, idx, replace=True)

            else:
                idx = np.random.choice(n_samples, idx, replace=False)

            feature_idx = standard_indexing(n_features, self.max_features)
            feature_idx = np.random.choice(n_features, feature_idx, replace=False)

            X_sub = X[idx]
            resid_sub = residuals[idx]
            X_sub_feat = X_sub[:, feature_idx]

            support = Support(alpha=self.support_alpha,
                                     fit_intercept=self.support_intercept)

            support.fit(X_sub_feat, resid_sub)
            self.supports_.append(support)
            support.feature_indices_ = feature_idx
            pred_update = support.predict(X[:, feature_idx])

            assert pred_update.shape == (n_samples, 1), f"pred_update shape mismatch: {pred_update.shape}"

            pred_update = np.clip(pred_update, -self.clip_value, self.clip_value)
            current_pred += (self.learning_rate * pred_update)

            if self.verbose and ((m % max(1, self.n_estimators // 20)) == 0 or m < 5):
                print(f"|=Trained model {m + 1}/{self.n_estimators}=|")

        self.initial_pred_ = np.mean(y, axis=0).reshape(1, -1)

        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy(dtype=np.float64)
        elif issparse(X):
            X = X.toarray().astype(np.float64)
        else:
            X = np.array(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array for X, got shape {X.shape}")

        n_samples = X.shape[0]
        pred = np.full((n_samples, 1), self.initial_pred_[0, 0], dtype=np.float64)

        for support in self.supports_:
            pred_update = support.predict(X[:, support.feature_indices_])
            pred_update = np.clip(pred_update, -self.clip_value, self.clip_value)
            pred += (self.learning_rate * pred_update)

        return pred.ravel()
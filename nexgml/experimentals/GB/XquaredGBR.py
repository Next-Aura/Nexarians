import numpy as np
from typing import Optional, Literal
from scipy.sparse import spmatrix, issparse
from math import sqrt, log2

from .childs.XGBR_support import Support
from nexgml.indexing import standard_indexing

class XquaredGBR:
    def __init__(self,
                n_estimators: int=100,
                max_iter: int=1000,
                learning_rate: float=0.1,
                verbose: int=0,
                penalty: Literal["l1", "l2", "elasticnet"] | None="l2",
                alpha: float=1e-4,
                l1_ratio: float=0.5,
                fit_intercept: bool=True,
                tol: float=0.0001,
                loss: Literal["mse", "rmse", "huber"] | None="mse",
                random_state: int | None=None,
                early_stopping: bool=True,
                bootstrap: bool | None=True,
                max_features: Optional[Literal['sqrt', 'log2']] | int | float | None='sqrt',
                max_samples: Optional[Literal['sqrt', 'log2']] | int | float | None='sqrt',
                huber_threshold: float | None=0.5,
                n_supports_child: int | None = 10
                ):
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.intercept = fit_intercept

        self.tol = tol
        self.loss = loss
        self.early_stop = early_stopping
        self.random_state = random_state
        self.bootstrap = bootstrap
        self.delta = huber_threshold

        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.supports_child = n_supports_child

        self.max_features = max_features
        self.max_samples = max_samples

        self.loss_history = []
        self.supports = []
        self.feature_indices = []
        self.model_data = []

    def fit(self, X: np.ndarray | spmatrix, y: np.ndarray):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        if not issparse(X):
            X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape

        
        max_features = standard_indexing(n_features, self.max_features)
        max_samples = standard_indexing(n_samples, self.max_samples)
        current_predictions = np.zeros(n_samples, dtype=float)

        for i in range(self.n_estimators):
            residuals = y - current_predictions

            
            if self.bootstrap:
                indices = np.random.choice(n_samples, size=max_samples, replace=True)
      
            else:
                indices = np.arange(max_samples)

            feature_idx = np.random.choice(n_features, size=max_features, replace=False)
            self.feature_indices.append(feature_idx)

            X_sub = X[indices]
            if issparse(X_sub):
                X_sub = X[:, feature_idx]
            else:
                X_sub = X_sub[:, feature_idx]
            y_sub = residuals[indices]

            support = Support(
                max_iter=self.max_iter,
                learning_rate=self.learning_rate,
                loss=self.loss,
                penalty=self.penalty,
                alpha=self.alpha,
                tol=self.tol,
                fit_intercept=self.intercept,
                l1_ratio=self.l1_ratio,
                huber_threshold=self.delta,
                n_estimators=self.supports_child
            )
            if self.verbose == 2:
                print(f"|=-Training {i + 1}th model-=|")

            model_data = support.fit(X_sub, y_sub)
            self.supports.append(support)
            self.model_data.append(model_data)
            self.loss_history.append(model_data['loss'])
            pred_sub = support.predict(X[:, feature_idx])
            current_predictions += self.learning_rate * pred_sub

            if self.verbose == 1:
                print(f"|=Trained {i+1}/{self.n_estimators}th model, Loss: {self.loss_history[-1]}=|")

            if self.early_stop and i > 1:
                if abs(self.loss_history[-1] - self.loss_history[-2]) < self.tol:
                  if self.verbose == 1:
                      print(f"|=Stopping supports training at estimator {i + 1}=|")
                  break

    def predict(self, X: np.ndarray | spmatrix) -> np.ndarray:
        if not issparse(X):
            X = np.asarray(X)

        n_samples = X.shape[0]
        final_predictions = np.zeros(n_samples, dtype=float)

        for i, support in enumerate(self.supports):
            feature_idx = self.feature_indices[i]
            if issparse(X):
                X_sub = X[:, feature_idx]

            else:
                X_sub = X[:, feature_idx]

            final_predictions += self.learning_rate * support.predict(X_sub)

        return final_predictions

    def get_params(self, deep=True):
        return {
            'n_estimators': self.n_estimators,
            'max_iter': self.max_iter,
            'learning_rate': self.learning_rate,
            'verbose': self.verbose,
            'penalty': self.penalty,
            'alpha': self.alpha,
            'l1_ratio': self.l1_ratio,
            'fit_intercept': self.intercept,
            'tol': self.tol,
            'loss': self.loss,
            'random_state': self.random_state,
            'early_stopping': self.early_stop,
            'bootstrap': self.bootstrap,
            'max_features': self.max_features,
            'max_samples': self.max_samples,
            'huber_threshold': self.delta
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
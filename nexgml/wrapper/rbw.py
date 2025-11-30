import numpy as np
from typing import Optional, Literal
from scipy.sparse import spmatrix, issparse
from ..indexing import standard_indexing
from ..modelkit.nexg import ModelLike as C

class RBWrapper:
    def __init__(self,
                model,
                n_estimators: int=100,
                learning_rate: float=0.1,
                verbose: int=0,
                random_state: int | None=None,
                early_stopping: bool=True,
                bootstrap: bool | None=True,
                max_features: Optional[Literal['sqrt', 'log2']] | int | float | None='sqrt',
                max_samples: Optional[Literal['sqrt', 'log2']] | int | float | None='sqrt',
                stoic_iter: int=10,
                tol: float | None=1e-6
                ) -> None:

        self.model = model if C(model).is_regressor() else None 
        if self.model is None:
            raise ValueError(f"Invalid model argument, {model}. Make sure your model is regressor and valid")

        self.verbose = int(verbose)

        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)

        self.early_stop = bool(early_stopping)
        self.random_state = random_state
        self.bootstrap = bool(bootstrap)
        self.stoic_iter = int(stoic_iter)
        self.tol = float(tol)

        self.max_features = max_features
        self.max_samples = max_samples
        
        self.residual_history = []
        self.supports = []
        self.feature_indices = []
        self.model_data = []

    def fit(self, X: np.ndarray | spmatrix, y: np.ndarray):
        if X.shape[0] != y.shape[0]:
             raise ValueError(f"Shape mismatch between X_train and y_train, ({X.shape[0]}) ({y.shape[0]})")
        
        if isinstance(X, spmatrix):
            X = X.tocsr()

        elif isinstance(X, (np.ndarray, list, tuple)):
            X = np.asarray(X)

        else:
            X = X.to_numpy()

        if X.ndim == 1:
            if issparse(X):
                X = X.reshape(-1, 1)
            else:
                X = X.reshape(-1, 1)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        if isinstance(y, spmatrix):
            y = y.toarray().ravel()

        elif isinstance(y, (np.ndarray, list, tuple)):
            y = np.asarray(y)

        else:
            y = y.to_numpy()

        y = y.ravel()

        if issparse(X):
            if not np.all(np.isfinite(X.data)):
                raise ValueError("Input features (X_train) contains NaN or Infinity values. Please clean your data.")

        else:
            if not np.all(np.isfinite(X)):
                raise ValueError("Input features (X_train) contains NaN or Infinity values. Please clean your data.")

        if not np.all(np.isfinite(y)):
          raise ValueError("Input target (y_train) contains NaN or Infinity values. Please clean your data.")

        n_samples, n_features = X.shape

        max_features = standard_indexing(n_features, self.max_features)
        self.stoic_iter = standard_indexing(self.n_estimators, self.stoic_iter)
        max_samples = standard_indexing(n_samples, self.max_samples)

        max_features = min(max_features, n_features)
        current_predictions = np.zeros(n_samples, dtype=float)

        for i in range(self.n_estimators):
            residuals = y - current_predictions

            self.residual_history.append(np.mean(residuals))

            if self.bootstrap:
                indices = np.random.choice(n_samples, size=max_samples, replace=True)
      
            else:
                indices = np.arange(max_samples)

            feature_idx = np.random.choice(n_features, size=max_features, replace=False)
            self.feature_indices.append(feature_idx)

            X_sub = X[indices]
            X_sub = X_sub[:, feature_idx]
            y_sub = residuals[indices]

            support = self.model() if callable(self.model) else self.model.__class__()

            if self.verbose == 1:
                print(f"|=-Training {i + 1}th model-=|")

            model_data = support.fit(X_sub, y_sub)
            self.supports.append(support)
            self.model_data.append(model_data)
            pred_sub = support.predict(X[:, feature_idx])
            current_predictions += self.learning_rate * pred_sub    

            if self.early_stop and i > 1 and i > self.stoic_iter:
                if abs(self.residual_history[-1] - self.residual_history[-2]) < self.tol:
                  if self.verbose == 1:
                      print(f"|=Stopping supports training at estimator {i + 1}=|")
                  break

    def predict(self, X: np.ndarray | spmatrix) -> np.ndarray:
        if isinstance(X, spmatrix):
            X = X.tocsr()
        else:
            X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = X.shape[0]
        final_predictions = np.zeros(n_samples, dtype=float)

        for i, support in enumerate(self.supports):
            feature_idx = self.feature_indices[i]
            X_sub = X[:, feature_idx]
            final_predictions += self.learning_rate * support.predict(X_sub)

        return final_predictions

    def get_params(self, deep=True):
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'verbose': self.verbose,
            'tol': self.tol,
            'random_state': self.random_state,
            'early_stopping': self.early_stop,
            'bootstrap': self.bootstrap,
            'max_features': self.max_features,
            'max_stoic_iter': self.stoic_iter,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
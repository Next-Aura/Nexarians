import numpy as np
from typing import Literal, Optional
from scipy.sparse import spmatrix, issparse
import warnings

from .childs.GGBR_support import Support
from ..indexing import standard_indexing

class ConfigLike:
    def __init__():
        pass

class GigaGBR:
    def __init__(self,
                config: Optional[ConfigLike] | None=None,
                n_estimators: int=100,
                max_iter: int=20,
                learning_rate: float=0.1,
                verbose: int=0,
                verbosity: Literal['light', 'heavy'] | None='light',
                penalty: Literal["l1", "l2", "elasticnet"] | None="l2",
                alpha: float=1e-4,
                l1_ratio: float=0.5,
                fit_intercept: bool=True,
                tol: float=0.0001,
                loss: Literal["mse", "rmse", "huber", "mae"] | None="mse",
                random_state: int | None=None,
                early_stopping: bool=True,
                bootstrap: bool | None=True,
                max_features: Optional[Literal['sqrt', 'log2']] | float | int | None='sqrt',
                max_samples: Optional[Literal['sqrt', 'log2']] | float | int | None='sqrt',
                huber_threshold: float | None=0.5,
                max_stoic_iter: Optional[Literal['sqrt', 'log2']] | float | int | None='sqrt',
                ):

        self.config = config

        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.estimator_lr_rate = learning_rate
        self.verbose = verbose
        self.intercept = fit_intercept

        self.tol = tol
        self.loss = loss
        self.early_stop = early_stopping
        self.random_state = random_state
        self.bootstrap = bootstrap
        self.delta = huber_threshold
        self.stoic_iter = max_stoic_iter

        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.verbosity = verbosity

        self.max_features = max_features
        self.max_samples = max_samples

        self.loss_history = []
        self.supports = []
        self.feature_indices = []
        self.model_data = []

        if self.verbose == 2 and self.verbosity == 'heavy':
          warnings.warn(f"Level 2 verbose with heavy verbosity may cause an over-log")

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

        max_samples = standard_indexing(n_samples, self.max_samples)

        self.stoic_iter = standard_indexing(self.n_estimators, self.stoic_iter)

        max_features = min(max_features, n_features)
        current_predictions = np.zeros(n_samples, dtype=float)

        for i in range(self.n_estimators):
            if self.verbose == 2:
                print(f"|=-Training {i + 1}th model-=|")

            residuals = y - current_predictions

            if self.bootstrap:
                indices = np.random.choice(n_samples, size=max_samples, replace=True)
      
            else:
                indices = np.arange(n_samples)

            feature_idx = np.random.choice(n_features, size=max_features, replace=False)
            self.feature_indices.append(feature_idx)

            X_sub = X[indices]
            X_sub = X_sub[:, feature_idx]
            y_sub = residuals[indices]

            support = Support(
                max_iter=self.max_iter,
                learning_rate=self.estimator_lr_rate,
                loss=self.loss,
                penalty=self.penalty,
                alpha=self.alpha,
                tol=self.tol,
                fit_intercept=self.intercept,
                l1_ratio=self.l1_ratio,
                verbose=self.verbose,
                huber_threshold=self.delta,
                verbosity=self.verbosity,
                early_stopping=False
                )
            
            model_data = support.fit(X_sub, y_sub)
            self.supports.append(support)
            self.model_data.append(model_data)
            self.loss_history.append(model_data['loss'])
            pred_sub = support.predict(X[:, feature_idx])

            if i > self.stoic_iter and i > 1:
                try:
                    if self.config is not None:
                        data = self.config.compute(
                            lr_input=self.estimator_lr_rate, 
                            loss_input=self.loss_history,
                            alpha_input=self.alpha,
                            tol_input=self.tol,
                            input_iter=i,
                            max_iter_input=self.n_estimators,
                            b_input=model_data['b'])

                        self.estimator_lr_rate = data['lr']
                        self.alpha = data['alpha']
                        model_data['b'] = data['b']

                except Exception as e:
                    raise ValueError(f"Config adjustment failed at estimator {i + 1} with error: {e}.")

            current_predictions += self.learning_rate * pred_sub + (model_data['b'] if self.intercept else 0)
                        
            if self.verbose == 1 and self.verbosity == 'light':
                print(f"|=Trained {i + 1}/{self.n_estimators} model, loss: {self.loss_history[-1]}=|")
              
            elif self.verbose == 1 and self.verbosity == 'heavy':
                print(f"|=Trained {i + 1}/{self.n_estimators} model, loss: {self.loss_history[-1]}, lr_rate: {self.estimator_lr_rate:.4f}, alpha: {self.alpha}=|")
    
            if self.early_stop and i > 1 and i > self.stoic_iter:
                if abs(self.loss_history[-1] - self.loss_history[-2]) < self.tol:
                  if self.verbose == 1:
                      print(f"|=Stopping estimator training at estimator {i + 1}=|")
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
            'huber_threshold': self.delta,
            'max_stoic_iter': self.stoic_iter,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
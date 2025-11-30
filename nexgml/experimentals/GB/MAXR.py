import numpy as np
from typing import Optional, Literal
from scipy.sparse import spmatrix, issparse
from math import sqrt, log2
import warnings

from .childs.MAXR_support import Support
    
class MAXR:
    def __init__(self,
                n_estimators: int=100,
                max_iter: int=200,
                learning_rate: float=0.1,
                verbose: int=0,
                verbosity: Literal['light', 'heavy'] = 'light',
                penalty: Literal["l1", "l2", "elasticnet"] | None="l2",
                invscaling: bool | None=False,
                power_t: float | None=0.25,
                alpha: float=1e-4,
                l1_ratio: float=0.5,
                fit_intercept: bool=True,
                tol: float=0.0001,
                loss: Literal["mse", "rmse", "huber", "mae"] | None="mse",
                random_state: int | None=None,
                early_stopping: bool=True,
                bootstrap: bool | None=True,
                max_features: Optional[int | str | float] = 'sqrt',
                huber_threshold: float | None=0.5,
                max_stoic_iter: Optional[int | str | float] | None='sqrt',
                adaptive_alpha: bool | None=True,
                ada_alpha_behav: Literal['parent', 'child'] | None='child',
                transfer_method: Literal['mean', 'last_data','min', 'max'] | None='mean',
                asymmetric_bias: bool | None=False,
                adaptive_parent: bool | None=False,
                min_residual_improve: float=0.0,
                parent_patience: Optional[int | str | float] | None=10
                ):
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.estimator_lr_rate = learning_rate
        self.verbose = verbose
        self.verbosity = verbosity
        self.intercept = fit_intercept
        self.power_t = power_t
        self.invscale = invscaling
        self.method = transfer_method

        self.tol = tol
        self.loss = loss
        self.early_stop = early_stopping
        self.random_state = random_state
        self.bootstrap = bootstrap
        self.delta = huber_threshold
        self.no_tol_epoch = max_stoic_iter

        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.ada_alpha = adaptive_alpha
        self.ada_alpha_behav = ada_alpha_behav

        self.max_features = max_features
        self.asym_bias = asymmetric_bias
        self.adaptive_parent = adaptive_parent
        self.min_residual_improve = min_residual_improve
        self.parent_patience = parent_patience

        self.loss_history = []
        self.alpha_history = []
        self.lr_rate_history = []
        self.residual_history = []
        self.supports = []
        self.feature_indices = []
        self.model_data = []
        self.W = None
        self.b_vec = None

        if self.verbosity == 'heavy' and self.verbose == 2:
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

        
        if self.max_features is None:
            max_features = n_features
    
        elif isinstance(self.max_features, str):
            if self.max_features == 'sqrt':
              max_features = max(1, int(sqrt(n_features)))

            if self.max_features == 'log2':
                max_features = max(1, int(log2(n_features)))

        elif isinstance(self.max_features, int):
            max_features = self.max_features

        elif isinstance(self.max_features, float):
            max_features = max(1, int(np.ceil(self.max_features * n_features)))

        else:
            raise ValueError(f"Unexpected max_feautures slicing per-estimator method {self.max_features}")

        if isinstance(self.no_tol_epoch, str):
            if self.no_tol_epoch == 'sqrt':
                self.no_tol_epoch = int(sqrt(self.n_estimators))

            elif self.no_tol_epoch == 'log2':
                self.no_tol_epoch = int(log2(self.n_estimators))

        elif isinstance(self.no_tol_epoch, int):
            self.no_tol_epoch = self.no_tol_epoch

        elif isinstance(self.no_tol_epoch, float):
            self.no_tol_epoch = int(self.no_tol_epoch * self.n_estimators)
    
        else:
            raise ValueError(f"Unexpected max_stoic_iter argument method {self.no_tol_epoch}")

        if isinstance(self.parent_patience, str):
            if self.parent_patience == 'sqrt':
                self.parent_patience = int(sqrt(self.n_estimators))

            elif self.parent_patience == 'log2':
                self.parent_patience = int(log2(self.n_estimators))

        elif isinstance(self.parent_patience, int):
            self.parent_patience = self.parent_patience

        elif isinstance(self.parent_patience, float):
            self.parent_patience = int(self.parent_patience * self.n_estimators)

        else:
            raise ValueError(f"Unexpected parent_patience argument method {self.parent_patience}")



        max_features = min(max_features, n_features)
        current_predictions = np.zeros(n_samples, dtype=float)
        patience_count = 0

        for i in range(self.n_estimators):
            residuals = y - current_predictions

            self.residual_history.append(residuals)

            if self.ada_alpha and i > 1 and self.ada_alpha_behav == 'parent' and i > self.no_tol_epoch:
                self.alpha = self.alpha * (np.mean(np.abs(self.residual_history[-1])) / np.mean(np.abs(self.residual_history[-2])))
                self.alpha = max(self.alpha, 1e-8)
                self.alpha_history.append(self.alpha)

            if self.bootstrap:
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
      
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
                method=self.method,
                scheduler=self.invscale,
                power_t=self.power_t,
                adaptive_alpha=True if self.ada_alpha and self.ada_alpha_behav == 'child' else False,
                asym_bias=self.asym_bias,
                verbosity=self.verbosity
            )

            if self.verbose == 2:
                print(f"|=Trained {i + 1}th model=|")

            model_data = support.fit(X_sub, y_sub)
            self.supports.append(support)
            self.model_data.append(model_data)
            self.loss_history.append(model_data['loss'])
            pred_sub = support.predict(X[:, feature_idx])
            current_predictions += self.learning_rate * pred_sub + (model_data['b'] if self.intercept else 0)
            self.lr_rate_history.append(model_data['lr_history'])
                
            if self.ada_alpha and self.ada_alpha_behav == 'child':
                self.alpha_history.append(model_data['alpha_history'])

            if self.adaptive_parent and i > 1 and i > self.no_tol_epoch:
                if (np.mean(np.abs(self.residual_history[-1])) - np.mean(np.abs(self.residual_history[-2]))) < self.min_residual_improve:
                    if patience_count < self.parent_patience:
                        self.n_estimators += 1
                        patience_count += 1

                    else:
                        patience_count = 0

            if self.verbose == 1 and self.verbosity == 'light':
                print(f"|=Trained {i + 1}/{self.n_estimators} model, Loss: {self.loss_history[-1]}=|")

            elif self.verbose == 1 and self.verbosity == 'heavy':
                print(f"|=Trained {i + 1}/{self.n_estimators} model, Loss: {self.loss_history[-1]}, Alpha: {self.alpha_history[-1]}, LR_rate: {self.lr_rate_history[-1]}=|")
              
            if self.early_stop and i > 1 and i > self.no_tol_epoch:
                if abs(self.loss_history[-1] - self.loss_history[-2]) < self.tol:
                  if self.verbose == 1:
                      print(f"|=Stopping supports training at estimator {i + 1}=|")
                  break
        
        n_est = len(self.supports)
        self.W = np.zeros((n_est, n_features))
        self.b_vec = np.zeros(n_est)
        for i, s in enumerate(self.supports):
            self.W[i, self.feature_indices[i]] = s.weights
            self.b_vec[i] = s.b

    def predict(self, X: np.ndarray | spmatrix) -> np.ndarray:
        if isinstance(X, spmatrix):
            X = X.tocsr()
        else:
            X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        P = X @ self.W.T
        P += self.b_vec
        final = (P * self.learning_rate).sum(axis=1)

        return final

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
            'max_stoic_iter': self.no_tol_epoch,
            'adaptive_alpha': self.ada_alpha,
            'ada_alpha_behav': self.ada_alpha_behav,
            'transfer_method': self.method,
            'parent_patience': self.parent_patience,
            'asymmetric_bias': self.asym_bias,
            'adaptive_parent': self.adaptive_parent,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
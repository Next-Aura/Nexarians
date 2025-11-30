import numpy as np
from typing import Literal, Optional
from scipy.sparse import spmatrix, issparse
import warnings

from .childs.GGBC_support import Support
from ..indexing import standard_indexing, one_hot_labeling
from ..amo.forlinear.probas import softmax

class ConfigLike:
    def __init__():
        pass

class GigaGBC:
    def __init__(self,
                n_estimators: int=100,
                max_iter: int=20,
                learning_rate: float=0.1,
                verbose: int=0,
                verbosity: Literal['light', 'heavy'] | None='light',
                penalty: Literal["l1", "l2", "elasticnet"] | None="l2",
                alpha: float=0.0001,
                l1_ratio: float=0.5,
                fit_intercept: bool=True,
                tol: float=0.0001,
                random_state: int | None=None,
                early_stopping: bool=True,
                bootstrap: bool | None=True,
                max_features: Optional[Literal['sqrt', 'log2']] | int | float | None='sqrt',
                max_samples: Optional[Literal['sqrt', 'log2']] | int | float | None='sqrt',
                max_stoic_estimator: Optional[int | str | float] | None=10,
                batch_size: Optional[int] = None,
                config: Optional[ConfigLike] | None=None
                ):
        
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.stoic_iter = max_stoic_estimator
        self.config = None

        self.verbose = verbose
        self.verbosity = verbosity
        self.intercept = fit_intercept
        self.tol = tol
        self.early_stop = early_stopping

        self.random_state = random_state
        self.bootstrap = bootstrap

        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.clip_value = 1000.0

        self.max_features = max_features
        self.max_samples = max_samples
        self.batch_size = batch_size
        
        if isinstance(learning_rate, tuple):
            if len(learning_rate) == 2:
                self.learning_rate, self.estimator_lr_rate = learning_rate

            elif len(learning_rate) == 1:
                self.learning_rate, self.estimator_lr_rate = learning_rate, learning_rate

            else:
                raise ValueError(f"Invalid learning_rate argument"
                                "Make sure you insert only 1 (or 2 in tuple) argument")
            
        else:
            self.learning_rate, self.estimator_lr_rate = float(learning_rate), float(learning_rate)

        self.loss_history = []
        self.supports = []
        self.feature_indices = []
        self.model_data = []
        self.classes_ = None
        self.n_classes_ = 0

        if self.verbose == 2 and self.verbosity == 'heavy':
          warnings.warn(f"Level 2 verbose with heavy verbosity may cause an over-log")

    def fit(self, X: np.ndarray | spmatrix, y: np.ndarray):
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Samples mismatch between X and y ({X.shape[0]}, {y.shape[0]})")
               
        if not issparse(X):
            X = np.asarray(X)

            if X.ndim == 1:
              X = X.reshape(-1, 1)

        if issparse(X):
            if not np.all(np.isfinite(X.data)):
                raise ValueError("Input features (X_train) contains NaN or Infinity values in its data. Please clean your data.")

        else:
            if not np.all(np.isfinite(X)):
                raise ValueError("Input features (X_train) contains NaN or Infinity values. Please clean your data.")
        
        if not np.all(np.isfinite(y)):
            raise ValueError("Input target (y_train) contains NaN or Infinity values. Please clean your data.")
        
        if self.random_state is not None:
            np.random.seed(self.random_state)

        y = np.asarray(y)
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        if self.n_classes_ < 2:
            raise ValueError("This classifier requires at least 2 classes.")
       
        self.max_features = standard_indexing(n_features, self.max_features)

        self.max_samples = standard_indexing(n_samples, self.max_samples)

        self.stoic_iter = standard_indexing(self.n_estimators, self.stoic_iter)

        max_features = self.max_features
        current_logits = np.zeros((n_samples, self.n_classes_), dtype=float)
        y_one_hot_full = one_hot_labeling(y, self.classes_)

        for i in range(self.n_estimators):
            if self.verbose == 2:
                print(f"|=-Training {i + 1}th model-=|")
            
            p_current = softmax(current_logits)
            
            residuals = y_one_hot_full - p_current

            if not self.bootstrap:
                indices = np.random.choice(n_samples, self.max_samples, replace=False)

            else:
                indices = np.random.choice(n_samples, self.max_samples, replace=True)
            
            feature_idx = np.random.choice(n_features, size=max_features, replace=False)
            self.feature_indices.append(feature_idx)
            X_sub = X[indices]

            if issparse(X_sub):
                X_sub = X_sub[:, feature_idx]

            else:
                X_sub = X_sub[:, feature_idx]
            
            y_sub = residuals[indices] 
            support = Support(
                n_classes=self.n_classes_,
                max_iter=self.max_iter,
                learning_rate=self.estimator_lr_rate,
                penalty=self.penalty,
                alpha=self.alpha,
                fit_intercept=self.intercept,
                l1_ratio=self.l1_ratio,
                verbose=self.verbose,
                batch_size=self.batch_size,
                random_state=self.random_state
            )
            
            model_data = support.fit(X_sub, y_sub)
            self.supports.append(support)
            self.model_data.append(model_data)
            self.loss_history.append(model_data['loss'])

            if i > self.stoic_iter and i > 1 and (self.config.behav == 'parent' if self.config else False):
                try:
                    if self.config is not None:
                        data = self.config.compute(
                            lr_input=self.learning_rate, 
                            loss_input=self.loss_history,
                            alpha_input=self.alpha,
                            tol_input=self.tol,
                            input_iter=i,
                            max_iter_input=self.n_estimators,
                            b_input=model_data['b'])

                        self.learning_rate = data['lr']
                        self.alpha = data['alpha']
                        model_data['b'] = data['b']

                except Exception as e:
                    raise ValueError(f"Config adjustment failed at estimator {i + 1} with error: {e}.")

            pred_logits = support.predict_logit(X[:, feature_idx])

            pred_logits = np.clip(pred_logits, -self.clip_value, self.clip_value)
            current_logits += self.learning_rate * pred_logits + (support.b if self.intercept and hasattr(support, 'b') else 0)

            if self.verbose == 1 and self.verbosity == 'light':
                print(f"|=Trained model {i+1}/{self.n_estimators}, loss: {self.loss_history[-1]:.6f}=|")

            elif self.verbose == 1 and self.verbosity == 'heavy':
                print(f"|=Trained model {i+1}/{self.n_estimators}, loss: {self.loss_history[-1]}, lr_rate: {self.estimator_lr_rate:.4f}, alpha: {self.alpha}=|")

            if self.early_stop and i > 1 and i > self.stoic_iter:
                 if abs(self.loss_history[-1] - self.loss_history[-2]) < self.tol:
                  if self.verbose == 1:
                      print(f"|=-Stopping supports training at estimator {i + 1}-=|")
                  break

    def predict_proba(self, X: np.ndarray | spmatrix) -> np.ndarray:
        if not issparse(X):
            X = np.asarray(X)
        n_samples = X.shape[0]
        final_logits = np.zeros((n_samples, self.n_classes_), dtype=float)

        for i, tree in enumerate(self.supports):
            feature_idx = self.feature_indices[i]
            if issparse(X):
                X_sub = X[:, feature_idx]
            else:
                X_sub = X[:, feature_idx]

            final_logits += self.learning_rate * tree.predict_logit(X_sub) + (self.model_data[i]['b'] if self.intercept else 0)

        return softmax(final_logits)

    def predict(self, X: np.ndarray | spmatrix) -> np.ndarray:
        probas = self.predict_proba(X)
        pred_class_indices = np.argmax(probas, axis=1)

        return np.array([self.classes_[idx] for idx in pred_class_indices])

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
            'random_state': self.random_state,
            'early_stopping': self.early_stop,
            'bootstrap': self.bootstrap,
            'max_features': self.max_features,
            'max_samples': self.max_samples
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
import numpy as np
from typing import Optional, Literal
from scipy.sparse import spmatrix, issparse
from ..indexing import standard_indexing, one_hot_labeling
from ..modelkit.nexg import ModelLike as C
from nexgml.amo.forlinear import softmax

class CBWrapper:
    def __init__(self,
                model,
                n_estimators: int=100,
                fit_intercept: bool=True,
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

        self.model = model if C(model).is_classifier() else None 
        if self.model is None:
            raise ValueError(f"Invalid model argument, {model}. Make sure your model is a classifier and valid")

        self.verbose = int(verbose)

        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)

        self.early_stop = bool(early_stopping)
        self.random_state = random_state
        self.intercept = fit_intercept
        self.bootstrap = bool(bootstrap)
        self.stoic_iter = int(stoic_iter)
        self.tol = float(tol)

        self.max_features = max_features
        self.max_samples = max_samples
        
        self.classes_ = None
        self.n_classes_ = None
        self.residual_history = []
        self.supports = []  # Will be list of lists, each per estimator and class
        self.feature_indices = []
        self.model_data = []

    def predict_proba(self, X: np.ndarray | spmatrix) -> np.ndarray:
        if not issparse(X):
            X = np.asarray(X)
        n_samples = X.shape[0]
        final_logits = np.zeros((n_samples, self.n_classes_), dtype=float)

        for i, supports_per_class in enumerate(self.supports):
            feature_idx = self.feature_indices[i]
            for c, support in enumerate(supports_per_class):
                if issparse(X):
                    X_sub = X[:, feature_idx]
                else:
                    X_sub = X[:, feature_idx]

                if hasattr(support, "predict_proba"):
                    probs = support.predict_proba(X_sub)
                    eps = 1e-15
                    clipped_probs = np.clip(probs, eps, 1 - eps)
                    logits = np.log(clipped_probs)
                    # Handle if support model is not strictly binary classifier
                    if logits.shape[1] == 2:
                        logit_class1 = logits[:, 1]
                    elif logits.shape[1] >= self.n_classes_:
                        # Defensive: If output has more classes, take the class c column if possible
                        if c < logits.shape[1]:
                            logit_class1 = logits[:, c]  # take logit for relevant class
                        else:
                            raise ValueError(f"Class index {c} out of range for model logits with shape {logits.shape}")
                    else:
                        raise ValueError(f"Mismatch in classes between model ({logits.shape[1]}) and expected 2 or {self.n_classes_} classes")
                    
                    try:
                        bias = self.model_data[i][c].get('b', 0) if self.intercept else 0
                    except AttributeError:
                        bias = 0
                    
                    final_logits[:, c] += self.learning_rate * logit_class1 + bias

                elif hasattr(support, "predict"):
                    preds = support.predict(X_sub)
                    one_hot = np.zeros((n_samples, logits.shape[1] if 'logits' in locals() else 2))
                    # Support model may have more than 2 classes; handle accordingly
                    n_preds_classes = one_hot.shape[1]
                    for idx in range(n_preds_classes):
                        one_hot[:, idx] = (preds == idx).astype(float)
                    eps = 1e-15
                    clipped_probs = np.clip(one_hot, eps, 1 - eps)
                    logits = np.log(clipped_probs)
                    bias = self.model_data[i][c].get('b', 0) if self.intercept else 0
                    # Defensive: if binary output, take index 1, else class c
                    logit_idx = 1 if logits.shape[1] == 2 else (c if c < logits.shape[1] else 0)
                    final_logits[:, c] += self.learning_rate * logits[:, logit_idx] + bias

                else:
                    raise AttributeError("Support model lacks predict_proba or predict")

        return softmax(final_logits)

    def fit(self, X: np.ndarray | spmatrix, y: np.ndarray):
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Shape mismatch between X and y, {X.shape[0]} != {y.shape[0]}")

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
        y = np.asarray(y.ravel())

        if issparse(X):
            if not np.all(np.isfinite(X.data)):
                raise ValueError("X contains NaN or Infinity values")
        else:
            if not np.all(np.isfinite(X)):
                raise ValueError("X contains NaN or Infinity values")

        if not np.all(np.isfinite(y)):
            raise ValueError("y contains NaN or Infinity values")

        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        max_features = standard_indexing(n_features, self.max_features)
        self.stoic_iter = standard_indexing(self.n_estimators, self.stoic_iter)
        max_samples = standard_indexing(n_samples, self.max_samples)
        max_features = min(max_features, n_features)

        current_logits = np.zeros((n_samples, self.n_classes_), dtype=float)
        y_one_hot_full = one_hot_labeling(y, self.classes_)

        for i in range(self.n_estimators):
            if self.verbose == 2:
                print(f"|=- Training model {i + 1} =-|")

            p_current = softmax(current_logits)
            residuals = y_one_hot_full - p_current

            if not self.bootstrap:
                indices = np.random.choice(n_samples, max_samples, replace=False)
            else:
                indices = np.random.choice(n_samples, max_samples, replace=True)

            feature_idx = np.random.choice(n_features, size=max_features, replace=False)
            self.feature_indices.append(feature_idx)
            X_sub = X[indices]
            X_sub = X_sub[:, feature_idx]

            supports_per_class = []
            model_data_per_class = []

            # Fit one support model per class on residuals[:,c]
            for c in range(self.n_classes_):
                y_sub = residuals[indices, c]

                support = self.model() if callable(self.model) else self.model.__class__()
                model_data = support.fit(X_sub, y_sub)
                supports_per_class.append(support)
                model_data_per_class.append(model_data)

            self.supports.append(supports_per_class)
            self.model_data.append(model_data_per_class)

            # Combine predictions of all supports for current estimator
            pred_proba = np.zeros((len(indices), self.n_classes_))
            for c, support in enumerate(supports_per_class):
                if hasattr(support, "predict_proba"):
                    pred_prob_c = support.predict_proba(X_sub)
                    # Use probability of positive class (index 1) for residual estimate
                    pred_proba[:, c] = pred_prob_c[:, 1]
                else:
                    preds = support.predict(X_sub)
                    pred_proba[:, c] = (preds == 1).astype(float)

            # Update the current logits for the sampled indices only
            current_logits[indices] += self.learning_rate * np.log(np.clip(pred_proba, 1e-15, 1 - 1e-15))

            residual_norm = np.linalg.norm(residuals)
            self.residual_history.append(residual_norm)

            if self.early_stop and i > 1 and i > self.stoic_iter:
                if abs(self.residual_history[-1] - self.residual_history[-2]) < self.tol:
                    if self.verbose == 1:
                        print(f"|= Stopping training at estimator {i + 1} =|")
                    break

    def predict(self, X: np.ndarray | spmatrix) -> np.ndarray:
        probas = self.predict_proba(X)
        pred_class_indices = np.argmax(probas, axis=1)
        return np.array([self.classes_[idx] for idx in pred_class_indices])

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
            'max_samples': self.max_samples,
            'stoic_iter': self.stoic_iter,
            'fit_intercept': self.intercept,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

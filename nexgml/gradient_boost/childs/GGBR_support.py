import numpy as np
from typing import Literal
from scipy.sparse import issparse
from ...amo import mean_squared_error, mean_absolute_error, smoothl1_loss

class Support:
    def __init__(self,
                max_iter: int=100,
                learning_rate: float=0.01,
                verbose: int=0,
                verbosity: str | None='light',
                penalty: Literal["l1", "l2", "elasticnet"] | None="l2",
                alpha: float=0.0001,
                l1_ratio: float=0.5,
                fit_intercept: bool=True,
                tol: float=0.0001,
                loss: Literal["mse", "rmse", "huber", "mae"] | None="mse",
                early_stopping: bool=True,
                huber_threshold: float | None=0.5,
                ):
        
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.intercept = fit_intercept
        self.verbosity = verbosity

        self.tol = tol
        self.loss = loss
        self.early_stop = early_stopping
        self.delta = huber_threshold
    
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.alpha = alpha

        self.loss_history = []
        self.weights = None
        self.b = 0.0

    def _calculate_loss(self, X, y) -> float:
        pred = X @ self.weights
        if self.intercept:
           pred += self.b

        if self.loss not in ("mse", "huber", "rmse", "mae"):
            raise ValueError(f"{self.loss} is not applied to the model, change the loss function to the applied one")
        loss = mean_squared_error(y, pred)

        penalty = 0

        if self.penalty == "l1":
          penalty = self.alpha * np.sum(np.abs(self.weights))

        elif self.penalty == "l2":
          penalty = self.alpha * np.sum(self.weights**2)

        elif self.penalty == "elasticnet":
          l1 = self.l1_ratio * np.sum(np.abs(self.weights))
          l2 = (1 - self.l1_ratio) * np.sum(self.weights**2)
          penalty = self.alpha * (l1 + l2)

        if self.loss == "rmse":
          loss = np.sqrt(loss)

        elif self.loss == 'huber':
           loss = smoothl1_loss(y, pred)  

        elif self.loss == 'mae':
           loss = mean_absolute_error(y, pred)

        return loss + penalty
    
    def grad(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        f = X @ self.weights + self.b - y
        grad_b = 0.0

        if self.loss == 'mse':
            grad_w = X.T @ (2 * f) / X.shape[0]

            if self.intercept:
              grad_b = np.mean(2 * f)

        elif self.loss == 'rmse':
           mse = np.mean(f**2)
           rmse = np.sqrt(mse)
           grad_w = (X.T @ f) / (X.shape[0] * rmse + 1e-10)

           if self.intercept:
              grad_b = np.mean(f) / (rmse + 1e-10)

        elif self.loss == 'huber':
           grad_w = X.T @ np.where(np.abs(f) <= self.delta, f, self.delta * np.sign(f)) / X.shape[0]

           if self.intercept:
             grad_b = np.mean(np.where(np.abs(f) <= self.delta, f, self.delta * np.sign(f)))

        elif self.loss == 'mae':
           grad_w = X.T @ np.sign(f) / len(X)
           
           # Calculate bias gradient if intercept is used
           if self.intercept:
            grad_b = np.mean(np.sign(f))

        grad_w_penalty = np.zeros_like(self.weights)

        if self.penalty == "l1":
            grad_w_penalty = self.alpha * np.sign(self.weights)

        elif self.penalty == "l2":
            grad_w_penalty = 2 * self.alpha * self.weights

        elif self.penalty == "elasticnet":
            l1 = self.l1_ratio * np.sign(self.weights)
            l2 = 2 * ((1 - self.l1_ratio) * self.weights)
            grad_w_penalty = self.alpha * (l2 + l1)

        else:
           raise ValueError("Invalid penalty type, Choose from 'l1', 'l2', or 'elasticnet'")

        grad_w = grad_w + grad_w_penalty

        return grad_w, grad_b
        
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if X_test.ndim == 1:
            X_processed = X_test.reshape(-1, 1)

        else:
            X_processed = X_test

        if self.weights is None:
            raise ValueError("Weight not defined, try to train the model with fit() function first")

        pred = X_processed @ self.weights + self.b

        return pred
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        if X_train.ndim == 1:
            X_processed = X_train.reshape(-1, 1)

        else:
            X_processed = X_train

        num_samples, num_features = X_processed.shape

        self.weights = np.zeros(num_features)
        self.b = 0.0

        if isinstance(y_train, (np.ndarray, list, tuple)):
            y_processed = np.asarray(y_train)

        else:
            y_processed = y_train.to_numpy()

        if issparse(X_processed):
            if not np.all(np.isfinite(X_processed.data)):
                raise ValueError("Input features (X_train) contains NaN or Infinity values. Please clean your data.")
        else:
            if not np.all(np.isfinite(X_processed)):
                raise ValueError("Input features (X_train) contains NaN or Infinity values. Please clean your data.")

        if not np.all(np.isfinite(y_processed)):
          raise ValueError("Input target (y_train) contains NaN or Infinity values. Please clean your data.")
        
        if X_processed.shape[0] != y_processed.shape[0]:
            raise ValueError(
                f"Number of samples in X_train ({X_processed.shape[0]}) "
                f"must match number of samples in y_train ({y_processed.shape[0]})."
            )
        
        y_processed = y_processed.ravel()
        
        for i in range(self.max_iter):
            grad_w, grad_b = self.grad(X_processed, y_processed)


            self.weights -= self.learning_rate * grad_w

            if self.intercept:
              self.b -= self.learning_rate * grad_b

            loss = self._calculate_loss(X_processed, y_processed)

            self.loss_history.append(loss)

            if not np.all(np.isfinite(self.weights)) or (self.intercept and not np.isfinite(self.b)):
                raise ValueError(f"Warning: Weights or bias became NaN/Inf at epoch {i + 1}. Stopping training early.")

            if np.any(np.isnan(self.weights)) or np.any(np.isinf(self.weights)) or np.isnan(self.b) or np.isinf(self.b):
                raise ValueError(f"There's NaN in epoch {i + 1} during the training process")

            if self.verbose == 2 and (1 + i) % 10 == 0 and self.verbosity == 'light':
                print(f"|={i + 1}. Grad_w: {np.mean(self.weights):.4f}, Grad_b: {self.b:.4f}, Loss: {loss:.4f}=|")

            elif self.verbose == 2 and self.verbosity == 'heavy':
                print(f"|={i + 1}. Grad_w: {np.mean(self.weights):.4f}, Grad_b: {self.b:.4f}, Loss: {loss:.4f}=|")


            if self.early_stop and i > 1:
                if abs(self.loss_history[-1] - self.loss_history[-2]) < self.tol:
                  break
        return {'weights': self.weights, 'b': self.b, 'loss': np.mean(self.loss_history)}
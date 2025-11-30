import numpy as np
from typing import Literal, Optional
from nexgml.indexing import standard_indexing

class FlashyRegressor:
    def __init__(self, 
                max_iter: int=1000, 
                learning_rate: float=0.01, 
                verbose: int=0, 
                penalty: Literal["l1", "l2", "elasticnet"] | None="l2", 
                alpha: float=0.0001,
                max_features: Optional[Literal['sqrt', 'log2']] | int | float | None='sqrt',
                max_samples: Optional[Literal['sqrt', 'log2']] | int | float | None='sqrt',
                shuffle: bool | None = True,
                l1_ratio: float=0.5, 
                fit_intercept: bool=True,
                adaptive_alpha: bool | None=False,
                tol: float=0.0001, 
                loss: Literal["mse", "rmse"] | None="mse",
                early_stopping: bool=True,
                lr_scheduler: Literal["constant", "invscaling"] | None='constant',
                power_t: float | None=0.25,
                stoic_iter: Optional[Literal['sqrt', 'log2']] | int | float | None=10
                ):
        """
        FlashyR or Flash Regressor is a linear regression model that uses gradient descent optimization to minimize the loss function.
        In every iteration FlashyR will only use some data from the given training data for speed boosting and low computation.
        It supports L1, L2, and Elastic Net regularization to prevent overfitting.
        MSE and RMSE loss functions are available.
        """
        # Hyperparameters
        self.max_iter = int(max_iter)
        self.learning_rate = float(learning_rate)
        self.verbose = int(verbose)
        self.intercept = bool(fit_intercept)
        self.ada_alpha = bool(adaptive_alpha)

        self.tol = float(tol)
        self.loss = loss
        self.early_stop = bool(early_stopping)
        self.max_features = max_features
        self.max_samples = max_samples
        self.shuffle = bool(shuffle)
        self.pow_t = float(power_t)
        self.stoic_iter = stoic_iter
    
        self.penalty = penalty
        self.l1_ratio = float(l1_ratio)
        self.alpha = float(alpha)
        self.lr_scheduler = lr_scheduler if lr_scheduler in ('constant', 'invscaling') else 'invscaling'

        self.loss_history = []
        self.feature_idx = []
        self.weights = None
        self.b = 0.0
    # Calculate loss with regulation.
    def _calculate_loss(self, X, y) -> float:
        """Calculating loss with regulation, MSE and RMSE available.
        Penalty, l1, l2, elasticnet available."""
        mse_cal = X @ self.weights[self.feature_idx[-1]] + self.b - y
        mse = np.mean(mse_cal**2)

        penalty = 0

        if self.penalty == "l1":
          penalty = self.alpha * np.sum(np.abs(self.weights[self.feature_idx[-1]]))

        elif self.penalty == "l2":
          penalty = self.alpha * np.sum(self.weights[self.feature_idx[-1]]**2)

        elif self.penalty == "elasticnet":
          l1 = self.l1_ratio * np.sum(np.abs(self.weights[self.feature_idx[-1]]))
          l2 = (1 - self.l1_ratio) * np.sum(self.weights[self.feature_idx[-1]]**2)
          penalty = self.alpha * (l1 + l2)

        else:
           raise ValueError(f"Invalid penalty argument {self.penalty}")

        if self.loss == "rmse":
          mse = np.sqrt(mse)
        
        elif self.loss == 'mse':
           mse = mse

        else:
           raise ValueError(f"Invalid loss argument {self.loss}")
           
        return mse + penalty
    # Calculate gradient with regulation
    def grad(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        """Calculate gradient of loss function with regulation.
        L1, L2, and Elastic Net available."""
        f = X @ self.weights[self.feature_idx[-1]] + self.b - y

        grad_b = 0.0

        if self.loss == 'mse':

            grad_w = X.T @ (2 * f) / len(X)

            if self.intercept:
              grad_b = np.mean(2 * f)

        elif self.loss == 'rmse':
           rmse = np.sqrt(np.mean(f**2))
           grad_w = (X.T @ (2 * f)) / (X.shape[0] * rmse + 1e-10)

           if self.intercept:
              grad_b = np.mean(2 * f) / (rmse + 1e-10)

        else:
           raise ValueError(f"Invalid loss argument {self.loss}")

        grad_w_penalty = np.zeros_like(self.weights[self.feature_idx[-1]])

        if self.penalty == "l1":
            grad_w_penalty = self.alpha * np.sign(self.weights[self.feature_idx[-1]])

        elif self.penalty == "l2":
            grad_w_penalty = 2 * self.alpha * self.weights[self.feature_idx[-1]]

        elif self.penalty == "elasticnet":
            l1 = self.l1_ratio * np.sign(self.weights[self.feature_idx[-1]])
            l2 = 2 * ((1 - self.l1_ratio) * self.weights[self.feature_idx[-1]])
            grad_w_penalty = self.alpha * (l2 + l1)

        else:
           raise ValueError(f"Invalid penalty argument {self.penalty}")

        grad_w = grad_w + grad_w_penalty

        return grad_w, grad_b
    # Prediction funstion
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict the target values for the given input features using the trained model.
        """
        if X_test.ndim == 1:
            X_processed = X_test.reshape(-1, 1)

        else:
            X_processed = X_test

        if self.weights is None:
            raise ValueError("Weight not defined, try to train the model with fit() function first")
        
        final_pred = X_processed @ self.weights + self.b

        return final_pred
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fit the model to the training data using gradient descent method."""
        stoic_iter = standard_indexing(self.max_iter, self.stoic_iter)

        if X_train.ndim == 1:
            X_processed = X_train.reshape(-1, 1)

        else:
            X_processed = X_train

        num_samples, num_features = X_processed.shape
        
        if self.weights is None or self.weights.shape[0] != num_features:
         self.weights = np.zeros(num_features)

        if isinstance(y_train, (np.ndarray, list, tuple)):
            y_processed = np.asarray(y_train)

        else:
            y_processed = y_train.to_numpy()

        if not np.all(np.isfinite(X_processed)):
          raise ValueError("Input features (X_train) contains NaN or Infinity values. Please clean your data.")

        if not np.all(np.isfinite(y_processed)):
          raise ValueError("Input target (y_train) contains NaN or Infinity values. Please clean your data.")
        
        if X_processed.shape[0] != y_processed.shape[0]:
            raise ValueError(
                f"Number of samples in X_train ({X_processed.shape[0]}) "
                f"must match number of samples in y_train ({y_processed.shape[0]})."
            )
        
        max_features = standard_indexing(num_features, self.max_features)
        max_samples = standard_indexing(num_samples, self.max_samples)

        for i in range(self.max_iter):
            if i > stoic_iter:
              if self.lr_scheduler == 'invscaling':
                self.learning_rate = self.learning_rate / ((i + 1)**self.pow_t + 1e-8)

            if self.shuffle:
              indices = np.random.choice(num_samples, max_samples, replace=True)

            else:
              indices = np.random.choice(num_samples, max_samples, replace=False)

            feature_idx = np.random.choice(num_features, max_features, replace=False)
            X_sliced = X_processed[indices]
            X_processed_feat = X_sliced[:, feature_idx]
            self.feature_idx.append(feature_idx)

            y_sliced = y_processed[indices]
            try:
              grad_w, grad_b = self.grad(X_processed_feat, y_sliced)

            except ValueError:
               grad_w, grad_b = self.grad(X_processed_feat, y_sliced.ravel())

            self.weights[self.feature_idx[-1]] -= self.learning_rate * grad_w

            if self.intercept:
             self.b -= self.learning_rate * grad_b

            try:
              loss = self._calculate_loss(X_processed_feat, y_sliced)

            except ValueError:
               loss = self._calculate_loss(X_processed_feat, y_sliced.ravel())

            self.loss_history.append(loss)

            if self.ada_alpha and i > stoic_iter:
                self.alpha *= (self.loss_history[-1] / self.loss_history[-2])

            if not np.all(np.isfinite(self.weights)) or (self.intercept and not np.isfinite(self.b)):
                raise ValueError(f"Warning: Weights or bias became NaN/Inf at epoch {i + 1}. Stopping training early.")

            if np.any(np.isnan(self.weights)) or np.any(np.isinf(self.weights)) or np.isnan(self.b) or np.isinf(self.b):
                raise ValueError(f"There's NaN in epoch {i + 1} during the training process")

            if self.verbose == 1:
                print(f"Epoch {i+1}/{self.max_iter}. Grad_w: {np.mean(self.weights):.6f}, Grad_b: {self.b:.6f}, Loss: {loss:.6f}, Lr_rate: {self.learning_rate:.6f}, Alpha: {self.alpha:.6f}")
            
            if self.early_stop and i > 1 and i > stoic_iter:
                if abs(self.loss_history[-1] - self.loss_history[-2]) < self.tol:
                  break
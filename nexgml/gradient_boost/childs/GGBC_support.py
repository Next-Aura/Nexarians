import numpy as np
from scipy.sparse import issparse, spmatrix
from typing import Literal

try:
    from numba import jit
except ImportError:
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

class Support:
    def __init__(
            self,
            n_classes: int | None = None,
            max_iter: int=1000,
            penalty: Literal["l1", "l2", "elasticnet"] | None="l2",
            learning_rate: float=0.01, alpha: float=0.0001,
            l1_ratio: float=0.5,
            fit_intercept: bool=True,
            epsilon: float=1e-8,
            random_state: int | None=None,
            verbose: int | None=0,
            verbosity: Literal['light', 'heavy'] | None='light',
            batch_size: int | None = None,
            ):

        self.intercept = fit_intercept
        self.max_iter = max_iter
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.lr_rate = learning_rate

        self.epsilon = epsilon

        self.verbose = verbose
        self.verbosity = verbosity
        self.random_state = random_state

        self.alpha = alpha

        self.weights = None
        self.b = None
        self.loss_history = []

        self.n_classes = n_classes
        self.batch_size = batch_size

        self.grad_clip = 10.0

        if self.random_state is not None:
            np.random.seed(self.random_state)

    def loss(self, y_true, z_pred):
        """
        If boosting==False: y_true assumed one-hot ; compute categorical CE (existing)
        If boosting==True: y_true are real residuals; compute mean squared error (MSE).
        Add regularization term accordingly.
        """
        n = y_true.shape[0]

        mse = np.mean((z_pred - y_true) ** 2)
        penalty = 0.0
        if self.penalty == "l1":
            penalty = self.alpha * np.sum(np.abs(self.weights)) / max(1, n)
        elif self.penalty == "l2":
            penalty = self.alpha * np.sum(self.weights ** 2) / max(1, n)
        elif self.penalty == "elasticnet":
            l1 = self.l1_ratio * np.sum(np.abs(self.weights))
            l2 = (1 - self.l1_ratio) * np.sum(self.weights**2)
            penalty = self.alpha * (l1 + l2) / max(1, n)
        return mse + penalty

    def grad(self, X_scaled: np.ndarray | spmatrix, y_true: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (grad_w, grad_b, model_output)
        If boosting: model_output = z (raw), grad computed wrt MSE (z - y_true)
        Else: compute as original (softmax cross entropy)
        """
        if X_scaled.ndim == 1:
            X_scaled = X_scaled.reshape(1, -1)

        if y_true.ndim == 1:
            raise ValueError("y_true in grad should be 2D (one-hot or residual matrix).")

        if X_scaled.shape[0] != y_true.shape[0]:
             raise ValueError(f"Mismatch in number of samples between X_scaled {X_scaled.shape} and y_true {y_true.shape}")

        if self.weights is None:
             raise RuntimeError("Model weights are not initialized. Call fit() first.")
        
        if X_scaled.shape[1] != self.weights.shape[0]:
             raise ValueError(f"Mismatch in number of features: X_scaled has {X_scaled.shape[1]}, weights expect {self.weights.shape[0]}")
        
        if y_true.shape[1] != self.weights.shape[1]:
             raise ValueError(f"Mismatch in number of classes: y_true has {y_true.shape[1]}, weights produce {self.weights.shape[1]}")

        z = X_scaled @ self.weights
        if self.intercept and self.b is not None:
            z = z + self.b  # broadcast across rows

        error = (z - y_true) / X_scaled.shape[0]
        if issparse(X_scaled):
            grad_w = (X_scaled.T @ error)
        else:
            grad_w = np.dot(X_scaled.T, error)
        grad_b = np.mean(z - y_true, axis=0) if self.intercept else np.zeros(self.n_classes)
        return grad_w, grad_b, z

    def fit(self, X_train, y_train):
        if X_train.shape[0] != y_train.shape[0]:
             raise ValueError(f"Samples mismatch between X_train {X_train.shape} and y_train {y_train.shape}")

        X_processed = X_train
        y_processed = y_train

        num_samples, num_features = X_processed.shape

        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer or None.")

        effective_batch_size = self.batch_size if self.batch_size is not None else num_samples
        effective_batch_size = min(effective_batch_size, num_samples)

        if self.n_classes is None:
             self.n_classes = y_processed.shape[1]
        else:
             if self.n_classes != y_processed.shape[1]:
                 raise ValueError(f"Provided n_classes ({self.n_classes}) does not match y_train shape ({y_processed.shape})")

        self.loss_history = []

        if self.n_classes < 2:
            raise ValueError("Class label must have at least 2 types.")

        if self.weights is None or self.weights.shape != (num_features, self.n_classes):
            # small initialization
            self.weights = np.random.normal(0, 0.01, (num_features, self.n_classes))

        if self.intercept and (self.b is None or self.b.shape != (self.n_classes,)):
            self.b = np.zeros(self.n_classes, dtype=np.float64)

        for i in range(self.max_iter):
            epoch_loss_sum = 0.0
            num_batches = 0

            # config handling omitted per instruction (config disabled)

            for batch_start in range(0, num_samples, effective_batch_size):
                batch_end = min(batch_start + effective_batch_size, num_samples)
                X_batch = X_processed[batch_start:batch_end]
                y_batch = y_processed[batch_start:batch_end]

                grad_w, grad_b, model_output = self.grad(X_batch, y_batch)
                # compute loss using appropriate target vs model_output
                batch_loss = self.loss(y_batch, model_output)
                epoch_loss_sum += batch_loss
                num_batches += 1

                # apply regularization to gradient (L2)
                if self.penalty == "l2":
                    grad_w += (2.0 * self.alpha / max(1, X_batch.shape[0])) * self.weights
                elif self.penalty == "l1":
                    grad_w += (self.alpha / max(1, X_batch.shape[0])) * np.sign(self.weights)

                # gradient clipping (option)
                if self.grad_clip is not None:
                    norm = np.linalg.norm(grad_w)
                    if norm > 0 and norm > self.grad_clip:
                        grad_w = grad_w * (self.grad_clip / norm)

                # update params
                self.weights -= self.lr_rate * grad_w
                if self.intercept:
                    self.b -= self.lr_rate * grad_b

            avg_epoch_loss = epoch_loss_sum / max(1, num_batches)
            self.loss_history.append(avg_epoch_loss)

            if self.verbose == 2 and (1 + i) % 10 == 0 and self.verbosity == 'light':
                print(f"|={i + 1}. Mean_w: {np.mean(self.weights):.6f}, Mean_b: {np.mean(self.b):.6f}, Loss: {avg_epoch_loss:.6f}=|")

            elif self.verbose == 2 and self.verbosity == 'heavy':
                print(f"|={i + 1}. Mean_w: {np.mean(self.weights):.6f}, Mean_b: {np.mean(self.b):.6f}, Loss: {avg_epoch_loss:.6f}=|")

        return {
            "weights": self.weights.copy(),
            "b": self.b.copy() if self.b is not None else None,
            "loss": float(self.loss_history[-1]) if self.loss_history else 0.0,
            }

    def predict_logit(self, X):
        """
        Return the raw logits (z) or softmax probabilities depending on mode.
        - boosting=True: return linear outputs (z) only
        - boosting=False: return softmax probabilities
        """
        if self.weights is None:
            raise RuntimeError("Model has not been fitted yet.")

        z = X @ self.weights
        if self.intercept and self.b is not None:
            z = z + self.b  # broadcast per class

        return z

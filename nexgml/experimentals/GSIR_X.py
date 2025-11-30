import numpy as np
from typing import Literal
from scipy.sparse import spmatrix, issparse
from math import sqrt, log2

class IntenseRegressorX:
    def __init__(
        self, 
        max_iter: int=1000, 
        lr_scheduler: Literal["constant", "invscaling", 'plateau'] | None='invscaling', 
        verbose: int=0, 
        penalty: Literal["l1", "l2", "elasticnet", 'asymmtric_bias'] | None="l2", 
        alpha: float=0.0001, 
        l1_ratio: float=0.5, 
        fit_intercept: bool=True, 
        tol: float=0.0001, 
        eta0: float=0.01, 
        power_t: float=0.5, 
        epsilon: float=1e-8, 
        batch_size: int=16, 
        shuffle: bool=True, 
        random_state: int | None=None, 
        optimizer: Literal['mbgd', 'adam', 'adamw'] | None='mbgd', 
        patience: int=5, 
        factor: float=0.8, 
        loss: Literal['mse', 'rmse', 'mae', 'huber'] | None='mse', 
        delta: int=0.5,
        early_stop: bool=True,
        max_stoic_iter: int | None = 10,
        adaptive_alpha: bool | None=True,
        bias_decay: bool | None = False,
        asymm_b_window: int | None=5):
        
        self.max_iter = max_iter
        self.lr_scheduler = lr_scheduler
        self.random_state = random_state
        self.best_loss = float('inf')
        self.wait = 0
        self.patience = patience
        self.factor = factor
        self.loss = loss

        self.verbose = verbose
        self.intercept = fit_intercept
        self.eta0 = eta0
        self.epsilon = epsilon
        self.current_lr = None
        self.delta = delta

        self.optimizer = optimizer
        self.early_stop = early_stop
        self.tol = tol
        self.stoic_iter = max_stoic_iter
        self.ada_alpha = adaptive_alpha

        self.power_t = power_t
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.b_decay = bias_decay
    
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.asymm_b_window = asymm_b_window

        self.loss_history = []
        self.alpha_history = []
        self.lr_rate_history = []
        self.weights = None
        self.b = 0.0

        if self.optimizer == 'adam' or self.optimizer == 'adamw':
            self.m_w = None
            self.v_w = None
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon_adam = 1e-8
            self.m_b = 0.0
            self.v_b = 0.0

    def _loss(self, X, y) -> float:
        mse_cal = X @ self.weights + self.b - y
        mse = np.mean(mse_cal**2)

        penalty = 0

        if self.penalty == "l1":
            penalty = self.alpha * np.sum(np.abs(self.weights))

        elif self.penalty == "l2":
            penalty = self.alpha * np.sum(self.weights**2)

        elif self.penalty == "elasticnet":
            l1 = self.l1_ratio * np.sum(np.abs(self.weights))
            l2 = (1 - self.l1_ratio) * np.sum(self.weights**2)
            penalty = self.alpha * (l1 + l2)
        
        elif self.penalty == 'asymmtric_bias':
            if len(self.loss_history) >= 2:
                if self.loss_history[-2] < self.loss_history[-1]:
                    penalty = self.alpha * np.sum(self.weights**2)

                else:
                    penalty = self.alpha * np.sum(np.abs(self.weights))

        if self.loss == 'rmse':
            mse = np.sqrt(mse)

        elif self.loss == 'mae':
            mse = np.mean(np.abs(mse_cal))

        elif self.loss == 'huber':
            mse = np.mean(np.where(np.abs(mse_cal) <= self.delta, 0.5 * mse_cal**2, self.delta * (np.abs(mse_cal) - 0.5 * self.delta)))

        else:
            mse = mse

        return mse + penalty
    
    def _grad(self, X, y: np.ndarray) -> tuple[np.ndarray, float]:
        f = X @ self.weights + self.b - y

        num_samples = X.shape[0]

        grad_b = 0.0

        if self.loss == 'mse':

            grad_w = X.T @ (2 * f) / num_samples

            if self.intercept:
              grad_b = np.mean(2 * f)

        elif self.loss == 'rmse':
            rmse = np.sqrt(np.mean(f**2))
            grad_w = (X.T @ (2 * f)) / (X.shape[0] * rmse + 1e-10)

            if self.intercept:
                grad_b = np.mean(2 * f) / (rmse + 1e-10)


        elif self.loss == 'mae':
            grad_w = X.T @ np.sign(f) / num_samples

            if self.intercept:
              grad_b = np.mean(np.sign(f))


        elif self.loss == 'huber':
            grad_w = X.T @ np.where(np.abs(f) <= self.delta, f, self.delta * np.sign(f)) / num_samples

            if self.intercept:
              grad_b = np.mean(np.where(np.abs(f) <= self.delta, f, self.delta * np.sign(f)))


        grad_w_penalty = np.zeros_like(self.weights)

        if self.penalty == "l1":
            grad_w_penalty = self.alpha * np.sign(self.weights)

        elif self.penalty == "l2" and self.optimizer != "adamw":
            grad_w_penalty = 2 * self.alpha * self.weights

        elif self.penalty == "l2" and self.optimizer == "adamw":
            grad_w_penalty = np.zeros_like(self.weights)

        elif self.penalty == "elasticnet":
            l1 = self.l1_ratio * np.sign(self.weights)
            l2 = 2 * ((1 - self.l1_ratio) * self.weights)
            grad_w_penalty = self.alpha * (l2 + l1)

        elif self.penalty == 'asymmtric_bias':
            if len(self.loss_history) >= 2:
                if self.loss_history[-2] < self.loss_history[-1]:
                    grad_w_penalty = 2 * self.alpha * self.weights

                else:
                    grad_w_penalty = -(self.alpha * np.sign(self.weights))

        else:
           raise ValueError("Invalid penalty type. Choose from 'l1', 'l2', or 'elasticnet")

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
        if self.optimizer == 'adamw' and self.penalty in ['l1', 'elasticnet']:
          print(f"WARNING: Using L1 or ElasticNet penalty with AdamW optimizer may not be optimal. Consider using SGD or Adam without weight decay for these penalties.")

        if isinstance(X_train, spmatrix):
            X_processed = X_train

        elif isinstance(X_train, (np.ndarray, list, tuple)):
            X_processed = np.asarray(X_train)

        else:
            X_processed = X_train.to_numpy()

        if X_processed.ndim == 1:
            if issparse(X_processed):
                # For sparse matrices, reshape to 2D
                X_processed = X_processed.reshape(-1, 1)
            else:
                X_processed = X_processed.reshape(-1, 1)

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

        num_samples, num_features = X_processed.shape

        rng = np.random.default_rng(self.random_state)

        num_batch = int(np.ceil(num_samples / self.batch_size))
        rng_init = np.random.default_rng(self.random_state)

        if self.weights is None or self.weights.shape[0] != num_features:
          self.weights = rng_init.normal(loc=0.0, scale=0.01, size=num_features)
        if self.intercept:
          self.b = rng_init.normal(loc=0.0, scale=0.01)

        if self.optimizer in ['adam', 'adamw']:
            self.m_w = np.zeros_like(self.weights)
            self.v_w = np.zeros_like(self.weights)
            self.m_b = 0.0
            self.v_b = 0.0

        if isinstance(self.stoic_iter, str):
            if self.stoic_iter == 'sqrt':
                self.stoic_iter = max(1, int(sqrt(self.max_iter)))

            elif self.stoic_iter == 'log2':
                self.stoic_iter = max(1, int(log2(self.max_iter)))

            else:
                raise ValueError(f"Unexpected max_stoic_iter argument method {self.stoic_iter}")
            
        elif isinstance(self.stoic_iter, float):
            self.stoic_iter = max(1, int(self.stoic_iter * self.max_iter))

        elif isinstance(self.stoic_iter, int):
            self.stoic_iter = self.stoic_iter

        else:
           raise ValueError(f"Unexpected max_stoic_iter argument method {self.stoic_iter}")

        
        self.current_lr = self.eta0

        for i in range(self.max_iter):
            if self.ada_alpha and i > 1:
                self.alpha *= (self.loss_history[-1] / self.loss_history[-2])
                self.alpha_history.append(self.alpha)

            if self.lr_scheduler == 'constant':
                self.current_lr = self.eta0

            elif self.lr_scheduler == 'invscaling':
                self.current_lr = self.eta0 / ((i + 1)**self.power_t + self.epsilon)
                self.lr_rate_history.append(self.current_lr)

            elif self.lr_scheduler == 'plateau':
                    if i > 0:
                      current_loss = self._loss(X_processed, y_processed)
                      if current_loss < self.best_loss - self.epsilon:
                        self.best_loss = current_loss
                        self.wait = 0

                      elif abs(current_loss - self.best_loss) < self.tol:
                        self.wait += 1

                      else:
                        self.wait = 0

                    if self.wait >= self.patience:
                      self.current_lr *= self.factor
                      self.wait = 0
                      self.lr_rate_history.append(self.current_lr)
                      if self.verbose == 1 or self.verbose == 2:
                        print(f"Epoch {i + 1} reducing learning rate to {self.current_lr:.6f}")
            
            else:
                raise ValueError("Invalid learning rate scheduler")

            if self.shuffle:
                indices = rng.permutation(num_samples)
                X_shuffled = X_processed[indices]
                y_shuffled = y_processed[indices]

            else:
                X_shuffled = X_processed
                y_shuffled = y_processed

            for j in range(num_batch):
                s_idx = j * self.batch_size
                e_idx = min((j + 1) * self.batch_size, num_samples)

                X_batch = X_shuffled[s_idx:e_idx]
                y_batch = y_shuffled[s_idx:e_idx]
                try:
                  grad_w, grad_b = self._grad(X_batch, y_batch)

                except ValueError:
                    grad_w, grad_b = self._grad(X_batch, y_batch.ravel())

                if self.optimizer == 'mbgd':
                   self.weights -= self.current_lr * grad_w
                   if self.intercept:
                      self.b -= self.current_lr * grad_b

                elif self.optimizer == 'adam':
                    t = i * num_batch + j + 1

                    self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * grad_w
                    self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (grad_w**2)

                    m_w_hat = self.m_w / (1 - self.beta1**t)
                    v_w_hat = self.v_w / (1- self.beta2**t)

                    self.weights -= self.current_lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon_adam)
                    if self.intercept:
                        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * grad_b
                        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (grad_b**2)

                        m_b_hat = self.m_b / (1 - self.beta1**t)
                        v_b_hat = self.v_b / (1- self.beta2**t)

                        self.b -= self.current_lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon_adam)
                elif self.optimizer == 'adamw':
                        t = i * num_batch + j + 1

                        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * grad_w
                        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (grad_w**2)

                        m_w_hat = self.m_w / (1 - self.beta1**t)
                        v_w_hat = self.v_w / (1 - self.beta2**t)

                        self.weights -= self.current_lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon_adam)
                        
                        if self.penalty == 'l2':
                            self.weights -= self.current_lr * self.alpha * self.weights
                            if self.b_decay:
                              self.b -= self.current_lr * self.alpha * self.b

                        if self.intercept:
                            self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * grad_b
                            self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (grad_b**2)

                            m_b_hat = self.m_b / (1 - self.beta1**t)
                            v_b_hat = self.v_b / (1 - self.beta2**t)

                            self.b -= self.current_lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon_adam)

                else:
                    raise ValueError(f"Unexpected optimizer argument {self.optimizer}")
                
                if self.optimizer == 'adamw' and self.penalty in ['l1', 'elasticnet']:
                    if self.penalty == 'l1':
                        self.weights -= self.current_lr * self.alpha * np.sign(self.weights)
                    
                    elif self.penalty == 'elasticnet':
                        l1 = self.l1_ratio * np.sign(self.weights)
                        l2 = (1 - self.l1_ratio) * 2 * self.weights
                        self.weights -= self.current_lr * self.alpha * (l1 + l2)


            loss = self._loss(X_processed, y_processed)
            self.loss_history.append(loss)

            if not np.all(np.isfinite(self.weights)) or (self.intercept and not np.isfinite(self.b)):
                print(f"Warning: Weights or bias became NaN/Inf at epoch {i + 1}. Stopping training early.")
                break

            if np.any(np.isnan(self.weights)) or np.any(np.isinf(self.weights)) or np.isnan(self.b) or np.isinf(self.b):
                    raise ValueError(f"There's NaN in epoch {i + 1} during the training process")
            
            if self.verbose == 2:
                print(f"- Epoch {i + 1}. Loss: {loss:.4f}, Weights: {np.mean(self.weights)}, Bias: {self.b}, LR: {self.current_lr:.4f}, Alpha: {self.alpha}")
            
            if self.verbose == 1 and ((i % max(1, self.max_iter // 20)) == 0 or i < 5):
                print(f"- Epoch {i + 1}. Loss: {loss:.4f}, Weights: {np.mean(self.weights)}, Bias: {self.b}, LR: {self.current_lr:.4f}, Alpha: {self.alpha}")

            try:
              if self.early_stop and i > self.stoic_iter:
                if i > 0 and abs(self.loss_history[-1] - self.loss_history[-2]) < self.tol:
                  break
             
            except IndexError:
                None
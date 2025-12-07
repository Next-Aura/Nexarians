# ========== LIBRARIES ==========
import numpy as np                           # For numerical computations
from scipy.sparse import issparse, spmatrix  # For sparse data handling
import pandas as pd                          # For DataFrame data handling
from nexgml.metrics import r2_score          # For R2 score calculation
from nexgml.guardians import safe_array      # For numerical stability

# ========== THE MODEL ==========
class ElasticNetRegressor:
    """
    ElasticNet Regressor, combining L1 and L2 regularization for linear regression.
    Uses Coordinate Descent to optimize the model, balancing sparsity (L1) and smoothness (L2).
    
    ## Attrs:
      **weights**: *np.ndarray*
      An array that stored features weight with shape (n_feature,).

      **b**: *float*
      A float that is a bias model bias for flexibility output.

      **loss_history**: list[float,...]
      A list that stored loss from all iteration, loss_history is plot-able.

      **n_outputs_**: *int*
      Store number of classes, target, or output from y data.

    ## Methods:
      **_add_intercept(X)**: *Return np.ndarray*
      Add one column for intercept terms.

      **_soft_thresholding(rho, lam)**: *Return np.ndarray*
      Apply the soft-thresholding operation (proximal operator for L1).

      **fit(X_train, y_train)**: *Return None*
      Train model with inputed X_train and y_train argument data.

      **predict(X_test)**: *Return np.ndarray*
      Predict using weights from training session.

      **score(X_test, y_test)**: *Return float*
      Calculate model classification accuracy.

      **get_params(deep)**: *Return dict*
      Return model's parameter.

      **set_params([params])**: *Return model's class*
      Set model parameter.

    ## Notes:
      Model is fully implemented on python that may be easy to understand for beginners,
      but also may cause a big latency comparing to another libraries models.

    ## Usage Example:
    ```python
      >>> model = ElasticNetRegressor(l1_ratio=0.8)
      >>> model.fit(X_train, y_train)
      >>>
      >>> acc = model.score(X_test, y_test)
      >>> print("ElasticNetRegressor accuracy:", acc)
    ```
    """
    def __init__(self,
                 max_iter: int=100,
                 alpha: float=1e-4,
                 l1_ratio: float=0.5,
                 fit_intercept: bool=True,
                 tol: float=1e-4,
                 early_stopping: bool=True,
                 verbose: int=0,
                 stoic_iter: int=10) -> None:
        """
        Initialize the ElasticNetRegressor model.

        ## Args:
            **max_iter**: *int, default=100*
            Maximum number of iterations for Coordinate Descent.

            **alpha**: *float, default=1e-4*
            Overall regularization strength (L1 + L2).

            **l1_ratio**: *float, default=0.5*
            Mixing parameter between L1 and L2 regularizations.

            **fit_intercept**: *bool, default=True*
            If True, the model will calculate the intercept (bias) term.

            **tol**: *float, default=1e-4*
            Tolerance for convergence in Coordinate Descent.

            **early_stopping**: *bool, default=True*
            If true, will make the model end the training loop early if the model in plateau performance.

            **verbose**: *int, default=0*
            Verbosity level (0: no output, 1: some output, 2: detailed output).

            **stoic_iter**: *int, default=10*
            Number of initial epochs to skip before checking for convergence/tolerance in early stopping.

        ## Returns:
          **None**

        ## Raises:
          **None**
        """
        # =========== HYPERPARAMETERS ==========
        self.alpha = np.float32(alpha)             # Alpha for regularization power
        self.intercept = bool(fit_intercept)       # Fit intercept (bias) or not
        self.verbose = int(verbose)                # Model progress logging
        self.max_iter = int(max_iter)              # Model max training iterations
        self.tol = np.float32(tol)                 # Training loss tolerance
        self.l1_ratio = np.float32(l1_ratio)       # Elastic net mixing ratio
        self.early_stop = bool(early_stopping)     # Early stopping flag
        self.stoic_iter = int(stoic_iter)          # Warm-up iterations before applying early stopping

        self.weights = None                        # Model weights
        self.b = None                              # Model bias
        self.n_outputs_ = None                     # Number of y data outputs
        self.loss_history = []                     # Store residual history per epoch

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """
        Add a column of ones to X for the intercept term.

        ## Args:
            **X**: *np.ndarray*
            Input feature matrix.

        ## Returns:
            **np.ndarray**: *Augmented matrix with intercept column.*
        """
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.hstack([ones, X])

    def _soft_thresholding(self, rho: float, lam: float) -> float:
        """
        Apply soft thresholding for L1 regularization.

        ## Args:
            **rho**: *float*
            Correlation value.

            **lam**: *float*
            Threshold value (L1 regularization strength).

        ## Returns:
            **float**: *Soft-thresholded value.*
        """
        if rho < -lam:
            return rho + lam
        elif rho > lam:
            return rho - lam
        else:
            return 0.0

    def fit(self, X_train: np.ndarray | spmatrix | pd.DataFrame, y_train: np.ndarray | pd.Series) -> 'ElasticNetRegressor':
        """
        Fit the ElasticNet Regressor model to the training data using Coordinate Descent.

        ## Args:
            **X_train**: *np.ndarray, pd.DataFrame, or spmatrix*
            Training input features.

            **y_train**: *np.ndarray or pd.Series*
            Training target values.

        ## Returns:
            **ElasticNetRegressor**: *self.*
            The fitted model object.

        ## Raises:
            **ValueError**: *If input data contains NaN/Inf or if dimensions mismatch.*
            **OverflowError**: *If model parameters become infinity during training loop.*
            **RuntimeWarning**: *If overflow is detected and values are clipped.*
        """
        if issparse(X_train):
            if not np.all(np.isfinite(X_train.data)) or not np.all(np.isfinite(y_train)):
                raise ValueError(f"There's a NaN or infinity value between X_train and y_train data, please clean your data first.")
            
        else:
            if not np.all(np.isfinite(X_train)) or not np.all(np.isfinite(y_train)):
                raise ValueError(f"There's a NaN or infinity value between X_train and y_train data, please clean your data first.")

        if isinstance(X_train, pd.DataFrame):
            X = X_train.to_numpy(dtype=np.float32)

        if issparse(X_train):
            X = X_train.toarray()

        else:
            X = np.array(X_train, dtype=np.float32)

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array for X, got shape {X.shape}")

        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(y_train, dtype=np.float32)
        Y = Y.reshape(-1, 1)
        n_samples, n_features = X.shape
        n_samples_y, n_classes = Y.shape
        assert n_samples == n_samples_y, "Mismatched number of samples between X and Y"

        self.n_outputs_ = n_classes

        if self.intercept:
            X_aug = self._add_intercept(X)
        else:
            X_aug = X

        d = X_aug.shape[1]
        self.weights = np.zeros((d, n_classes), dtype=np.float32)

        for k in range(n_classes):
            y = Y[:, k]
            w = np.zeros(d, dtype=np.float32)
            for iteration in range(self.max_iter):
                w_old = w.copy()
                for j in range(d):
                    if self.intercept and j == 0:
                        # Update intercept without regularization
                        residual = y - (X_aug @ w) + w[j] * X_aug[:, j]
                        w[j] = np.sum(residual) / n_samples
                    else:
                        residual = y - (X_aug @ w) + w[j] * X_aug[:, j]
                        rho = np.dot(X_aug[:, j], residual)
                        z = np.sum(X_aug[:, j] ** 2)
                        lam1 = self.alpha * self.l1_ratio
                        lam2 = self.alpha * (1 - self.l1_ratio)
                        w_j = self._soft_thresholding(rho, lam1) / (z + lam2)
                        w[j] = w_j

                residual_mean = np.mean(residual)
                self.loss_history.append(residual_mean)

                if np.any(np.isnan(w)):
                    w = safe_array(w)

                if np.any(np.isinf(w)):
                    raise OverflowError("Model parameters became infinity during training.")

                # Level 1 verbose logging
                if self.verbose == 1 and ((iteration % max(1, self.max_iter // 20)) == 0 or iteration < 5):
                    print(f"Epoch {iteration + 1}/{self.max_iter}. Residual: {residual_mean:.6f}")

                # Level 2 verbose logging
                elif self.verbose == 2:
                    print(f"Epoch {iteration + 1}/{self.max_iter}. Residual: {residual_mean:.8f}")


                # ========== EARLY STOPPING ==========
                if self.early_stop and iteration > self.stoic_iter:
                    if abs(self.loss_history[-1] - self.loss_history[-2]) < self.tol:
                        break 
                    
                    if iteration > 2 * self.stoic_iter:
                        if abs(np.mean(self.loss_history[-self.stoic_iter:]) - np.mean(self.loss_history[-2*self.stoic_iter:-self.stoic_iter])) < self.tol:
                            break


            self.weights[:, k] = w

        if self.intercept:
            self.b = self.weights[0, :].reshape(1, -1)
            self.weights = self.weights[1:, :]
        else:
            self.b = np.zeros((1, n_classes), dtype=np.float32)

        return self

    def predict(self, X_train: np.ndarray | spmatrix | pd.DataFrame) -> np.ndarray:
        """
        Predict target values using the fitted ElasticNet Regressor model.

        ## Args:
            **X_train**: *np.ndarray, pd.DataFrame, or spmatrix*
            Input features for prediction.

        ## Returns:
            **np.ndarray**: *Predicted target values.*
            Shape is (n_samples,) for single output or (n_samples, n_outputs) for multi-output.

        ## Raises:
            **ValueError**: *If input data contains NaN/Inf or if dimensions mismatch.*
        """
        if issparse(X_train):
            if not np.all(np.isfinite(X_train.data)):
                raise ValueError(f"There's a NaN or infinity value between in X data, please clean your data first.")
            
        else:
            if not np.all(np.isfinite(X_train)):
                raise ValueError(f"There's a NaN or infinity value in X data, please clean your data first.")

        if isinstance(X_train, pd.DataFrame):
            X = X_train.to_numpy(dtype=np.float32)

        elif issparse(X_train):
            X = X_train.toarray().astype(np.float32)

        else:
            X = np.array(X_train, dtype=np.float32)

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array for X, got shape {X.shape}")
        
        X = np.asarray(X, dtype=np.float32)
        n_samples, n_features = X.shape
        assert self.weights is not None, "Model not fitted"
        assert n_features == self.weights.shape[0], f"Feature mismatch: got {n_features}, expected {self.coef_.shape[0]}"
        preds = X @ self.weights + self.b
        return preds.squeeze() if self.n_outputs_ == 1 else preds
    
    def score(self, X_test: np.ndarray | spmatrix, y_test: np.ndarray) -> float:
        """
        Calculate the coefficient of determination R^2 of the prediction.

        ## Args:
            **X_test**: *np.ndarray* or *spmatrix*
            Feature matrix.

            **y_test**: *np.ndarray*
            True target values.

        ## Returns:
            **float**: *R^2 score.*

        ## Raises:
            **None**
        """
        # ========== PREDICTION ==========
        y_pred = self.predict(X_test)
        
        # ========== R2 SCORE CALCULATION ==========
        return r2_score(y_test, y_pred)
    
    def get_params(self, deep=True) -> dict[str, object]:
        """
        Returns model paramters.

        ## Args:
            **deep**: *bool, default=True*
            If True, will return the parameters for this estimator and contained subobjects that are estimators.

        ## Returns:
            **dict**: *Model parameters.*

        ## Raises:
            **None**
        """
        return {
            "max_iter": self.max_iter,
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
            "fit_intercept": self.intercept,
            "tol": self.tol,
            "early_stopping": self.early_stop,
            "verbose": self.verbose
        }

    def set_params(self, **params) -> 'ElasticNetRegressor':
        """
        Returns model's attribute that ready to set.

        ## Args:
            **params**: *dict*
            Model parameters to set.

        ## Returns:
            **ElasticNetRegressor**: *The model instance with updated parameters.*

        ## Raises:
            **None**
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
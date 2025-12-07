# ========== LIBRARIES ==========
import numpy as np                           # For numerical computations
from scipy.sparse import issparse, spmatrix  # For sparse data handling
import pandas as pd                          # For DataFrame data handling
from nexgml.metrics import r2_score          # For R2 score calculation

# ========== THE MODEL ==========
class L2Regressor:
    """
    L2 Regressor, also known as Ridge Regression, is a linear regression model that uses 
    L2 regularization (Tikhonov regularization) to prevent overfitting and handle 
    multicollinearity in the data. It finds the optimal weights using the closed-form 
    solution (Normal Equation) with a penalty term.

    ## Attrs:
      **weights**: *np.ndarray*
      An array that stored features weight with shape (n_feature,).

      **b**: *float*
      A float that is a bias model bias for flexibility output.

    ## Methods:
      **_add_intercept(X)**: *Return np.ndarray*
      Add one column for intercept terms.

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
        >>> model = L2Regressor(alpha=0.0005)
        >>> model.fit(X_train, y_train)
        >>>
        >>> acc = model.score(X_test, y_test)
        >>> print("L2Regressor accuracy:", acc)
    ```
    """
    
    def __init__(self, 
                 alpha: float=1e-4, 
                 fit_intercept: bool=True) -> None:
        """
        Initialize the L2Regressor model.
        
        ## Args:
            **alpha**: *float, default=1e-4*
            Regularization strength (used if penalty is not None).

            **fit_intercept**: *bool, default=True*
            If True, the model will calculate the intercept (bias) term.
            
        ## Returns:
          **None**

        ## Raises:
          **None**
        """
        # ========== HYPERPARAMETERS ==========
        self.alpha = np.float32(alpha)                          # Alpha for regularization power
        self.intercept = bool(fit_intercept)                    # Fit intercept (bias) or not
    
        self.weights = None                                     # Model weights
        self.b = None                                           # Model bias

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """
        Prepends a column of ones to the feature matrix X to account for the intercept term.
        
        ## Args:
            **X**: *np.ndarray*
            The input feature matrix.
            
        ## Returns:
            **np.ndarray**: *Augmented feature matrix.*

        ## Raises:
          **None**
        """
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.hstack([ones, X])

    def fit(self, X_train: np.ndarray | pd.DataFrame | spmatrix, y_train: np.ndarray | pd.Series) -> 'L2Regressor':
        """
        Fit the L2 Regressor model to the training data using the Normal Equation 
        (closed-form solution).

        ## Args:
            **X_train**: *np.ndarray, pd.DataFrame, or spmatrix*
            Training input features.

            **y_train**: *np.ndarray or pd.Series*
            Training target values.
            
        ## Returns:
            **L2Regressor**: *self.*
            The fitted model object.
            
        ## Raises:
            **ValueError**: *If input data contains NaN/Inf or if dimensions mismatch.*
            **np.linalg.LinAlgError**: *If $X^T X + \lambda I$ is singular (rare in Ridge Regression).*
        """
        # Data validation and conversion
        if issparse(X_train):
            if not np.all(np.isfinite(X_train.data)) or not np.all(np.isfinite(y_train)):
                raise ValueError(f"There's a NaN or infinity value between X and Y data, please clean your data first.")
            
        else:
            if not np.all(np.isfinite(X_train)) or not np.all(np.isfinite(y_train)):
                raise ValueError(f"There's a NaN or infinity value between X and Y data, please clean your data first.")
        
        if isinstance(X_train, pd.DataFrame):
            X = X_train.to_numpy(dtype=np.float32)

        elif issparse(X_train):
            X = X_train.toarray().astype(np.float32)

        else:
            X = np.array(X_train, dtype=np.float32)

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array for X, got shape {X.shape}")

        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(y_train, dtype=np.float32)
        Y = Y.reshape(-1, 1)
        n_samples, n_features = X.shape
        n_samples_y, n_classes = Y.shape
        assert n_samples == n_samples_y, "Number of samples in X and Y must match. Error"

        # Augment X for intercept
        if self.intercept:
            X_aug = self._add_intercept(X) 
        else:
            X_aug = X

        # Normal Equation (Ridge Regression: W = (X^T X + alpha*I)^-1 X^T Y)
        XtX = X_aug.T @ X_aug
        d = XtX.shape[0]
        reg = np.eye(d, dtype=np.float32) * self.alpha
        
        # Do not regularize the intercept term (first element)
        if self.intercept:
            reg[0, 0] = np.float32(0.0)
            
        A = XtX + reg
        Xty = X_aug.T @ Y

        try:
            # Prefer np.linalg.solve for speed and stability
            W = np.linalg.solve(A, Xty)

        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if A is singular or ill-conditioned
            W = np.linalg.pinv(A) @ Xty

        # Separate weights and bias
        if self.intercept:
            self.b = W[0, :].reshape(1, -1)
            self.weights = W[1:, :]
            
        else:
            self.b = np.zeros((1, n_classes), dtype=np.float32)
            self.weights = W

        return self

    def predict(self, X_test: np.ndarray | pd.DataFrame | spmatrix) -> np.ndarray:
        """
        Predict the target values for the given input features using the trained model.

        ## Args:
            **X_test**: *np.ndarray, pd.DataFrame, or spmatrix*
            Input features for prediction.

        ## Returns:
            **np.ndarray**: *Predicted target values.*

        ## Raises:
            **ValueError**: *If model weights are not defined (model not trained) or data contains NaN/Inf.*
        """
        # Data validation and conversion
        if issparse(X_test):
            if not np.all(np.isfinite(X_test.data)):
                raise ValueError(f"There's a NaN or infinity value between in X data, please clean your data first.")
            
        else:
            if not np.all(np.isfinite(X_test)):
                raise ValueError(f"There's a NaN or infinity value in X data, please clean your data first.")

        if isinstance(X_test, pd.DataFrame):
            X = X_test.to_numpy(dtype=np.float32)

        elif issparse(X_test):
            X = X_test.toarray().astype(np.float32)

        else:
            X = np.array(X_test, dtype=np.float32)

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array for X, got shape {X.shape}")


        X = np.asarray(X, dtype=np.float32)
        n_samples, n_features = X.shape
        
        # Check if the model has been fitted
        assert self.weights is not None, "Model not fitted, please call fit() first."
        assert n_features == self.weights.shape[0], f"Feature mismatch: got {n_features}, expected {self.weights.shape[0]}"
        
        # Linear combination: $\hat{y} = XW + b$
        preds = X @ self.weights + self.b
        return preds
    
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
            "alpha": self.alpha,
            "fit_intercept": self.intercept
        }

    def set_params(self, **params) -> 'L2Regressor':
        """
        Returns model's attribute that ready to set.

        ## Args:
            **params**: *dict*
            Model parameters to set.

        ## Returns:
            **L2Regressor**: *The model instance with updated parameters.*

        ## Raises:
            **None**
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
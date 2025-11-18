# ========== LIBRARIES ==========
import numpy as np                           # For numerical computations
from scipy.sparse import issparse, spmatrix  # For sparse data handling
import pandas as pd                          # For DataFrame data handling
from nexgml.indexing import one_hot_labeling # For encoding utility

# ========== THE MODEL ==========
class L2Classifier:
    """
    L2 (Ridge) Classifier is a linear classifier that minimizes a loss function with L2 regularization.
    It uses a closed-form solution (Normal Equation) rather than iterative gradient descent.
    This model is effectively a multi-class logistic regression solved via least squares.
    """
    def __init__(self, 
                 alpha: float=1e-4, 
                 fit_intercept: bool=True):
        """
        Initialize the L2Classifier model.
        
        ## Args:
            **alpha**: *float, default=1e-4*
            Regularization strength (lambda in the L2 regularization term). Must be non-negative.

            **fit_intercept**: *bool, default=True*
            If True, include a bias term (intercept) in the model.

        ## Returns:
            **None**

        ## Raises:
            **None**
        """
        # ========== HYPERPARAMETERS ===========
        self.alpha = float(alpha)                  # Alpha for regularization power
        self.intercept = bool(fit_intercept)       # Fit intercept (bias) or not
        
        self.weights = None                        # Model weights
        self.b = None                              # Model bias
        self.classes = None                        # Array of unique class labels from training data
        self.n_classes = None                      # Number of unique classes (determined during fit)

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """
        Helper function to add a column of ones for the intercept term.
        
        ## Args:
            **X**: *np.ndarray*
            Input features array.

        ## Returns:
            **np.ndarray**: *Augmented array with an intercept column.*

        ## Raises:
            **None**
        """
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.hstack([ones, X])

    def fit(self, X_train: np.ndarray | pd.DataFrame | spmatrix, y_train: np.ndarray | pd.DataFrame) -> 'L2Classifier':
        """
        Fit the model to the training data using the closed-form Normal Equation with L2 regularization.
        
        ## Args:
            **X_train**: *np.ndarray, pd.DataFrame, or sparse matrix*
            Training input features.

            **y_train**: *np.ndarray or pd.DataFrame*
            Training target values (class labels).
            
        ## Returns:
            **L2Classifier**: *The fitted instance of the model.*
            
        ## Raises:
            **ValueError**: *If input data contains NaN/Inf, if X is not 2D, or if dimensions mismatch.*
        """
        # ========== Data Validation and Preprocessing ==========
        if issparse(X_train):
            if not np.all(np.isfinite(X_train.data)) or not np.all(np.isfinite(y_train)):
                raise ValueError(f"There's a NaN or infinity value between X and Y data, please clean your data first.")
            
        else:
            if not np.all(np.isfinite(X_train)) or not np.all(np.isfinite(y_train)):
                raise ValueError(f"There's a NaN or infinity value between X and Y data, please clean your data first.")

        if isinstance(X_train, pd.DataFrame):
            X = X_train.to_numpy(dtype=np.float64)

        elif issparse(X_train):
            X = X_train.toarray().astype(np.float64)

        else:
            X = np.array(X_train, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array for X, got shape {X.shape}")
        
        X = np.asarray(X, dtype=np.float64)
        
        # ========== Label Processing ==========
        classes = np.unique(y_train)
        self.classes = classes
        self.n_classes = len(classes)
        y_one_hot = one_hot_labeling(y_train, classes)
        Y = np.asarray(y_one_hot, dtype=np.float64)
        
        n_samples, n_features = X.shape
        n_samples_y, n_classes = Y.shape
        assert n_samples == n_samples_y, "Error"

        # ========== Model Fitting (Closed-form Solution) ==========
        if self.intercept: # Changed from self.fit_intercept
            X_aug = self._add_intercept(X) 
        else:
            X_aug = X

        XtX = X_aug.T @ X_aug
        d = XtX.shape[0]
        reg = np.eye(d) * self.alpha
        
        if self.intercept: # Changed from self.fit_intercept
            reg[0, 0] = 0.0 # Do not regularize the intercept term
            
        A = XtX + reg
        Xty = X_aug.T @ Y

        try:
            W = np.linalg.solve(A, Xty)

        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if linear system is singular
            W = np.linalg.pinv(A) @ Xty

        if self.intercept: # Changed from self.fit_intercept
            self.b = W[0, :].reshape(1, -1)     # Changed from self.intercept_
            self.weights = W[1:, :]             # Changed from self.coef_
        else:
            self.b = np.zeros((1, n_classes), dtype=np.float64) # Changed from self.intercept_
            self.weights = W                    # Changed from self.coef_

        return self

    def predict(self, X_test: np.ndarray | pd.DataFrame | spmatrix) -> np.ndarray:
        """
        Predict class labels using the trained model.

        ## Args:
            **X_test**: *np.ndarray, pd.DataFrame, or spmatrix*
            Input features for prediction.

        ## Returns:
            **np.ndarray**: *Predicted class labels.*

        ## Raises:
            **ValueError**: *If the model is not fitted or if the number of features mismatches.*
        """
        # ========== Data Validation and Preprocessing ==========
        if issparse(X_test):
            if not np.all(np.isfinite(X_test.data)):
                raise ValueError(f"There's a NaN or infinity value between in X data, please clean your data first.")
            
        else:
            if not np.all(np.isfinite(X_test)):
                raise ValueError(f"There's a NaN or infinity value in X data, please clean your data first.")

        if isinstance(X_test, pd.DataFrame):
            X = X_test.to_numpy(dtype=np.float64)

        elif issparse(X_test):
            X = X_test.toarray().astype(np.float64)

        else:
            X = np.array(X_test, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array for X, got shape {X.shape}")
        
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        
        # ========== Prediction ==========
        assert self.weights is not None, "Model not fitted" # Changed from self.coef_
        assert n_features == self.weights.shape[0], f"Feature mismatch: got {n_features}, expected {self.weights.shape[0]}" # Changed from self.coef_
        
        # Calculate raw scores (logits)
        preds = X @ self.weights + self.b # Changed from self.coef_ and self.intercept_
        
        # Choose class with highest score
        pred_class = np.argmax(preds, axis=1)
        
        # Map indices to original classes
        if self.classes is not None and len(self.classes) == self.n_classes:
            pred_class = np.array([self.classes[idx] for idx in pred_class])
            
        return pred_class
    
    def score(self, X_test: np.ndarray | spmatrix, y_test: np.ndarray) -> float:
        """
        Calculate the mean accuracy on the given test data and labels.

        ## Args:
            **X_test**: *np.ndarray* or *spmatrix*
            Feature matrix.

            **y_test**: *np.ndarray*
            True target labels.

        ## Returns:
            **float**: *Mean accuracy score.*

        ## Raises:
            **None**
        """
        # ========== PREDICTION ==========
        y_pred = self.predict(X_test)
        # Compare prediction with true labels and compute mean
        return np.mean(y_pred == y_test)

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

    def set_params(self, **params) -> 'L2Classifier':
        """
        Returns model's attribute that ready to set.

        ## Args:
            **params**: *dict*
            Model parameters to set.

        ## Returns:
            **L2Classifier**: *The model instance with updated parameters.*

        ## Raises:
            **None**
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
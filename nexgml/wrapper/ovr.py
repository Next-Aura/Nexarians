# ========== LIBRARIES ==========
import numpy as np                           # For numerical computations
from scipy.sparse import issparse, spmatrix  # For sparse matrix support
from typing import Optional                  # More specific type hints
from ..modelkit.nexg import ModelLike        # For checking if the model is support OneVsRestWith class

# ========== THE MODEL ==========
class OneVsRestWith:
    """
    OneVsRestWith is a linear classifier wrapper that uses 
    models probability computational 
    binary and multi-class classification (OvR).

    Handles both dense and sparse input matrices.
    """
    def __init__(
        self,
        model,
        repetitions: int=1,
        random_state: Optional[int] | None=None,
        verbose: int=0,
        ) -> None:
        """
        Initialize the OneVsRestWith wrapper.

        ## Args:
            **model**: *ModelLike*
            Model that'll be trained by OvR method.

            **repetitions**: *int, default=1*
            Max OvR repetitions.

            **random_state**: *float, default=None*
            Seed for random number generator for reproducibility.

            **verbose**: *int, default=0*
            If 1, print training progress (epoch, loss, etc.).
            
        ## Returns:
            **None**

        ## Raises:
            **ValueError: If inputed model has an issue like non-classifier model or non-compatible model**
        """
        
        # ========== HYPERPARAMETERS ==========
        self.model = model if ModelLike(model).is_classifier() else None  # Model that'll be trained with OvR method
        self.reps = int(repetitions)               # Max iter for OvR
        self.random_state = random_state           # Random state for reproducibility
        self.verbose = int(verbose)                # Verbose level for logging
        
        # ========== INTERNAL VARIABLES ==========
        self.epsilon = 1e-15                       # Small value to avoid division by zero
        self.weights = None                        # Model weights
        self.b = 0.0                               # Bias term
        self.loss_history= []                      # Loss history for each epoch
        self.current_lr = None                     # Current learning rate per epoch

        # ---------- Random state setup and model compability checking---------
        if self.random_state is not None:
            np.random.seed(self.random_state)

        if self.model is None:
            raise ValueError(f"Invalid model argument, {model}. Make sure your model is classifier and valid")

        self.wait = 0                          # Wait counter for plateau scheduler/early stopping
        self.best_loss = float('inf')          # Best loss for plateau scheduler
        
        self.classes = None                    # Unique class labels
        self.n_classes = 0                     # Number of unique classes
        self.binary_classifiers = []           # List of binary classifiers for OvR strategy
    
    
    def predict_proba(self, X_test: np.ndarray | spmatrix) -> np.ndarray:
        """
        Predict class probabilities for input samples.

        ## Args:
            **X_test**: *np.ndarray* or *spmatrix*
            Input features for prediction.

        ## Returns:
            **np.ndarray**: *Predicted class probabilities. For binary: (n_samples,). For multi-class: (n_samples, n_classes).*

        ## Raises:
            **ValueError**: *If model is not trained or weights are uninitialized.*
        """
        # --- Pre-processing Input X_test ---
        # Check if input is sparse
        if not issparse(X_test):
            # If 1D array, reshape to 2D
            if X_test.ndim == 1:
                X_processed = X_test.reshape(-1, 1)
            # Or if already 2D, use as is
            else:
                X_processed = X_test
            # Convert to numpy array
            X_processed = np.asarray(X_processed)
        else:
            # Or if sparse, use as is
            X_processed = X_test
        # Check if the model already trained
        if self.n_classes == 0:
             raise ValueError("Model not trained. Call fit() first.")
        
        # --- PREDICT PROBA LOGIC ---
        if self.n_classes == 2:
            # Binary classification
            if not self.binary_classifiers:
                raise ValueError("Binary classifier not trained. Call fit() first.")
            clf = self.binary_classifiers[0]
            class_proba = clf.predict_proba(X_processed)
            # Return probability for the positive class
            return class_proba[:, 1] if class_proba.ndim > 1 else class_proba

        # Multi-class classification
        elif self.n_classes > 2:
            # === MULTI-CLASS WITH ONE-VS-REST METHOD (OvR) ===
            # Check if binary classifiers are trained
            if not self.binary_classifiers:
                raise ValueError("OvR classifiers not trained. Call fit() first for multi-class data.")

            # Initialize probability array
            all_probas = np.zeros((X_processed.shape[0], self.n_classes))
            # For each class, average probabilities across repetitions
            for class_idx in range(self.n_classes):
                probas_for_class = []
                for rep in range(self.reps):
                    clf_idx = rep * self.n_classes + class_idx
                    clf = self.binary_classifiers[clf_idx]
                    # Get probabilities from binary classifier
                    class_proba = clf.predict_proba(X_processed)
                    # Extract positive class probability
                    proba = class_proba[:, 1] if class_proba.ndim > 1 else class_proba
                    probas_for_class.append(proba)
                # Average probabilities across repetitions
                all_probas[:, class_idx] = np.mean(probas_for_class, axis=0)

            # Return probabilities for all classes (no normalization for OvR)
            return all_probas
    
    # ========== MAIN METHODS ==========
    def fit(self, X_train: np.ndarray | spmatrix, y_train: np.ndarray | spmatrix) -> None:
        """
        Fit the model to the training data using the specified optimizer.
        For binary classification, trains a single model.
        For multi-class, trains one binary classifier per class using One-vs-Rest (OvR).

        ## Args:
            **X_train**: *np.ndarray* or *spmatrix*
            Training input features.

            **y_train**: *np.ndarray* or *spmatrix*
            Training target values.

        ## Returns:
            **None**

        ## Raises:
            **ValueError**: *If input data contains NaN/Inf, dimensions mismatch, or < 2 classes.*
        """
        # --- Pre-processing Input X_train and y_train ---
        # Check if input is sparse
        if not issparse(X_train):
            # If 1D array, reshape to 2D
            if X_train.ndim == 1:
                X_processed = X_train.reshape(-1, 1)
            # Or if already 2D, use as is
            else:
                X_processed = X_train
            # Convert to numpy array
            X_processed = np.asarray(X_processed)
        # If sparse, use as is
        else:
            X_processed = X_train.tocsr()

        # Flatten y data
        y_processed = np.asarray(y_train).ravel()

        # Check if sparse data input only has finite values
        if issparse(X_processed):
            if not np.all(np.isfinite(X_processed.data)):
                raise ValueError("Input features (X_train) contains NaN or Infinity values in its data. Please clean your data.")
        # Check if dense data input only has finite values
        else:
            if not np.all(np.isfinite(X_processed)):
                raise ValueError("Input features (X_train) contains NaN or Infinity values. Please clean your data.")

        # Check for y data this time
        if not np.all(np.isfinite(y_processed)):
            raise ValueError("Input target (y_train) contains NaN or Infinity values. Please clean your data.")


        # Check if number of samples match between X and y
        if X_processed.shape[0] != y_processed.shape[0]:
            raise ValueError(
                f"Number of samples in X_train ({X_processed.shape[0]}) "
                f"must match number of samples in y_train ({y_processed.shape[0]})."
            )


        # Get classes from y data
        self.classes = np.unique(y_processed)
        # Get number of classes
        self.n_classes = len(self.classes)

        # Raise an error if class is less than 2
        if self.n_classes < 2:
           raise ValueError("Class label must have at least 2 types.")

        # --- Binary classification ---
        if self.n_classes == 2:
            # Reset binary classifiers list
            self.binary_classifiers = []
            if self.verbose:
                print(f"  Training binary classifier for class {self.classes[1]} vs {self.classes[0]}...")

            # Create binary labels for positive class vs negative
            y_ovr = (y_processed == self.classes[1]).astype(int)

            # Instantiate a new classifier for the binary problem
            clf = self.model

            # Fit the binary classifier
            clf.fit(X_processed, y_ovr)
            # Store the trained binary classifier
            self.binary_classifiers.append(clf)

            # Set loss history
            self.loss_history = clf.loss_history if hasattr(clf, 'loss_history') else []

        # --- Multi-class classification (OvR) ---
        elif self.n_classes > 2:
            # Reset binary classifiers list
            self.binary_classifiers = []

            # Aggregate loss for monitoring
            avg_loss_history = []

            # Train a binary classifier for each class
            for i in range(self.reps):
              for class_label in self.classes:
                if self.verbose:
                    print(f"[{i + 1}] Training classifier for class {class_label} vs all others...")

                # Create binary labels for current class vs rest
                y_ovr = (y_processed == class_label).astype(int)

                # Instantiate a new IntenseClassifier for the binary problem
                clf = self.model() if callable(self.model) else self.model.__class__()

                # Fit the binary classifier
                clf.fit(X_processed, y_ovr)
                # Store the trained binary classifier
                self.binary_classifiers.append(clf)

            # Aggregate loss from this classifier
                if hasattr(clf, 'loss_history'):
                  avg_loss_history.append(clf.loss_history)

            # Average loss across all OvR classifiers for monitoring
            if avg_loss_history:
                self.loss_history = [np.mean([lh[i] for lh in avg_loss_history if i < len(lh)])
                                   for i in range(max(len(lh) for lh in avg_loss_history))]

      
    def predict(self, X_test: np.ndarray | spmatrix) -> np.ndarray:
        """
        Predict class labels for input samples.

        ## Args:
            **X_test**: *np.ndarray* or *spmatrix*
            Input features for prediction.

        ## Returns:
            **np.ndarray**: *Predicted class labels.*

        ## Raises:
            **ValueError**: *If model is not trained or weights are uninitialized.*
        """
        # Get classes probabilities
        probas = self.predict_proba(X_test)
        # Binary classification
        if self.n_classes == 2:
            pred_class = (probas >= 0.5).astype(int)
            # Map to original class labels
            pred_class = np.where(pred_class == 1, self.classes[1], self.classes[0])
        # Multi-class classification
        elif self.n_classes > 2:
            pred_class = np.argmax(probas, axis=1)
            # Map indices back to original class labels
            if self.classes is not None and len(self.classes) == self.n_classes:
                pred_class = np.array([self.classes[idx] for idx in pred_class])
        # Raise an error if class is less than 2
        else:
            raise ValueError("Model is not trained or invalid number of class.")

        # Return the predicted class
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
            **dict**: *Wrapper parameters.*

        ## Raises:
            **None**
        """
        return {
            "repetitions": self.reps,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }

    def set_params(self, **params) -> 'OneVsRestWith':
        """
        Returns wrapper's attribute that ready to set.

        ## Args:
            **params**: *dict*
            Model parameters to set.

        ## Returns:
            **OneVsRestWith**: *The wrapper instance with updated parameters.*

        ## Raises:
            **None**
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
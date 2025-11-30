# ========== LIBRARIES ==========
import numpy as np                            # Numpy for numerical computations
from typing import Literal, Optional          # More specific type hints
from scipy.sparse import issparse, spmatrix   # For sparse matrix handling
from nexgml.indexing import standard_indexing # For indexing utilities
from nexgml.amo.fortree import squared_error, absolute_error, friedman_squared_error, poisson_deviance # For some math operations
from nexgml.metrics import r2_score           # For R2 score calculation

# ========== THE MODEL ==========
class TreeBackendRegressor:
    """
    TreeBackendRegressor (TBR) is a simple decision tree model for regression tasks. 
    It builds a tree by recursively splitting nodes to minimize a given criterion and supports various pruning/stopping parameters to prevent overfitting.
    
    ## Attrs:
      **tree**: *dict*
      Model's tree structure.

    ## Methods: 
      **compute_variance_sparse(X)**: *Return np.ndarray*
      Compute the variance of each feature column in a sparse matrix.

      **_impurity(y)**: *Return float*
      Compute label variance (impurity).

      **criterion_score(y, left_y, right_y)**: *Return float*
      Calculate the impurity decrease (gain) from splitting.

      **find_best_split(X, y, feature_idx)**: *Returns float, float or None*
      Find the best split value for a given feature by evaluating potential split points and selecting the one that maximizes the impurity gain.
    
      **fit(X_train, y_train, depth)**: *Return dict*
      Train model with inputed X_train and y_train argument data using recursive method.

      **_predict_single(x, tree)**: *Return int*
      Predict the class label for a single sample by traversing the decision tree.

      **predict(X_test)**: *Return np.ndarray*
      Predict using tree structure from training session.

      **score(X_test)**: *Return float*
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
      >>> model = TreeBackendRegressor(max_depth=8)
      >>> model.fit(X_train, y_train)
      >>>
      >>> acc = model.score(X_test)
      >>> print("TreeBackendRegressor accuracy:", acc)
    ```
    """
    def __init__(
            self,
            max_depth: Optional[int] = 5,
            min_samples_leaf: Optional[int] = 1,
            criterion: Literal['squared_error', 'friedman_mse', 'absolute_error', 'poisson'] = 'squared_error',
            max_features: Optional[Literal['sqrt', 'log2']] | int | float | None=None,
            max_samples: Optional[Literal['sqrt', 'log2']] | int | float | None=None,
            random_state: Optional[int] = None,
            min_samples_split: Optional[int] = 2,
            min_impurity_decrease: Optional[float] = 0.0
            ) -> None:
        """
        Initialize the TreeBackendRegressor model.

        ## Args:
            **max_depth**: *int, default=5*
            Maximum depth of the tree.

            **criterion**: *{'squared_error', 'friedman_mse', 'absolute_error', 'poisson'}, default='mse'*
            The function to measure the quality of a split.

            **min_impurity_decrease**: *float, default=0.0*
            Tolerance for splitting. A node will be split if this split induces a decrease of the impurity greater than or equal to this value.

            **max_features**: *{'sqrt', 'log2'} or int or float or None, default=None*
            The number of features to consider when looking for the best split.

            **max_samples**: *{'sqrt', 'log2'} or int or float or None, default=None*
            The number of samples to draw from X to train the tree.

            **random_state**: *int or None, default=None*
            Seed for random number generator for reproducibility.

            **min_samples_leaf**: *int, default=1*
            The minimum number of samples required to be at a leaf node.

            **min_samples_split**: *int, default=2*
            The minimum number of samples required to split an internal node.

        ## Returns:
            **None**

        ## Raises:
            **ValueError**: *If invalid criterion is provided or if min_samples_split < 2 * min_samples_leaf.*
        """
        # ========== PARAMETER VALIDATIONS ==========
        if criterion not in ('squared_error', 'friedman_mse', 'absolute_error', 'poisson'):
            raise ValueError(f"Invalid criterion argument, {criterion}. Choose from 'squared_error', 'friedman_mse', 'absolute_error' or 'poisson'.")

        if max_depth is None or max_depth <= 0:
            raise ValueError(f"Invalid max_depth argument, {max_depth}. max_depth should be a positive integer.")

        if min_samples_leaf is None or min_samples_leaf <= 0:
            raise ValueError(f"Invalid min_samples_leaf argument, {min_samples_leaf}. min_samples_leaf should be a positive integer.")

        if min_samples_split is None or min_samples_split <= 0:
            raise ValueError(f"Invalid min_samples_split argument, {min_samples_split}. min_samples_split should be a positive integer.")

        if min_impurity_decrease is None or min_impurity_decrease < 0:
            raise ValueError(f"Invalid min_impurity_decrease argument, {min_impurity_decrease}. min_impurity_decrease should be a non-negative float.")

        if 2 * min_samples_leaf < min_samples_split:
            raise ValueError(f"Invalid min_samples_leaf and min_samples_split argument, {min_samples_leaf} | {min_samples_split}. min_samples_split must be at least 2 * min_samples_leaf")

        # ========== HYPERPARAMETERS ==========
        self.max_depth = int(max_depth)                               # Tree's max depth
        self.min_samples_leaf = int(min_samples_leaf)                 # Minimum samples per-leaf
        self.criterion = criterion                                    # Impurity calculate method
        self.max_features = max_features                              # Max features per-split
        self.max_samples = max_samples                                # Max samples per-split
        self.random_state = random_state                              # For reproductivity
        self.min_samples_split = int(min_samples_split)               # Minimum samples before split
        self.min_impurity_decrease = float(min_impurity_decrease)     # Minimum impurity decrease per-split
        
        self.tree = None                                              # Model tree structure
        
        # ---------- Random state setup ----------
        if self.random_state is not None:
            np.random.seed(self.random_state)

    # ========== HELPER METHODS ==========
    def _impurity(self, y: np.ndarray) -> float:
        """
        Calculate the impurity of the given labels based on the selected criterion.

        ## Args:
            **y**: *np.ndarray*
            Array of target values.

        ## Returns:
            **float**: *Impurity value according to the criterion.*

        ## Raises:
            **ValueError**: *If the criterion is unknown.*
        """
        # Compute varinace with squared error
        if self.criterion == 'squared_error':
            return squared_error(y)
        
        # Compute varinace with friedman MSE
        elif self.criterion == 'friedman_mse':
            return friedman_squared_error(y)
        
        # Compute varinace with absolute error
        elif self.criterion == 'absolute_error':
            return absolute_error(y)
        
        # Compute varinace with poisson deviance
        elif self.criterion == 'poisson':
            return poisson_deviance(y)

    def criterion_score(self, y: np.ndarray, left_y: np.ndarray, right_y: np.ndarray) -> float:
        """
        Calculate the impurity decrease (gain) from splitting.

        ## Args:
            **y**: *np.ndarray*
            Parent node target values.

            **left_y**: *np.ndarray*
            Left child target values.

            **right_y**: *np.ndarray*
            Right child target values.

        ## Returns:
            **float**: *Impurity decrease (gain) from the split.*
            
        ## Raises:
            **None**
        """
        # Get label size
        n = y.size
        # If label size is 0 then return 0.0
        if n == 0:
            return 0.0
        
        # Get label impurity
        parent_imp = self._impurity(y)
        # Left child target values size
        n_left = left_y.size
        # Left child target values size
        n_right = right_y.size

        # Get left child impurity, if it's 0 then the value is 0.0
        left_imp = self._impurity(left_y) if n_left > 0 else 0.0
        # Get right child impurity, if it's 0 then the value is 0.0
        right_imp = self._impurity(right_y) if n_right > 0 else 0.0
        # All children impurity calculation
        children_imp = (n_left * left_imp + n_right * right_imp) / n
        # Calculate impurity gain
        gain = parent_imp - children_imp
        # Return impurity
        return gain

    def find_best_split(self, X: np.ndarray | spmatrix, y: np.ndarray, feature_idx: int) -> tuple[float, float | None]:
        """
        Find the best split value for a given feature by evaluating potential split points and selecting the one that maximizes the impurity gain.

        ## Args:
            **X**: *np.ndarray* or *spmatrix*
            Feature matrix containing the input features for all samples.

            **y**: *np.ndarray*
            Target values corresponding to the samples in X.

            **feature_idx**: *int*
            Index of the feature column to evaluate for the best split.

        ## Returns:
            **tuple**: *(best_gain, best_value)* where best_gain is the impurity gain from the split and best_value is the threshold value for the split, or *(None, None)* if no valid split is found.

        ## Raises:
            **None**
        """
        # Check if data is sparse
        if issparse(X):
            # Slice with feature index and ransform to array then flatten
            feature_values = X[:, feature_idx].toarray().ravel()
        
        # If not sparse then just slice with feature index
        else:
            feature_values = X[:, feature_idx]
        
        # Get unique value from sliced data
        unique_values = np.unique(feature_values)
        # Get total unique value, if same or less than 0 then return tuple (None, None)
        if len(unique_values) <= 1:
            return None, None
        
        # If size of label is less than min_samples_split then return tuple (None, None)
        if y.size < self.min_samples_split:
            return None, None
        
        # Check if total unique values is less than 100
        # If many unique values, use percentiles for efficiency
        if len(unique_values) > 100:
            # Fewer points for sparse matrices to reduce computation
            num_points = 51 if issparse(X) else 101
            # Compute percentiles as split candidates
            split_points = np.percentile(feature_values, np.linspace(0, 100, num_points))
            # Remove duplicates from percentiles
            split_points = np.unique(split_points)
        
        # If few unique values, use midpoints between consecutive uniques
        else:
            # Calculate midpoints for split points
            split_points = (unique_values[:-1] + unique_values[1:]) / 2.0
            
        best_gain = -np.inf
        best_value = None
        
        # Iterate over each potential split point
        for value in split_points:
            # Create boolean mask for samples going to left child (feature <= threshold)
            left_mask = feature_values <= value
            # Create boolean mask for samples going to right child (feature > threshold)
            right_mask = feature_values > value
            # Extract target values for left child samples
            left_labels = y[left_mask]
            # Extract target values for right child samples
            right_labels = y[right_mask]
            
            # Skip split if either child has too few samples
            if left_labels.size < self.min_samples_leaf or right_labels.size < self.min_samples_leaf:
                continue
            
            # Calculate impurity gain for this split
            gain = self.criterion_score(y, left_labels, right_labels)
            
            # Update best split if gain is better and meets minimum threshold
            if gain > best_gain and gain >= self.min_impurity_decrease:
                best_gain = gain
                best_value = value
        
        # Return None if no valid split found
        if best_value is None or best_gain <= 0.0:
            return None, None
        
        # Return the best gain and threshold value
        return best_gain, best_value

    def find_best_feature_split(self, X: np.ndarray | spmatrix, y: np.ndarray) -> tuple[int | None, float | None, float]:
        """
        Find the best feature and corresponding split value across all considered features by evaluating the impurity gain for each feature's best split.

        ## Args:
            **X**: *np.ndarray* or *spmatrix*
            Feature matrix containing the input features for all samples.

            **y**: *np.ndarray*
            Target values corresponding to the samples in X.

        ## Returns:
            **tuple**: *(best_feature, best_value, best_gain)* where best_feature is the index of the optimal feature, best_value is the threshold for splitting, and best_gain is the impurity gain, or *(None, None, None)* if no valid split is found.

        ## Raises:
            **None**
        """
        # Initialize best gain as -infinity
        best_gain = -np.inf
        # Initialize best feature as None
        best_feature = None
        # Initialize best value as None
        best_value = None
        # Get X column shape
        n_features = X.shape[1]
        # Get max features slicing index
        self.max_features = standard_indexing(n_features, self.max_features)
        # Feature slicing
        features = np.random.choice(n_features, self.max_features, replace=False)
        
        # Looping throught the features
        for feature_idx in features:
            # Get the best split
            result = self.find_best_split(X, y, feature_idx)
            # If result is None then skip
            if result is None:
                continue
            
            # Get gain and value from result
            gain, value = result
            # If gain is None then skip
            if gain is None:
                continue

            # Check if calculated gain is better than best gain
            if gain > best_gain:
                # Update best gain
                best_gain = gain
                # Update best feature
                best_feature = feature_idx
                # Update best value
                best_value = value
        
        # If best feature is None, then return tuple (None, None, None)
        if best_feature is None:
            return None, None, None
        
        # Return best feature, value and gain
        return best_feature, best_value, best_gain

    # ========== MAIN METHODS ==========
    def fit(self, X_train: np.ndarray | spmatrix, y_train: np.ndarray, depth: int = 0) -> dict:
        """
        Recursively fit the decision tree regressor to the training data by building a tree structure through optimal splits based on impurity reduction.

        ## Args:
            **X_train**: *np.ndarray* or *spmatrix*
            Training input features, where each row is a sample and each column is a feature.

            **y_train**: *np.ndarray*
            Training target values corresponding to each sample in X.

            **depth**: *int, default=0*
            Current depth of the tree during recursive building (used internally for recursion control).

        ## Returns:
            **dict**: *A nested dictionary representing the fitted tree structure, with keys 'feature', 'threshold', 'left', and 'right' for internal nodes, or 'value' for leaf nodes.*

        ## Raises:
            **ValueError**: *If input data is empty, dimensions mismatch, or contains invalid values.*
        """
        # Check if data is not sparse
        if not issparse(X_train):
            # If it is, make sure it's an array
            X = np.asarray(X_train)

        # If data is sparse, transform to CSR or CSC depend on the shape
        else:
            if X_train.shape[0] > X_train.shape[1]:
              X = X_train.tocsr()

            else:
              X = X_train.tocsc()

        y = np.asarray(y_train)

        # ---------- Data validation ----------
        # Make sure X and y data is not empty
        if X.shape[0] == 0 or y.size == 0:
            raise ValueError("X and y cannot be empty")
        
        # Make sure X and y data has the same length
        if X.shape[0] != y.size:
            raise ValueError("X and y must have the same length")

        # ---------- Sampling (if max_samples is set) ----------
        if depth == 0 and self.max_samples is not None:
            # Get slicing indexing for samples
            n_samples = standard_indexing(y.size, self.max_samples)
            # Slice the data if slicing index is not more than y array size
            if n_samples < y.size:
                indices = np.random.choice(y.size, n_samples, replace=False)
                # Slice X data
                X = X[indices]
                # Slice y data
                y = y[indices]

        if depth == 0:
            self.tree = {"value": float(np.mean(y))}

        # Stopping conditions (Leaf Node)
        if depth >= self.max_depth or y.size <= self.min_samples_leaf or y.size < self.min_samples_split:
            return {"value": float(np.mean(y))}

        # Find best split
        feature_idx, value, gain = self.find_best_feature_split(X, y)

        # Stopping conditions (No improvement or no split found)
        if feature_idx is None:
            return {"value": float(np.mean(y))}

        if gain < self.min_impurity_decrease:
            return {"value": float(np.mean(y))}

        # ---------- Recursive split ----------
        # Get sliced X data
        col = X[:, feature_idx]
        # If data is sparse transform to an array and flat it
        if issparse(col):
            col = col.toarray().ravel()
        
        # Create boolean mask for samples going to left child
        left_mask = col <= value
        # Create boolean mask for samples going to right child
        right_mask = col > value
        
        # Extract data values for left child samples
        left_X, left_y = X[left_mask], y[left_mask]
        # Extract data values for right child samples
        right_X, right_y = X[right_mask], y[right_mask]

        # If left or right child target value size is 0 return mean of target data
        if left_y.size == 0 or right_y.size == 0:
            return {"value": float(np.mean(y))}
        
        # Initialize node structure
        node = {
            "feature": int(feature_idx),
            "threshold": float(value),
            "left": self.fit(left_X, left_y, depth + 1),
            "right": self.fit(right_X, right_y, depth + 1)
        }

        # Store the final tree at the root level
        if depth == 0:
            self.tree = node

        # Return node
        return node

    def _predict_single(self, x: np.ndarray | spmatrix, tree: dict=None) -> float:
        """
        Predict the target value for a single sample by traversing the decision tree from root to leaf based on feature thresholds.

        ## Args:
            **x**: *np.ndarray* or *spmatrix*
            Feature vector for a single sample, with each element corresponding to a feature value.

            **tree**: *dict, optional*
            The tree or subtree dictionary to traverse for prediction. If None, uses the full trained tree.

        ## Returns:
            **float**: *The predicted target value for the input sample.*

        ## Raises:
            **None**
        """
        # ========== TRAVERSAL ==========
        if tree is None:
            tree = self.tree
        
        # Check if data is sparse
        if issparse(x):
            # Transform X to array and flat it
            x = x.toarray().ravel()

        # If not sparse, just make sure it's an array and flat it
        else:
            x = np.asarray(x).ravel()

        
        # If key "value" in tree and key "feature" no, then return tree/node key "value"
        if "value" in tree and "feature" not in tree:
            return tree["value"]
        
        # Get feature index from tree dict
        feature_idx = tree["feature"]
        # Get threshold from tree dict
        threshold = tree["threshold"]
        # Check if X with index 'feature_idx' is less than or same with threshold
        if x[feature_idx] <= threshold:
            # If it is, return predict single result from X and key "left" from tree dict
            return self._predict_single(x, tree["left"])
        
        # If it's not, return predict single result from X and key "right" from tree dict
        else:
            return self._predict_single(x, tree["right"])

    def predict(self, X_test: np.ndarray | spmatrix) -> np.ndarray:
        """
        Predict the target values for the given input features using the trained model.

        ## Args:
            **X_test**: *np.ndarray* or *spmatrix*
            Input features for prediction.

        ## Returns:
            **np.ndarray**: *predicted target values*

        ## Raises:
            **ValueError**: *If model tree is not defined (model not trained).*
        """
        # Check if data is sparse
        if not issparse(X_test):
            # Make sparse data to an array
            X = np.asarray(X_test)

        else:
            X = X_test
        
        # If X data dimention if 1 then reshape it to 2D
        if X_test.ndim == 1:
            X = X_test.reshape(1, -1)

        # Error check
        if self.tree is None:
            raise ValueError("Tree not defined, try to train the model with fit() function first")

        # ========== PREDICTION ==========
        predictions = []
        for x in X:
            predictions.append(self._predict_single(x))
        return np.array(predictions, dtype=float)

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
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "criterion": self.criterion,
            "max_features": self.max_features,
            "max_samples": self.max_samples,
            "random_state": self.random_state,
            "min_samples_split": self.min_samples_split,
            "min_impurity_decrease": self.min_impurity_decrease
        }

    def set_params(self, **params) -> 'TreeBackendRegressor':
        """
        Returns model's attribute that ready to set.

        ## Args:
            **params**: *dict*
            Model parameters to set.

        ## Returns:
            **TreeBackendRegressor**: *The model instance with updated parameters.*

        ## Raises:
            **None**
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
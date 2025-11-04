# ========== LIBRARIES ==========
import numpy as np                           # Numpy for numerical computations
from typing import Literal, Optional         # More specific type hints
from scipy.sparse import issparse, spmatrix  # For sparse matrix handling
from nexgml.helper.indexing import Indexing  # For indexing utilities
from nexgml.helper.amo import ForTree        # For some math operations

# ========== THE MODEL ==========
class TreeBackendClassifier:
    """
    TreeBackendClassifier (TBC) is a simple decision tree model for classification tasks.
    It builds a tree by recursively splitting nodes to minimize a given criterion and supports various pruning/stopping parameters to prevent overfitting.
    """
    def __init__(
            self,
            max_depth: int | None=6,
            min_samples_leaf: int | None=5,
            criterion: Literal['gini', 'entropy', 'log_loss'] | None='gini',
            max_features: Optional[Literal['sqrt', 'log2']] | int | float | None = None,
            max_samples: Optional[Literal['sqrt', 'log2']] | int | float | None = None,
            random_state: int | None=None,
            min_samples_split: int | None=2,
            min_impurity_decrease: float | None=0.0
            ) -> None:
        """
        Initialize the TreeBackendClassifier model.

        ## Args:
            **max_depth**: *int, default=6*
            Maximum depth of the tree.

            **min_samples_leaf**: *int, default=5*
            The minimum number of samples required to be at a leaf node.

            **criterion**: *{'gini', 'entropy', 'log_loss'}, default='gini'*
            The function to measure the quality of a split.

            **max_features**: *{'sqrt', 'log2'} or int or float or None, default=None*
            The number of features to consider when looking for the best split.

            **max_samples**: *{'sqrt', 'log2'} or int or float or None, default=None*
            The number of samples to draw from X to train the tree.

            **random_state**: *int or None, default=None*
            Seed for random number generator for reproducibility.

            **min_samples_split**: *int, default=2*
            The minimum number of samples required to split an internal node.
            
            **min_impurity_decrease**: *float, default=0.0*
            Tolerance for splitting. A node will be split if this split induces a decrease of the impurity greater than or equal to this value.

        ## Returns:
            **None**

        ## Raises:
            **ValueError**: *If invalid criterion is provided, if min_samples_split < 2 * min_samples_leaf, if max_depth, min_samples_leaf, min_samples_split and min_impurity_decrease is non-positive.*
        """
        # ========== PARAMETER VALIDATIONS ==========
        if criterion not in ('gini', 'entropy', 'log_loss'):
            raise ValueError(f"Invalid criterion argument, {criterion}. Choose from 'gini', 'entropy', or 'log_loss'.")

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

    def compute_variance_sparse(self, X: spmatrix) -> np.ndarray:
        """
        Compute the variance of each feature column in a sparse matrix.

        This method calculates the variance for each column of a sparse matrix X.
        Variance is computed as E[X^2] - (E[X])^2.

        ## Args:
            **X**: *scipy.sparse.spmatrix*
            Sparse input feature matrix.

        ## Returns:
            **np.ndarray**: *Array of variances for each feature column.*
            
        ## Raises:
            **None**
        """
        # Initialize list for variances
        variances = []
        # Loop over each column
        for col in range(X.shape[1]):
            # Get data from column
            col_data = X[:, col]
            # Calculate mean
            mean = col_data.mean()
            # Calculate mean of squared data
            mean_sq = (col_data.multiply(col_data)).mean()
            # Calculate variance
            var = mean_sq - mean**2
            # Append to list
            variances.append(var)
        # Return as numpy array
        return np.array(variances)

    def find_best_split(self, X: np.ndarray | spmatrix, y: np.ndarray, feature_idx: int) -> tuple[float, float | None]:
        """
        Find the best split value for a given feature by evaluating potential split points and selecting the one that minimizes the impurity.

        ## Args:
            **X**: *np.ndarray* or *spmatrix*
            Feature matrix containing the input features for all samples.

            **y**: *np.ndarray*
            Target labels corresponding to the samples in X.

            **feature_idx**: *int*
            Index of the feature column to evaluate for the best split.

        ## Returns:
            **tuple**: *(best_impurity, best_value)* where best_impurity is the minimum impurity from the split and best_value is the threshold value for the split, or *(float('inf'), None)* if no valid split is found.

        ## Raises:
            **None**
        """
        # Check if label size is less than min samples split
        if len(y) < self.min_samples_split:
            return float('inf'), None

        # Check if data is sparse
        if issparse(X):
            # Slice and transform to array then flatten
            feature_values = X[:, feature_idx].toarray().ravel()
        # If not sparse
        else:
            # Just slice
            feature_values = X[:, feature_idx]

        # Get unique values
        unique_values = np.unique(feature_values)
        # If 1 or less unique values, no split
        if len(unique_values) <= 1:
            return float('inf'), None

        # If many unique values, use percentiles for efficiency
        if len(unique_values) > 100:
            # Set number of split points
            num_points = 51 if issparse(X) else 101
            # Get percentile
            split_points = np.percentile(feature_values, np.linspace(0, 100, num_points))
            # Get unique percentile
            split_points = np.unique(split_points)
        # If few unique values
        else:
            # Use midpoints
            split_points = (unique_values[:-1] + unique_values[1:]) / 2.0

        # Initialize best impurity
        best_impurity = float('inf')
        # Initialize best value
        best_value = None

        # Iterate over split points
        for value in split_points:
            # Get left mask
            left_mask = feature_values <= value
            # Get right mask
            right_mask = feature_values > value

            # Get left labels
            left_labels = y[left_mask]
            # Get right labels
            right_labels = y[right_mask]

            # If no labels, continue
            if len(left_labels) == 0 or len(right_labels) == 0:
                continue

            # Calculate impurity based on criterion
            if self.criterion == 'gini':
                left_impurity = ForTree.gini_impurity(left_labels)
                right_impurity = ForTree.gini_impurity(right_labels)

            elif self.criterion == 'entropy':
                left_impurity = ForTree.entropy_impurity(left_labels)
                right_impurity = ForTree.entropy_impurity(right_labels)

            elif self.criterion == 'log_loss':
                left_impurity = ForTree.log_loss_impurity(left_labels)
                right_impurity = ForTree.log_loss_impurity(right_labels)

            # Calculate weighted impurity
            weighted_impurity = (len(left_labels) / len(y)) * left_impurity + \
                                (len(right_labels) / len(y)) * right_impurity

            # Get current impurity
            if self.criterion == 'gini':
                current_impurity = ForTree.gini_impurity(y)
            elif self.criterion == 'entropy':
                current_impurity = ForTree.entropy_impurity(y)
            elif self.criterion == 'log_loss':
                current_impurity = ForTree.log_loss_impurity(y)

            # Check if split improves impurity and meets decrease threshold
            if weighted_impurity < best_impurity and (current_impurity - weighted_impurity) >= self.min_impurity_decrease:
                best_impurity = weighted_impurity
                best_value = value

        # Return best impurity and value
        return best_impurity, best_value
    
    def find_best_feature_split(self, X: np.ndarray | spmatrix, y: np.ndarray) -> tuple[int | None, float | None, float]:
        """
        Find the best feature and corresponding split value across all considered features by evaluating the minimum impurity for each feature's best split.

        ## Args:
            **X**: *np.ndarray* or *spmatrix*
            Feature matrix containing the input features for all samples.

            **y**: *np.ndarray*
            Target labels corresponding to the samples in X.

        ## Returns:
            **tuple**: *(best_feature, best_value, best_impurity)* where best_feature is the index of the optimal feature, best_value is the threshold for splitting, and best_impurity is the minimum impurity, or *(None, None, float('inf'))* if no valid split is found.

        ## Raises:
            **None**
        """
        # Initialize best impurity
        best_impurity = float('inf')
        # Initialize best feature
        best_feature = None
        # Initialize best value
        best_value = None

        # Get number of features
        n_features = X.shape[1]
        # Get max features index
        self.max_features = Indexing.standard_indexing(n_features, self.max_features)

        # Select subset of features if max_features is set
        if self.max_features < n_features:
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)
        else:
            feature_indices = range(n_features)

        # Calculate variance for feature selection
        if issparse(X):
            variance = self.compute_variance_sparse(X)
        else:
            variance = np.var(X, axis=0)

        # If no variance, no split
        if np.all(variance == 0):
            return None, None, float('inf')

        # Iterate over selected features
        for feature_idx in feature_indices:
            # Skip features with no variance
            if variance[feature_idx] == 0:
                continue

            # Find best split for this feature
            impurity, value = self.find_best_split(X, y, feature_idx)

            # Update if this split is better
            if impurity < best_impurity:
                best_impurity = impurity
                best_feature = feature_idx
                best_value = value
        
        # Return best split info
        return best_feature, best_value, best_impurity

    # ========== MAIN METHODS ==========
    def fit(self, X: np.ndarray | spmatrix, y: np.ndarray, depth=0) -> dict:
        """
        Recursively fit the decision tree classifier to the training data by building a tree structure through optimal splits based on impurity reduction.

        ## Args:
            **X**: *np.ndarray* or *spmatrix*
            Training input features, where each row is a sample and each column is a feature.

            **y**: *np.ndarray*
            Training target labels corresponding to each sample in X.

            **depth**: *int, default=0*
            Current depth of the tree during recursive building (used internally for recursion control).

        ## Returns:
            **dict**: *A nested dictionary representing the fitted tree structure, with keys 'feature', 'value', 'left', and 'right' for internal nodes, or 'label' for leaf nodes.*

        ## Raises:
            **ValueError**: *If input data is empty or dimensions mismatch.*
        """
        # ---------- Data validation ----------
        # Ensure X is array if not sparse
        if not issparse(X):
            X = np.asarray(X)
        # Convert to CSR or CSC if sparse
        else:
            if X.shape[0] > X.shape[1]:
              X = X.tocsr()

            else:
              X = X.tocsc()

        # Ensure y is array
        y = np.asarray(y)
        
        # Check for empty data or mismatch length
        if not issparse(X) and not issparse(y):
            if len(X) == 0 or len(y) == 0:
                raise ValueError("X and y cannot be empty")
                
            if len(X) != len(y):
                raise ValueError("X and y must have the same length")
        
        # Handle sparse matrix shape access
        else:
            if X.shape[0] == 0 or y.shape[0] == 0:
                raise ValueError("X and y cannot be empty")

            if X.shape[0] != y.shape[0]:
                raise ValueError("X and y must have the same length")
            
        # ---------- Sampling (if max_samples is set) ----------
        if depth == 0 and self.max_samples is not None:
            # Get number of samples to use
            n_samples = Indexing.standard_indexing(y.size, self.max_samples)
            # Subsample if n_samples is less than total
            if n_samples < y.size:
                indices = np.random.choice(y.size, n_samples, replace=False)
                X = X[indices]
                y = y[indices]
            
        # ---------- Stopping conditions (Leaf Node) ----------
        # If pure node (all labels are the same)
        if len(np.unique(y)) == 1:
            return {"label": int(y[0])}

        # If max depth reached or not enough samples
        if depth >= self.max_depth or len(y) < self.min_samples_leaf or len(y) < self.min_samples_split:
            unique_classes, counts = np.unique(y, return_counts=True)
            prediction = unique_classes[np.argmax(counts)]
            return {"label": int(prediction)}

        # Find best split
        feature_idx, value, impurity_value = self.find_best_feature_split(X, y)

        # If no valid split found
        if feature_idx is None:
            unique_classes, counts = np.unique(y, return_counts=True)
            prediction = unique_classes[np.argmax(counts)]
            return {"label": int(prediction)}

        # ---------- Recursive split ----------
        # Get feature column
        col = X[:, feature_idx]
        # Convert to array if sparse
        if issparse(col):
            col = col.toarray().ravel()
        
        # Create masks for split
        left_mask = col <= value
        right_mask = col > value
            
        # Split data
        left_X, left_y = X[left_mask], y[left_mask]
        right_X, right_y = X[right_mask], y[right_mask]

        # If split results in empty child, create leaf
        if len(left_y) == 0 or len(right_y) == 0:
            unique_classes, counts = np.unique(y, return_counts=True)
            prediction = unique_classes[np.argmax(counts)]
            return {"label": int(prediction)}

        # Build node
        node = {
            "feature": feature_idx,
            "value": value,
            "left": self.fit(left_X, left_y, depth + 1),
            "right": self.fit(right_X, right_y, depth + 1)
        }
        
        # Store tree at root
        if depth == 0:
            self.tree = node
        
        # Return node
        return node
    
    def _predict_single(self, x: np.ndarray, tree: dict=None) -> int:
        """
        Predict the class label for a single sample by traversing the decision tree.

        ## Args:
            **x**: *np.ndarray* or *spmatrix*
            Feature vector for a single sample.

            **tree**: *dict, optional*
            The tree or subtree dictionary to traverse. If None, uses the full trained tree.

        ## Returns:
            **int**: *The predicted class label for the input sample.*

        ## Raises:
            **None**
        """
        # ========== TRAVERSAL ==========
        if tree is None:
            tree = self.tree

        # Convert sparse to dense array
        if issparse(x):
            x = x.toarray().ravel()
        # Ensure dense is 1D array
        else:
            x = np.asarray(x).ravel()

        # If leaf node, return label
        if "label" in tree:
            return tree["label"]

        # Get split info
        feature_idx = tree["feature"]
        value = tree["value"]

        # Recurse down the tree
        if x[feature_idx] <= value:
            return self._predict_single(x, tree["left"])
        else:
            return self._predict_single(x, tree["right"])

    def predict(self, X: np.ndarray | spmatrix) -> np.ndarray:
        """
        Predict class labels for multiple samples.

        ## Args:
            **X**: *np.ndarray* or *spmatrix*
            Input features for prediction.

        ## Returns:
            **np.ndarray**: *Array of predicted class labels.*

        ## Raises:
            **ValueError**: *If model tree is not defined (model not trained).*
        """
        # Ensure X is array if not sparse
        if not issparse(X):
            X = np.asarray(X)

        # Handle single sample prediction
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        # Error check
        if self.tree is None:
            raise ValueError("Tree not defined, try to train the model with fit() function first")

        # ========== PREDICTION ==========
        predictions = []
        for x in X:
            predictions.append(self._predict_single(x))
        return np.array(predictions)

    def score(self, X: np.ndarray | spmatrix, y: np.ndarray) -> float:
        """
        Calculate the mean accuracy on the given test data and labels.

        ## Args:
            **X**: *np.ndarray* or *spmatrix*
            Feature matrix.

            **y**: *np.ndarray*
            True target labels.

        ## Returns:
            **float**: *Mean accuracy score.*

        ## Raises:
            **None**
        """
        # ========== PREDICTION ==========
        y_pred = self.predict(X)
        # Compare prediction with true labels and compute mean
        return np.mean(y_pred == y)
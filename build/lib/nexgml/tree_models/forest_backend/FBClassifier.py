# ========== LIBRARIES ==========
import numpy as np                                # Numpy for numerical computations
from typing import Optional, Literal              # More specific type hints
from scipy.sparse import spmatrix, issparse       # For sparse matrix handling
from nexgml.tree_models.tree_backend.TBClassifier import TreeBackendClassifier  # Estimator model
from nexgml.indexing import standard_indexing     # For indexing utilities
from nexgml.metrics import accuracy_score         # For accuracy metric

# ========== THE MODEL ==========
class ForestBackendClassifier:
    """
    ForestBackendClassifier (FBC) is an ensemble model for classification tasks based on decision trees.
    It builds multiple trees (a "forest") by randomly sampling data and features, and aggregates their predictions (majority vote) 
    to improve accuracy and control overfitting.
    """

    def __init__(
        self,
        n_estimators: int = 10,
        bootstrap: bool = True,
        double_sampling: bool | None=False,
        max_depth: int | None = 6,
        min_samples_leaf: int | None = 5,
        criterion: Literal['gini', 'entropy', 'log_loss'] | None = 'gini',
        max_features: Optional[Literal['sqrt', 'log2']] | int | float | None='sqrt',
        max_samples: Optional[Literal['sqrt', 'log2']] | int | float | None='sqrt',
        random_state: Optional[int] = None,
        min_samples_split: int | None = 2,
        min_impurity_decrease: float | None = 0.0,
        verbose: int | None=0
    ):
        """
        Initialize the ForestBackendClassifier model.

        ## Args:
            **n_estimators**: *int, default=10*
            The number of trees in the forest.

            **bootstrap**: *bool, default=True*
            Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.
            
            **double_sampling**: *bool, default=False*
            Whether to apply feature and sample slicing inside the TreeBackendClassifier as well (double sampling).

            **max_depth**: *int, default=6*
            Maximum depth of each tree.

            **min_samples_leaf**: *int, default=5*
            The minimum number of samples required to be at a leaf node for each tree.

            **criterion**: *{'gini', 'entropy', 'log_loss'}, default='gini'*
            The function to measure the quality of a split for each tree.

            **max_features**: *{'sqrt', 'log2'} or int or float or None, default='sqrt'*
            The number of features to consider when looking for the best split for each tree.

            **max_samples**: *{'sqrt', 'log2'} or int or float or None, default='sqrt'*
            The number of samples to draw from X to build each tree (if bootstrap=True).

            **random_state**: *int or None, default=None*
            Seed for random number generator for reproducibility.

            **min_samples_split**: *int, default=2*
            The minimum number of samples required to split an internal node for each tree.

            **min_impurity_decrease**: *float, default=0.0*
            Tolerance for splitting. A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
            
            **verbose**: *int, default=0*
            Controls the verbosity when fitting the model.

        ## Returns:
            **None**

        ## Raises:
            **ValueError**: *If invalid criterion is provided or if min_samples_split < 2 * min_samples_leaf, if n_estimators, max_depth, min_samples_leaf, min_samples_split and min_impurity_decrease is non-positive.*
        """
        
        # ========== PARAMETER VALIDATIONS ==========
        if n_estimators is None or n_estimators <= 0:
            raise ValueError(f"Invalid n_estimators argument, {n_estimators}. n_estimators should be a positive integer.")

        if criterion not in ('gini', 'entropy', 'log_loss'):
            raise ValueError(f"Invalid criterion argument, {criterion}. Choose from 'gini', 'entropy' or 'log_loss'.")

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
        self.n_estimators = int(n_estimators)
        self.bootstrap = bool(bootstrap)
        self.double_sampling = double_sampling
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.criterion = criterion
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state
        self.min_samples_split = int(min_samples_split)
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.verbose = int(verbose)

        # ---------- Model structures ----------
        # Our "forest" - list of trained decision trees
        self.trees = []
        # Which features each tree used
        self.feature_indices = []

    # ========== MAIN METHODS ==========
    def fit(self, X: np.ndarray | spmatrix, y: np.ndarray):
        """
        Train the Random Forest Classifier on the training data by building multiple decision trees.

        ## Args:
            **X**: *np.ndarray* or *spmatrix*
            Training input features, where each row is a sample and each column is a feature.

            **y**: *np.ndarray*
            Training target labels corresponding to each sample in X.

        ## Returns:
            **None**

        ## Raises:
            **None**
        """
        # ---------- Random state setup ----------
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # ---------- Data preparation ----------
        # Convert to numpy arrays if needed
        if not issparse(X):
            X = np.asarray(X)
        y = np.asarray(y)

        # Get data shape
        n_samples, n_features = X.shape

        # ---------- Feature/Sample size setup ----------
        # Determine how many features/samples each tree should use
        self.max_features = standard_indexing(n_features, self.max_features)
        self.max_samples = standard_indexing(n_samples, self.max_samples)

        # ---------- Train each tree ----------
        for i in range(self.n_estimators):
            # ------ Bootstrap sampling ------
            # Create a random subset of the data
            if self.bootstrap:
                indices = np.random.choice(n_samples, size=self.max_samples, replace=True)
            else:
                indices = np.arange(n_samples)  # Use all data

            # ------ Feature subset selection ------
            # Select random subset of features for this tree
            feature_idx = np.random.choice(n_features, size=self.max_features, replace=False)
            self.feature_indices.append(feature_idx)

            # ------ Data subset creation ------
            X_sub = X[indices]
            if issparse(X_sub):
                X_sub = X_sub[:, feature_idx]  # Sparse matrix slicing
            else:
                X_sub = X_sub[:, feature_idx]
            y_sub = y[indices]

            # ------ Tree creation and training ------
            # Create and train the decision tree
            tree = TreeBackendClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                criterion=self.criterion,
                min_samples_split=self.min_samples_split,
                min_impurity_decrease=self.min_impurity_decrease,
                max_features=self.max_features if self.double_sampling else None,
                max_samples=self.max_samples if self.double_sampling else None
            )
            tree.fit(X_sub, y_sub)
            self.trees.append(tree)

            # ------ Progress update ------
            if self.verbose == 1:
              print(f"Trained tree {i+1}/{self.n_estimators}")

    def predict(self, X: np.ndarray | spmatrix) -> np.ndarray:
        """
        Predict class labels for the given input features by aggregating predictions from all trees in the forest (majority vote).

        ## Args:
            **X**: *np.ndarray* or *spmatrix*
            Input features for prediction.

        ## Returns:
            **np.ndarray**: *predicted class labels*

        ## Raises:
            **ValueError**: *If model is not trained (trees list is empty).*
        """
        # ---------- Model validation ----------
        if not self.trees:
            raise ValueError("Forest not defined, try to train the model with fit() function first")

        # ---------- Data preparation ----------
        if not issparse(X):
            X = np.asarray(X)
        
        n_samples = X.shape[0]

        # ---------- Prediction Aggregation ----------
        # Collect predictions from all trees
        all_predictions = np.zeros((n_samples, self.n_estimators), dtype=int)

        for i, tree in enumerate(self.trees):
            # Get features used by this tree
            feature_idx = self.feature_indices[i]
            
            # Slice X to only include those features
            if issparse(X):
                X_sub = X[:, feature_idx]
            else:
                X_sub = X[:, feature_idx]
                
            # Get prediction
            all_predictions[:, i] = tree.predict(X_sub)

        # ---------- Majority Voting ----------
        # Perform majority voting
        final_predictions = np.zeros(n_samples, dtype=int)
        for sample_idx in range(n_samples):
            sample_preds = all_predictions[sample_idx, :]
            unique_classes, counts = np.unique(sample_preds, return_counts=True)
            # Choose the class with the most votes
            final_predictions[sample_idx] = unique_classes[np.argmax(counts)]

        return final_predictions

    def predict_proba(self, X: np.ndarray | spmatrix) -> np.ndarray:
        """
        Predict class probabilities for the input data.
        The probability is the proportion of trees that voted for each class.

        ## Args:
            **X**: *np.ndarray* or *spmatrix*
            Input features for prediction.

        ## Returns:
            **np.ndarray**: *predicted probabilities of shape (n_samples, n_classes)*

        ## Raises:
            **ValueError**: *If model is not trained (trees list is empty).*
        """
        # ---------- Model validation ----------
        if not self.trees:
            raise ValueError("Forest not defined, try to train the model with fit() function first")

        # ---------- Data preparation ----------
        if not issparse(X):
            X = np.asarray(X)

        n_samples = X.shape[0]

        # ---------- Prediction Aggregation ----------
        # Collect predictions from all trees
        all_predictions = np.zeros((n_samples, self.n_estimators), dtype=int)

        for i, tree in enumerate(self.trees):
            # Get features used by this tree
            feature_idx = self.feature_indices[i]
            
            # Slice X
            if issparse(X):
                X_sub = X[:, feature_idx]
            else:
                X_sub = X[:, feature_idx]
                
            # Get prediction
            all_predictions[:, i] = tree.predict(X_sub)

        # ---------- Probability Calculation ----------
        # Get unique classes from all predictions
        all_unique_classes = np.unique(all_predictions)
        n_classes = len(all_unique_classes)
        class_to_idx = {cls: idx for idx, cls in enumerate(all_unique_classes)}

        # Compute probabilities
        probabilities = np.zeros((n_samples, n_classes))
        for sample_idx in range(n_samples):
            sample_preds = all_predictions[sample_idx, :]
            for cls in all_unique_classes:
                count = np.sum(sample_preds == cls)
                probabilities[sample_idx, class_to_idx[cls]] = count / self.n_estimators

        return probabilities
    
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
        
        # ========== ACCURACY CALCULATION ==========
        return accuracy_score(y_test, y_pred)
    
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
            "n_estimators": self.n_estimators,
            "bootstrap": self.bootstrap,
            "double_sampling": self.double_sampling,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "criterion": self.criterion,
            "max_features": self.max_features,
            "max_samples": self.max_samples,
            "random_state": self.random_state,
            "min_samples_split": self.min_samples_split,
            "min_impurity_decrease": self.min_impurity_decrease,
            "verbose": self.verbose
        }

    def set_params(self, **params) -> "ForestBackendClassifier":
        """
        Returns model's attribute that ready to set.

        ## Args:
            **params**: *dict*
            Model parameters to set.

        ## Returns:
            **ForestBackendClassifier**: *The model instance with updated parameters.*

        ## Raises:
            **None**
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
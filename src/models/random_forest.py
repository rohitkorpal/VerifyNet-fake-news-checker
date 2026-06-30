import numpy as np
from src.models.base import BaseModel

class DecisionNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        """
        Decision Tree Node.
        If value is not None, this is a leaf node.
        """
        self.feature = feature          # Index of feature to split on
        self.threshold = threshold      # Threshold value to split at
        self.left = left                # Left subtree
        self.right = right              # Right subtree
        self.value = value              # Predicted class if leaf node

    def is_leaf(self):
        return self.value is not None


class DecisionTreeClassifier(BaseModel):
    def __init__(self, max_depth=10, min_samples_split=2, max_features=None):
        """
        Decision Tree Classifier from scratch.
        max_features: number of features to consider when looking for the best split.
                      If None, consider all features.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None

    def _gini(self, y):
        """
        Computes Gini Impurity of labels y.
        """
        m = len(y)
        if m == 0:
            return 0.0
        counts = np.bincount(y)
        probs = counts / m
        return 1.0 - np.sum(probs ** 2)

    def _best_split(self, X, y, feature_indices):
        """
        Finds the best split using Gini Impurity.
        """
        best_gain = -1.0
        split_idx, split_thresh = None, None
        
        n_samples = X.shape[0]
        current_gini = self._gini(y)
        
        for feat_idx in feature_indices:
            X_column = X[:, feat_idx]
            
            # To speed up continuous search, inspect a subset of thresholds (e.g. 5 percentiles)
            # instead of every single unique value in the dataset
            uniques = np.unique(X_column)
            if len(uniques) > 10:
                thresholds = np.percentile(X_column, [10, 30, 50, 70, 90])
            else:
                thresholds = uniques
                
            for thresh in thresholds:
                # Split indices
                left_mask = X_column <= thresh
                right_mask = ~left_mask
                
                # Check min samples requirement
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                    
                # Gini calculation for split
                y_left, y_right = y[left_mask], y[right_mask]
                w_left = len(y_left) / n_samples
                w_right = len(y_right) / n_samples
                
                gain = current_gini - (w_left * self._gini(y_left) + w_right * self._gini(y_right))
                
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thresh
                    
        return split_idx, split_thresh

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # Check stopping criteria
        if (depth >= self.max_depth or 
            n_labels == 1 or 
            n_samples < self.min_samples_split):
            
            # Predict majority class for leaf
            leaf_value = np.argmax(np.bincount(y)) if len(y) > 0 else 0
            return DecisionNode(value=leaf_value)
            
        # Select features to consider for splitting
        if self.max_features is None:
            feature_indices = np.arange(n_features)
        else:
            # Randomly select a subset of features without replacement
            max_feat = min(n_features, self.max_features)
            feature_indices = np.random.choice(n_features, max_feat, replace=False)
            
        # Find best split
        best_feat, best_thresh = self._best_split(X, y, feature_indices)
        
        # If no split gives any information gain, create leaf
        if best_feat is None:
            leaf_value = np.argmax(np.bincount(y)) if len(y) > 0 else 0
            return DecisionNode(value=leaf_value)
            
        # Recursive split
        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask
        
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return DecisionNode(feature=best_feat, threshold=best_thresh, left=left_child, right=right_child)

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).astype(int)
        self.root = self._build_tree(X, y)
        return self

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
            
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def predict(self, X):
        X = np.array(X)
        return np.array([self._traverse_tree(x, self.root) for x in X])


class RandomForestClassifier(BaseModel):
    def __init__(self, n_estimators=10, max_depth=8, min_samples_split=2, max_features='sqrt'):
        """
        Random Forest Classifier from scratch.
        n_estimators: number of trees in the forest.
        max_features: 'sqrt' or int. Number of features to consider at each split.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        """
        Fits n_estimators trees to bootstrap samples.
        """
        X = np.array(X)
        y = np.array(y).astype(int)
        
        n_samples, n_features = X.shape
        self.trees = []
        
        # Set max_features count
        if self.max_features == 'sqrt':
            max_feats = int(np.sqrt(n_features))
        elif isinstance(self.max_features, int):
            max_feats = self.max_features
        else:
            max_feats = n_features
            
        for _ in range(self.n_estimators):
            # Generate bootstrap sample
            bootstrap_idx = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[bootstrap_idx]
            y_bootstrap = y[bootstrap_idx]
            
            # Initialize and fit tree
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_feats
            )
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
            
        return self

    def predict(self, X):
        """
        Aggregates predictions from all trees using majority vote.
        """
        X = np.array(X)
        # Collect predictions from each tree: shape (n_trees, n_samples)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        
        # Transpose to (n_samples, n_trees)
        tree_preds = tree_preds.T
        
        # Take majority vote for each sample
        final_preds = []
        for sample_preds in tree_preds:
            counts = np.bincount(sample_preds)
            final_preds.append(np.argmax(counts))
            
        return np.array(final_preds)

from .utils import (mean_adjacent, impurity_function, 
                    gini_impurity, get_majority_class, 
                    loss_function, sse, get_mean, check_data_validity)
import numpy as np 
from .node import Node, LeafNode

EPSILON = np.finfo('float32').eps
class DecisionTree:
    def __init__(self, max_depth=5, min_samples_to_split=4, min_samples_leaf=2,
                 feature_names=[], class_names=[], tree_type='classification'):
        self.root: Node = Node(None, None)
        self.tree_type: str = tree_type
        
        # hyperparameters
        self.max_depth = max_depth
        self.min_samples_to_split = min_samples_to_split
        self.min_samples_leaf = min_samples_leaf

        self.feature_names = feature_names
        self.class_names = class_names
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        # checking if the data is valid
        check_data_validity(X)
        
        # calling the grow method with the data and the feature indices
        self.root = self._grow(np.hstack((X, y)))
    
    # predict a single sample
    def predict(self, X):
        return self.root.predict(X)
    
    # split the data based on the feature and the threshold
    def _split_data(self, feature_data, labels, thresh: float):
        #splitting left
        left_indices = feature_data < thresh
        left_feature_data = feature_data[left_indices]
        left_labels = labels[left_indices]

        #splitting right
        right_indices  = feature_data >= thresh
        right_feature_data = feature_data[right_indices]
        right_labels = labels[right_indices]

        return (left_feature_data, left_labels), (right_feature_data, right_labels)
    
    # find the best split for the data based on the partitioning criterion for each pair of feature and threshold
    def _best_split(self, feature_data, labels, thresholds):
        min_split_cost = np.inf
        min_cost_thresh = None
        # for each threshold
        for thresh in thresholds:
            left_data, right_data = self._split_data(feature_data, labels, thresh)

            left_labels = left_data[1]
            right_labels = right_data[1]

            cost_function = impurity_function
            if self.tree_type == 'regression':
                cost_function = loss_function

            # calculate the partitioning criterion for the feature with the specific threshold
            cost_value = cost_function(left_labels, right_labels)
            
            if cost_value < min_split_cost:
                min_split_cost = cost_value
                min_cost_thresh = thresh
        
        return min_cost_thresh, min_split_cost
    

    def _best_feature(self, data, feature_idxs):

        min_feature_cost = np.inf
        min_feature_threshold = None
        selected_feature = None
        # for each feature in a list of the features index
        for feature_idx in feature_idxs:

            # get the feature data
            feature_data = data[:, feature_idx]
            labels = data[:, -1]
            
            unique_values = np.sort(np.unique(feature_data))
            
            # generate the thresholds
            thresholds = mean_adjacent(unique_values, window_size=2)
            min_thresh, min_split_cost = self._best_split(feature_data, labels, thresholds)

            if min_split_cost < min_feature_cost:
                min_feature_cost = min_split_cost
                min_feature_threshold = min_thresh
                selected_feature = feature_idx
        

        return selected_feature, min_feature_threshold, min_feature_cost
        

    def _grow(self, data, depth=1):
        compute_criterion_value = gini_impurity
        get_result = get_majority_class
        if self.tree_type == 'regression':
            compute_criterion_value = sse
            get_result = get_mean

        y = data[:, -1]
        # Calculate the criterion value of the data
        criterion_value = compute_criterion_value(y)
        result = get_result(y)

        class_name = self.class_names[result] if self.tree_type=='classification' else f"{result:.4f}"

        # Stopping criteria
        if self.max_depth and depth >= self.max_depth:
            return LeafNode(data, criterion_value=criterion_value, 
                            _result = result, class_name=class_name)
        if len(data) < self.min_samples_to_split:
            return LeafNode(data, criterion_value=criterion_value, 
                            _result=result, class_name=class_name)
        
        if criterion_value < EPSILON:
            return LeafNode(data, criterion_value=criterion_value, 
                            _result=result,class_name=class_name)
        
        feature_idxs = np.arange(data.shape[-1] - 1)

        # splitting
        selected_feature, min_feature_threshold, _ = self._best_feature(data, feature_idxs)
        
        # Split data based on best split
        left_data = data[data[:, selected_feature] < min_feature_threshold]
        right_data = data[data[:, selected_feature] >= min_feature_threshold]

        # Stopping criteria
        if (len(left_data) < self.min_samples_leaf 
            or len(right_data) < self.min_samples_leaf):
            return LeafNode(data, criterion_value=criterion_value, 
                            _result=result, class_name=class_name)

        # Create child nodes
        left_node = self._grow(left_data, depth=depth+1)
        right_node = self._grow(right_data, depth=depth+1)

        return Node(left_node, 
                    right_node,
                    data, 
                    selected_feature, 
                    min_feature_threshold, 
                    feature_name=self.feature_names[selected_feature] if any(self.feature_names) else "NA",
                    criterion_value = criterion_value,
                    _result=result, class_name=class_name)
    
    # DFS traversal
    def _traverse_dfs(self, node: Node):
        if node is None:
            return []

        # Recursively traverse the left and right subtrees
        left_nodes = self._traverse_dfs(node.left)
        right_nodes = self._traverse_dfs(node.right)

        # Combine the nodes from the current node, left subtree, and right subtree
        nodes = [node] + left_nodes + right_nodes

        return nodes
    
    # BFS traversal
    def _traverse_bfs(self, node: Node):
        if node is None:
            return []

        queue = [node]
        nodes = []
        while queue:
            current_node = queue.pop(0)
            nodes.append(current_node)
            if current_node.left:
                queue.append(current_node.left)
            if current_node.right:
                queue.append(current_node.right)
        
        return nodes
    
    def get_tree_structure_dfs(self):
        return self._traverse_dfs(self.root)
    
    def get_tree_structure_bfs(self):
        return self._traverse_bfs(self.root)

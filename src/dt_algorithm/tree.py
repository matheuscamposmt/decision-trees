from .utils import (mean_adjacent, impurity_function, 
                    gini_impurity, get_majority_class, 
                    loss_function, sse, get_mean)
import numpy as np
from .node import Node, LeafNode

EPSILON = np.finfo('double').eps
class DecisionTree:
    def __init__(self, max_depth=None, min_samples_to_split=4, feature_names=[], class_names=[]):
        Node.count = 0
        self.root: Node = Node(None, None)
        
        self.max_depth = max_depth
        self.min_samples_to_split = min_samples_to_split
        self.feature_names = feature_names
        
        self.classes = []
        self.class_names = class_names

        self.tree_type: str = 'classification'
    
    def fit(self, X: np.ndarray, y: np.ndarray):

        # checking the dtype of the labels
        if y.dtype == np.float64 or y.dtype == np.float32:
            self.tree_type = 'regression'
        
        # redundancy for sake of clarity
        elif y.dtype == np.int64 or y.dtype == np.int32:
            self.tree_type = 'classification'

        # if not a valid dtype
        else:
            raise TypeError("y must be of type np.float64, np.float32, np.int64, or np.int32")
        
        # calling the grow method with the data and the feature indices
        self.root = self._grow(np.hstack((X, y)), np.arange(X.shape[-1]))
    
    # predict a single sample
    def predict(self, X):
        return np.apply_along_axis(self.root.predict, arr=X, axis=1)
    
    # split the data based on the feature and the threshold
    def _split_data(self, feature_data: np.ndarray, labels: np.ndarray, thresh: float):
        #splitting left
        left_indices = feature_data < thresh
        left_feature_data = feature_data[left_indices]
        left_labels = labels[left_indices]

        #splitting right
        right_indices  = feature_data >= thresh
        right_feature_data = feature_data[right_indices]
        right_labels = labels[right_indices]

        return (left_feature_data, left_labels), (right_feature_data, right_labels)
    
    # find the best split for the data based on the cost function for each pair of feature and threshold
    def _best_split(self, feature_data: np.ndarray, labels: np.ndarray, thresholds):
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

            # calculate the cost for the feature with the specific threshold
            cost = cost_function(left_labels, right_labels)
            
            if cost < min_split_cost:
                min_split_cost = cost
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
        

    def _grow(self, data, feature_idxs, depth=1):
        compute_criteria = gini_impurity
        get_result = get_majority_class
        if self.tree_type == 'regression':
            compute_criteria = sse
            get_result = get_mean

        # Calculate the criteria value of the data
        y = data[:, -1]
        criteria_value = compute_criteria(y)
        result = get_result(y)

        class_name = self.class_names[result] if self.tree_type=='classification' else result

        # Stopping criteria
        if self.max_depth and depth >= self.max_depth:
            #print(f"Limit depth reached: {depth}. Number of samples: {len(data)}")
            return LeafNode(data, criteria_value=criteria_value, 
                            _result = result, class_name=class_name)
        if self.min_samples_to_split and (len(data) < self.min_samples_to_split):
            #print(f"Data with {len(data)} samples, returning LeafNode with depth {depth}")
            return LeafNode(data, criteria_value=criteria_value, 
                            _result=result, class_name=class_name)
        if criteria_value < EPSILON:
            return LeafNode(data, criteria_value=criteria_value, 
                            _result=result,class_name=class_name)

        # splitting
        selected_feature, min_feature_threshold, min_feature_cost = self._best_feature(data, feature_idxs)
        
        # Split data based on best split
        left_data = data[data[:, selected_feature] < min_feature_threshold]
        right_data = data[data[:, selected_feature] >= min_feature_threshold]

        # Create child nodes
        left_node = self._grow(left_data, feature_idxs, depth=depth+1)
        right_node = self._grow(right_data, feature_idxs, depth=depth+1)

        return Node(left_node, 
                    right_node,
                    data, 
                    selected_feature, 
                    min_feature_threshold, 
                    feature_name=self.feature_names[selected_feature] if any(self.feature_names) else "NA",
                    criteria_value = criteria_value,
                    n_sample = len(data), _result=result,
                    class_name=class_name)
    
    #DFS
    def _traverse(self, node: Node):
        if node is None:
            return []

        # Process the current node
        # ... do something with the node ...

        # Recursively traverse the left and right subtrees
        left_nodes = self._traverse(node.left)
        right_nodes = self._traverse(node.right)

        # Combine the nodes from the current node, left subtree, and right subtree
        nodes = [node] + left_nodes + right_nodes

        return nodes
    
    #BFS
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
    
    def get_tree_structure(self):
        return self._traverse(self.root)
    
    def get_tree_structure_bfs(self):
        return self._traverse_bfs(self.root)
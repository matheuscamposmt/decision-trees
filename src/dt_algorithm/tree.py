from dt_algorithm.utils import mean_adjacent, cost_function, gini_impurity, get_majority
import numpy as np
from .node import Node, LeafNode

EPSILON = np.finfo('double').eps
class DecisionTree:
    def __init__(self, max_depth=None, min_samples_leaf=4, feature_names=[], class_names=[]):
        Node.count = 0
        self.root: Node = Node(None, None)
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.feature_names = feature_names
        self.classes = []
        self.class_names = class_names
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        
        self.root = self._grow(np.hstack((X, y)), np.arange(X.shape[-1]))
    
    def predict(self, X):
        return np.apply_along_axis(self.root.predict, arr=X, axis=1)
    
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
    
    def _best_split(self, feature_data: np.ndarray, labels: np.ndarray, thresholds):
        split_costs = []
        # for each threshold
        for thresh in thresholds:
            left_data, right_data = self._split_data(feature_data, labels, thresh)

            left_labels = left_data[1]
            right_labels = right_data[1]

            # calculate the cost for the feature with the specific threshold
            cost = cost_function(left_labels, right_labels, 
                                self.classes)
            
            split_costs.append(cost)
        
        min_idx = np.argmin(split_costs)
        min_split_cost = split_costs[min_idx]
        min_thresh = thresholds[min_idx]
        
        return min_thresh, min_split_cost
    

    def _best_feature(self, data, feature_idxs):
        feature_costs = []
        min_thresholds = []
        # for each feature in a list of the features index
        for feature_idx in feature_idxs:

            # get the feature data
            feature_data = data[:, feature_idx]
            labels = data[:, -1]

            # generate the thresholds
            thresholds = mean_adjacent(np.unique(np.sort(feature_data)), window_size=2)
            min_thresh, min_split_cost = self._best_split(feature_data, labels, thresholds)

            feature_costs.append(min_split_cost)
            min_thresholds.append(min_thresh)
        
        min_idx = np.argmin(feature_costs)

        min_feature_cost = feature_costs[min_idx]
        min_feature_threshold = min_thresholds[min_idx]
        selected_feature = feature_idxs[min_idx]

        return selected_feature, min_feature_threshold, min_feature_cost
        

    def _grow(self, data, feature_idxs, depth=1):
        data_gini_impurity = gini_impurity(data[:, -1])
        major_class = get_majority(self.classes, data[:, -1])

        # Stopping criteria
        if self.max_depth and depth >= self.max_depth:
            #print(f"Limit depth reached: {depth}. Number of samples: {len(data)}")
            return LeafNode(data, gini=data_gini_impurity, 
                            _class = major_class, class_name=self.class_names[major_class])
        if self.min_samples_leaf and (len(data) < self.min_samples_leaf):
            #print(f"Data with {len(data)} samples, returning LeafNode with depth {depth}")
            return LeafNode(data, gini=data_gini_impurity, 
                            _class=major_class, class_name=self.class_names[major_class])
        if data_gini_impurity < EPSILON:
            return LeafNode(data, gini=data_gini_impurity, 
                            _class=major_class,class_name=self.class_names[major_class])

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
                    selected_feature, 
                    min_feature_threshold, 
                    feature_name=self.feature_names[selected_feature] if any(self.feature_names) else None,
                    gini_value = data_gini_impurity,
                    n_sample = len(data), _class=major_class,
                    class_name=self.class_names[major_class])
    
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
    
    def get_tree_structure(self):
        return self._traverse(self.root)
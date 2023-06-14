import numpy as np
from typing import Iterable
from utils import proportions, gini_impurity


class Node:
    count=0
    def __init__(self, left, right, feature_idx=None, 
                 threshold=None, feature_name=None, 
                 gini_value=None, n_sample=None, _class=None):
        self.id=Node.count
        Node.count += 1

        self.left: Node = left
        self.right: Node = right
        self.feature_idx = feature_idx
        self.feature_name = feature_name
        self.threshold = threshold
        self.gini= gini_value
        self.n_sample = n_sample
        self._class = _class

    def predict(self, x: np.ndarray) -> float:
        if x[self.feature_idx] < self.threshold:
            return self.left.predict(x)
        
        elif x[self.feature_idx] >= self.threshold:
            return self.right.predict(x)
        
        else:
            raise TypeError("Invalid type in input.")
        

class LeafNode(Node):
    def __init__(self, data: Iterable, gini=None, _class=None):
        super().__init__(None, None)
        self.data = data
        self.classes = np.unique(data[:, -1])
        self._class = _class
        self.feature_idx = self._class
        self.gini = gini
        self.n_sample=len(data)

    def predict(self, X=None):
        return self._class
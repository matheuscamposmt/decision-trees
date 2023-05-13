import numpy as np
from typing import Iterable, List
from utils import proportion


class Node:
    def __init__(self, left, right, feature_idx=None, threshold=None):
        self.left: Node = left
        self.right: Node = right
        self.feature_idx = feature_idx
        self.threshold = threshold

    def predict(self, x: np.ndarray) -> float:
        if x[self.feature_idx] < self.threshold:
            return self.left.predict(x)
        
        elif x[self.feature_idx] >= self.threshold:
            return self.right.predict(x)
        
        else:
            raise TypeError("Invalid type in input.")
        

class LeafNode(Node):
    def __init__(self, data):
        super().__init__(None, None, None, None)
        self.data = data

    def predict(self, X=None):
        labels = self.data[:, -1]
        classes = np.unique(labels,)
        probs = []
        for _class in classes:
            probs.append(proportion(_class, self.data[:, -1]))

        return int(classes[np.argmax(probs)])
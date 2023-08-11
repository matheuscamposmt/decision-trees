class Node:
    count=0
    def __init__(self, left, right, data=None, feature_idx=None, 
                 threshold=None, feature_name=None, 
                 criteria_value=None, n_sample=None, _result=None, 
                 class_name=None):
        self.id=Node.count
        Node.count += 1
    
        self.left: Node = left
        self.right: Node = right

        self.data = data
        self.feature_idx: int = feature_idx
        self.feature_name: str = feature_name
        self.threshold: float = threshold
        self.criteria_value: float = criteria_value
        self.n_sample: int = n_sample
        self._class = _result
        self.class_name: str = class_name

    def predict(self, x):
        if x[self.feature_idx] < self.threshold:
            return self.left.predict(x)
        
        elif x[self.feature_idx] >= self.threshold:
            return self.right.predict(x)

class LeafNode(Node):
    def __init__(self, data, criteria_value=None, _result=None, class_name=None):
        super().__init__(None, None)
        self.data = data
        self._result = _result
        self.criteria_value = criteria_value
        self.n_sample=len(data)
        self.class_name=class_name

    def predict(self, X=None):
        return self._result
    
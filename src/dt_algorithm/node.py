class Node:
    count=0
    def __init__(self, left, right, data=None, feature_idx=None, 
                 threshold=None, feature_name=None, 
                 criterion_value=None, _result=None, 
                 class_name=None):
        self.id=Node.count
        Node.count += 1
    
        self.left: Node = left
        self.right: Node = right

        self.data = data
        self.feature_idx: int = feature_idx
        self.feature_name: str = feature_name
        self.threshold: float = threshold
        self.criterion_value: float = criterion_value
        self.n_sample: int = len(data) if data is not None else 0
        self._result = _result
        self.class_name: str = class_name

    def predict(self, x):
        if x[self.feature_idx] < self.threshold:
            return self.left.predict(x)
        
        elif x[self.feature_idx] >= self.threshold:
            return self.right.predict(x)

class LeafNode(Node):
    def __init__(self, data, criterion_value, _result, class_name):
        super().__init__(None, None, data=data, 
                         criterion_value=criterion_value, 
                         _result=_result, class_name=class_name)

    def predict(self, X=None):
        return self._result
    
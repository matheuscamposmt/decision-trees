from tree import DecisionTree
from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np

iris = load_iris()
X, y = iris.data, iris.target.reshape(-1 ,1) # type: ignore

dtree = DecisionTree(max_depth=5)

dtree.fit(X, y)

mypred= dtree.predict(X)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

skpred = clf.predict(X)
diff = np.where(mypred != skpred)
print(f"My prediction: {diff[0]} -> {mypred[diff]} \nsklearn prediction:{diff[0]} -> {skpred[diff]}")
      

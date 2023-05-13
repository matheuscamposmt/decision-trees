# Decision Tree from scratch using Python and NumPy
![Decision Tree Illustration](https://www.xoriant.com/cdn/ff/weqpbrtpXGjLpVQ_X-gWqsFlvjAxpv5Wv3xNW0A4vuQ/1602007254/public/2020-10/a-decisionTreesforClassification-AMachineLearningAlgorithm.jpg)

This is an implementation of the Decision Tree algorithm from scratch using the CART (Classification and Regression Trees) algorithm, which is a widely used algorithm for building decision trees. This implementation is written in Python and uses NumPy for numerical computations.

## Overview of the Decision Tree Algorithm

The decision tree algorithm builds a tree-like model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility. It is used for both classification and regression tasks. The algorithm recursively splits the data into subsets based on the values of a selected feature, creating decision nodes. At each node, the feature is selected that provides the best split of the data, and this process is repeated until a stopping criterion is met. The resulting tree is then used to make predictions on new data.

## CART Algorithm

The Classification and Regression Trees (CART) algorithm is a decision tree algorithm that can be used for both classification and regression tasks. The algorithm recursively partitions the data into subsets based on the value of a single feature, using the Gini impurity measure for classification or the sum of squared errors for regression. The feature and threshold that result in the lowest impurity or error are selected for the split, and the process is repeated until a stopping criterion is met.

## Implementation Details

This implementation of the Decision Tree algorithm uses the CART algorithm for splitting the data at each node. The `DecisionTree` class has two main methods, `fit` and `predict`. The `fit` method takes a numpy array `X` of input features and a numpy array `y` of corresponding target values, and trains the decision tree. The `predict` method takes a numpy array `X` of input features and returns the predicted target values.

The `DecisionTree` class uses a `Node` class to represent internal nodes of the tree and a `LeafNode` class to represent the leaf nodes. The `Node` class contains a left and a right child node, a selected feature index, and a threshold value for splitting the data at the node. The `LeafNode` class contains the target value for the leaf node.

The `_grow` method is the main recursive function that builds the decision tree. It takes the training data and a list of feature indices, and returns the root node of the decision tree. The `_split_data` method splits the data into left and right subsets based on the selected feature and threshold.

The implementation includes some hyperparameters that can be adjusted to control the growth of the tree. These include `max_depth`, `min_samples_leaf`.

## Conclusion

This implementation of the Decision Tree algorithm from scratch using the CART algorithm demonstrates the basic workings of a decision tree and provides a foundation for more complex tree-based models. It is a useful tool for both classification and regression tasks and can be adapted to a wide range of applications.

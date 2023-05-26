# Decision Tree from Scratch using Python and NumPy

![Decision Tree Illustration](https://www.xoriant.com/cdn/ff/weqpbrtpXGjLpVQ_X-gWqsFlvjAxpv5Wv3xNW0A4vuQ/1602007254/public/2020-10/a-decisionTreesforClassification-AMachineLearningAlgorithm.jpg)

This repository contains an implementation of the Decision Tree algorithm from scratch using the CART (Classification and Regression Trees) algorithm. The decision tree is a powerful algorithm widely used for building models that can handle both classification and regression tasks. This implementation is written in Python and utilizes the NumPy library for efficient numerical computations.

## Overview of the Decision Tree Algorithm

The Decision Tree algorithm constructs a tree-like model of decisions and their potential outcomes, incorporating chance event results, resource costs, and utility considerations. It recursively partitions the data into subsets based on the values of selected features, creating decision nodes. At each node, the algorithm identifies the feature that offers the optimal split of the data, and this process continues until a stopping criterion is met. The resulting tree can then be used to make predictions on new, unseen data.

### Decision Tree Algorithm Illustration

Here is an illustration of the Decision Tree algorithm:

![Decision Tree Algorithm](https://www.saedsayad.com/images/Decision_Tree_1.png)

## CART Algorithm

The Classification and Regression Trees (CART) algorithm is a popular decision tree algorithm that can handle both classification and regression tasks. The CART algorithm partitions the data recursively, selecting a single feature and threshold to split the data based on either the Gini impurity measure for classification or the sum of squared errors for regression. This selection process is repeated until a stopping criterion is satisfied.

### CART Algorithm Process

The following table summarizes the process of the CART algorithm:

| Step | Description |
|------|-------------|
| 1.   | Select the best feature to split the data based on a specific criterion (Gini impurity or sum of squared errors). |
| 2.   | Partition the data based on the selected feature and threshold. |
| 3.   | Repeat steps 1 and 2 recursively for each subset until a stopping criterion is met. |
| 4.   | Create decision nodes and leaf nodes based on the splits. |
| 5.   | Assign the majority class (classification) or the mean value (regression) to the leaf nodes. |

## Implementation Details

This implementation of the Decision Tree algorithm employs the CART algorithm for data partitioning at each node. The main class, `DecisionTree`, consists of two essential methods: `fit` and `predict`. The `fit` method accepts a numpy array `X` containing the input features and a numpy array `y` containing the corresponding target values. It then trains the decision tree using the provided data. On the other hand, the `predict` method accepts a numpy array `X` of input features and returns the predicted target values.

The `DecisionTree` class utilizes a `Node` class to represent internal nodes of the decision tree and a `LeafNode` class to represent the leaf nodes. The `Node` class contains attributes such as left and right child nodes, a selected feature index, and a threshold value for data partitioning at the node. Conversely, the `LeafNode` class stores the target value for the leaf node.

The main recursive function responsible for building the decision tree is `_grow`. This function takes the training data and a list of feature indices as inputs and returns the root node of the decision tree. Additionally, the `_split_data` method is used to split the data into left and right subsets based on the selected feature and threshold.

### Decision Tree Example

Here is an example of a decision tree:

![Decision Tree Example](https://miro.medium.com/v2/resize:fit:720/format:webp/1*fGX0_gacojVa6-njlCrWZw.png)

This decision tree is built using the CART algorithm and can be used to make predictions on new data.

## Conclusion

This implementation showcases the Decision Tree algorithm built from scratch using the CART algorithm. It provides a fundamental understanding of the inner workings of decision trees and serves as a solid foundation for more advanced tree-based models. With its versatility in handling both classification and regression tasks, the Decision Tree algorithm can be customized and applied to a wide range of real-world applications.

For more information and usage examples, please refer to the [Documentation](docs/documentation.md).

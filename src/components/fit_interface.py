from dash import Input, Output, State, callback
import pickle
from typing import List
import numpy as np
from sklearn.datasets import load_iris
from dt_algorithm import Node
from dt_algorithm import DecisionTree


def load_dataset(name: str, as_frame=False):
    if name.lower() == 'iris':
        data = load_iris(as_frame=as_frame)
        return data

    return np.loadtxt("data/" + name + ".csv", delimiter=",")


@callback(
    Output("fit-button", "disabled"),
    [Input("fit-button", "n_clicks")],
    [State("dataset-dropdown", "value"), State("max-depth-input", "value"), State("min-samples-split-input", "value")],
)
def fit_tree(n_clicks, dataset_name: str, max_depth: int, min_samples_leaf: int) -> List[Node]:
    if n_clicks == 0:
        return False

    # Load the dataset
    data = load_dataset(dataset_name)

    X, y = data.data, data.target.reshape(-1 ,1)
    # Create a decision tree.
    tree = DecisionTree(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                        feature_names=data.feature_names, class_names=data.target_names)
    tree.fit(X, y)

    root = tree.root
    filename = "root.pkl"
    with open(filename, 'wb') as root_file:
        pickle.dump(root, root_file)

    return False


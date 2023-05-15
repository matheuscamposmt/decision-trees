import dash
from dash import html, dcc, Input, Output, State
import numpy as np
from node import Node, LeafNode
from tree import DecisionTree
from sklearn.datasets import load_iris

app = dash.Dash(__name__)
# Create a dropdown menu to select the dataset to use.
dataset_options = [
    {'label': "Iris", 'value': "iris"},
    {'label': "Wine", 'value': "wine"},
    {'label': "Breast Cancer" ,'value': "breast_cancer"}
]

dataset_dropdown = dcc.Dropdown(
    id="dataset-dropdown",
    options=dataset_options,
    value="iris",
)

# Create a text input to enter the maximum depth of the tree.
max_depth_input = dcc.Input(
    id="max-depth-input",
    type="number",
    value=None,
)

# Create a text input to enter the minimum number of samples required to create a leaf node.
min_samples_leaf_input = dcc.Input(
    id="min-samples-leaf-input",
    type="number",
    value=None,
)

# Create a button to fit the tree to the dataset.
fit_button = html.Button(
    "Fit",
    id="fit-button",
    n_clicks=0
)
# Create a table to display the tree structure.
tree_table = dash.dash_table.DataTable(
    id="tree-table",
    columns=[
        {"name": "Feature", "id": "feature"},
        {"name": "Threshold", "id": "threshold"},
        {"name": "Left Child", "id": "left_child"},
        {"name": "Right Child", "id": "right_child"},
    ],
    data=[],
)

# Create a layout for the app.
app.layout = html.Div(
    [
        html.H1("Decision Tree Visualization"),
        html.Div(
            [
                dataset_dropdown,
                max_depth_input,
                min_samples_leaf_input,
                fit_button,
                tree_table,
            ],
            style={"width": "50%"},
        ),
    ]
)

# Create a callback to fit the tree to the dataset when the fit button is clicked.
@app.callback(
    Output("tree-table", "data"),
    [Input("fit-button", "n_clicks")],
    [State("dataset-dropdown", "value"), State("max-depth-input", "value"), State("min-samples-leaf-input", "value")],
)
def fit_tree(n_clicks, dataset_name, max_depth, min_samples_leaf):
    if n_clicks == 0:
        return []
    data = None

    # Load the dataset.
    if dataset_name.lower() == "iris":
        data=load_iris()

    else:
        np.loadtxt("data/" + dataset_name + ".csv", delimiter=",")

    X, y = data.data, data.target.reshape(-1 ,1)
    # Create a decision tree.
    tree = DecisionTree(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    tree.fit(X, y)

    # Get the tree structure.
    tree_structure = tree.get_tree_structure()
    # Convert the tree structure to a list of dictionaries.
    tree_data = []
    for node in tree_structure:
        tree_data.append(
            {
                "Feature": node.feature_idx,
                "Threshold": node.threshold,
                "Left Child": node.left.id if node.left is not None else None,
                "Right Child": node.right.id if node.right is not None else None,
            }
        )

    print(tree_data)

    return tree_data

# Run the app.
app.run_server(debug=True)

import dash
from dash import html, dcc, Input, Output, State
import plotly.graph_objects as go
import numpy as np
import math
from node import Node, LeafNode
from tree import DecisionTree
from sklearn.datasets import load_iris
from typing import List
import pickle

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

graph = dcc.Graph(id='tree-graph')

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
                graph,
                html.Button('Play', id='play-button', n_clicks=0),
                dcc.Interval(id='animation-interval', interval=1000, n_intervals=0),
                dcc.Store(id='tree_root_filename')
            ],
            style={"width": "50%"},
        )
    ]
)

@app.callback(
    Output("tree_root_filename", "data"),
    [Input("fit-button", "n_clicks")],
    [State("dataset-dropdown", "value"), State("max-depth-input", "value"), State("min-samples-leaf-input", "value")],
)
def fit_tree(n_clicks, dataset_name: str, max_depth: int, min_samples_leaf: int) -> List[Node]:
    if n_clicks == 0:
        return None

    data = None

    # Load the dataset.
    if dataset_name.lower() == "iris":
        data = load_iris()

    else:
        data = np.loadtxt("data/" + dataset_name + ".csv", delimiter=",")

    X, y = data.data, data.target.reshape(-1 ,1)
    # Create a decision tree.
    tree = DecisionTree(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    tree.fit(X, y)

    root = tree.root
    filename="root.pkl"
    with open(filename, 'wb') as root_file:
        pickle.dump(root, root_file)

    return filename


# Define the callback function to update the tree visualization
@app.callback(
    Output('tree-graph', 'figure'),
    Output('play-button', 'disabled'),
    [Input('play-button', 'n_clicks')],#, Input('animation-interval', 'n_intervals')],
    State('tree_root_filename', 'data')
)
def update_tree(n_clicks, tree_filename: str):
    if n_clicks == 0:
        return dash.no_update, dash.no_update
    #if n_intervals == 0:
    #    return dash.no_update, dash.no_update
    
    def count_nodes(tree):
        if tree is None:
            return 0
        
        count = 1  # Start with the root node
        
        count += count_nodes(tree.left)  # Recursively count nodes in the left subtree
        count += count_nodes(tree.right)  # Recursively count nodes in the right subtree
    
        return count
    
    tree = None
    with open(tree_filename, 'rb') as tree_file:
        tree = pickle.load(tree_file)

    tree_size = count_nodes(tree)
    adj_list = [[] for _ in range(tree_size)]

    # Traversal algorithm to generate tree nodes in a list
    nodes = []
    queue = [(tree, 0, 20, 0)]

    node_i = 0
    while queue:
        node, level, x, node_id = queue.pop(0)
        nodes.append((node_id, node.feature_idx, level, x))
        
        if node.left:
            left_id = node_i + 1  # Assign a unique ID to the left child
            node_i += 1
            queue.append((node.left, level + 1, x - 15*(level+1), left_id))
            adj_list[node_id].append(left_id)  # Add a connection from the current node to the left child
        
        if node.right:
            right_id = node_i + 1  # Assign a unique ID to the right child
            node_i += 1
            queue.append((node.right, level + 1, x + 15*(level+1), right_id))
            adj_list[node_id].append(right_id)  # Add a connection from the current node to the right child
    
    # Create the tree visualization
    node_ids=[node[0] for node in nodes]
    labels = np.array([node[1] for node in nodes])
    levels = np.array([node[2] for node in nodes])
    x_s = np.array([node[3] for node in nodes])
    
    max_level = levels.max()
    inverted_levels = max_level - levels

    line_traces=[]
    for node_i, node_connections in enumerate(adj_list):
        for conn in node_connections:
            line_trace = go.Scatter(
                x=[x_s[node_i], x_s[conn]],
                y=[inverted_levels[node_i], inverted_levels[conn]],
                mode='lines',
                line=dict(color='green')
            )
            line_traces.append(line_trace)


    # Create scatter trace for the tree nodes
    scatter = go.Scatter(
            x=x_s, y=inverted_levels,
            mode='markers+text',
            marker=dict(size=25, color='black'),
            text=x_s,
            textposition="middle center",
            hovertemplate="%{text}<extra></extra>",
            textfont=dict(color='blue') 
        )
    
    # Create the plotly figure for the tree
    figure = go.Figure(data=line_traces+[scatter])
    figure.update_layout(
        showlegend=False,
        xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
        yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
        width=1200,
        height=900)
    
    return figure, True


# Create a callback to fit the tree to the dataset when the fit button is clicked.

# Run the app.
app.run_server(debug=True)

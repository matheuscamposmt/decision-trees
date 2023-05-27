import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State
import plotly.graph_objects as go
import numpy as np
import igraph
from node import Node
from tree import DecisionTree
from sklearn.datasets import load_iris
from typing import List
import pickle

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.ZEPHYR])

# Create a dropdown menu to select the dataset to use.
dataset_options = [
    {'label': "Iris", 'value': "iris"},
    {'label': "Wine", 'value': "wine"},
    {'label': "Breast Cancer", 'value': "breast_cancer"}
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
fit_button = dbc.Button("Fit", id="fit-button", color="primary", className="mr-3")

graph = dcc.Graph(id='tree-graph')

app.layout = dbc.Container(
    [
        html.H1("Decision Tree Visualization", className="text-center mt-5 mb-3"),
        dbc.Row(
            [
                dbc.Col(dataset_dropdown, width=4, className="text-center"),
                dbc.Col(max_depth_input, width=2, className="text-center"),
                dbc.Col(min_samples_leaf_input, width=2, className="text-center"),
                dbc.Col(fit_button, width=2, className="text-center"),
            ],
            className="mb-3",
            justify="center"  # Align the row contents in the center
        ),
        dbc.Row(
            dbc.Col(html.Button('Play', id='play-button', n_clicks=0), width=2, className="text-center"),
            justify="center"  # Align the column contents in the center
        ),
        dbc.Row(
            dbc.Col(graph, width=12, className="text-center"),
            justify="center"  # Align the column contents in the center
        ),
        dbc.Row(
            dbc.Col(dcc.Interval(id='animation-interval', interval=1000, n_intervals=0), width=2, className="text-center"),
            justify="center"  # Align the column contents in the center
        ),
        dbc.Row(
            dbc.Col(dcc.Store(id='tree_root_filename'), width=2, className="text-center"),
            justify="center"  # Align the column contents in the center
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
    tree = DecisionTree(max_depth=max_depth, min_samples_leaf=min_samples_leaf, feature_names=data.feature_names)
    tree.fit(X, y)

    root = tree.root
    filename = "root.pkl"
    with open(filename, 'wb') as root_file:
        pickle.dump(root, root_file)

    return filename


@app.callback(
    Output('tree-graph', 'figure'),
    Output('play-button', 'disabled'),
    [Input('play-button', 'n_clicks')],
    State('tree_root_filename', 'data')
)
def update_tree(n_clicks, tree_filename: str):
    if n_clicks == 0:
        return dash.no_update, dash.no_update

    def count_nodes(tree):
        if tree is None:
            return 0

        count = 1  # Start with the root node

        count += count_nodes(tree.left)  # Recursively count nodes in the left subtree
        count += count_nodes(tree.right)  # Recursively count nodes in the right subtree

        return count

    with open(tree_filename, 'rb') as tree_file:
        tree = pickle.load(tree_file)

    tree_size = count_nodes(tree)
    adj_list = [[] for _ in range(tree_size)]

    # Traversal algorithm to generate tree nodes in a list
    nodes = []
    levels = []
    labels = []
    thresholds = []
    costs = []
    queue = [(tree, 0, 0)]

    node_i = 0
    while queue:
        node, level, node_id = queue.pop(0)
        nodes.append(node)
        levels.append(level)
        labels.append(node.feature_name)
        thresholds.append(node.threshold)
        costs.append(node.gini_value)

        if node.left:
            left_id = node_i + 1  # Assign a unique ID to the left child
            node_i += 1
            queue.append((node.left, level + 1, left_id))
            adj_list[node_id].append(left_id)  # Add a connection from the current node to the left child

        if node.right:
            right_id = node_i + 1  # Assign a unique ID to the right child
            node_i += 1
            queue.append((node.right, level + 1, right_id))
            adj_list[node_id].append(right_id)  # Add a connection from the current node to the right child

    hoverdata = ["feature="+label+"\n threshold="+str(threshold)
                 for label, threshold in zip(labels, thresholds) if threshold]
    max_level = max(levels)

    G = igraph.Graph()
    # Add vertices to the graph
    for vertex, _ in enumerate(adj_list):
        G.add_vertex(vertex)

    # Add edges to the graph
    for vertex, neighbors in enumerate(adj_list):
        for neighbor in neighbors:
            G.add_edge(vertex, neighbor)

    # Calculate the 'rt' layout based on the levels
    layout = G.layout_reingold_tilford(mode="in", root=[0])
    coords = layout.coords
    x_s = [coord[0] for coord in coords]
    y_s = [max_level - coord[1] for coord in coords]

    line_traces = []
    for node_i, node_connections in enumerate(adj_list):
        for conn in node_connections:
            line_trace = go.Scatter(
                x=[x_s[node_i], x_s[conn]],
                y=[y_s[node_i], y_s[conn]],
                mode='lines',
                line=dict(color='black')
            )
            line_traces.append(line_trace)

    hoverdata = [
        f"<span style='color:brown'>Feature: {label}</span><br><span style='color:blue'>Threshold={threshold}</span><br><span style='color:green'>Gini={gini:.3f}</span>"
        for label, threshold, gini in zip(labels, thresholds, costs) if threshold
    ]
    print(labels)

    # Create scatter trace for the tree nodes
    scatter = go.Scatter(
        x=x_s, y=y_s,
        mode='markers',
        marker=dict(size=40, color='#1f77b4'),
        hoverinfo='text',
        text=hoverdata,
        hovertemplate="%{text}<extra></extra>",
        hoverlabel=dict(
            bgcolor="white"
        )
    )

    # Create the plotly figure for the tree
    figure = go.Figure(data=line_traces + [scatter])
    figure.update_layout(
        showlegend=False,
        xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
        yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': True},
        width=1200,
        height=900
    )

    return figure, False

if __name__ == "__main__":
    app.run_server(debug=True)
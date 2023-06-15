import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, dash_table
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import igraph
from node import Node
from tree import DecisionTree
from sklearn.datasets import load_iris
from typing import List
import pickle

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.ZEPHYR],
                )

# Create a dropdown menu to select the dataset to use.
dataset_options = [
    {'label': "Iris", 'value': "iris"},
#    {'label': "Wine", 'value': "wine"},
#    {'label': "Breast Cancer", 'value': "breast_cancer"}
]

dataset_dropdown = dcc.Dropdown(
    id="dataset-dropdown",
    options=dataset_options,
    value="iris",
)


# Create a button to fit the tree to the dataset.
fit_button = dbc.Button("Fit", id="fit-button", 
                        color="primary", className="mr-3", 
                        n_clicks=0)
show_button = dbc.Button("Show", id="show-button", 
                         color="secondary", className="btn btn-outline-primary", 
                         n_clicks=0)

# Create a slider choose the maximum depth of the tree.
max_depth_input = dbc.Input(
    type='number',
    id='max-depth-input',
    value=5
    )

# Create a text input to enter the minimum number of samples required to create a leaf node.
min_samples_leaf_input = dbc.Input(
    type='number',
    id='min-samples-leaf-input',
    value=4
)

hyperparameters_input = dbc.Form(
    dbc.Row(
        [
            dbc.Col(
                [dbc.Label("Maximum levels", html_for='max-depth-slider', className="mr-2"),
                max_depth_input]
            ),
            dbc.Col(
                [dbc.Label(
                "Minimum samples for a leaf node", 
                html_for='min-samples-leaf-slider', 
                className="mr-2"),
                min_samples_leaf_input]
            ),
            dbc.Col(
                [dbc.Label("Dataset", html_for='dataset-dropdown', className="mr-2"),
                    dataset_dropdown]
            ),
            dbc.Col(fit_button, width=1, class_name='mx-1'),
            dbc.Col(show_button, width=1, class_name='text-start')
        ], align="end"
    ), className='g-2'
)

data_table = dash_table.DataTable(id='data_table_viz', data=[])
graph = dcc.Graph(id='tree-graph')

first_card = dbc.Card(dbc.CardBody([hyperparameters_input, graph]), style={"height": 1000})
second_card = dbc.Card(dbc.CardBody([html.H5("Random sample with 20 observations from the data", className="text-center mt-5 mb-3"), data_table]))

app.layout = dbc.Container(
    [
        html.H1("Decision Tree Visualization", className="text-center mt-5 mb-3"),
        html.H4("Explore and understand decision trees with a interactive visualization", className="text-center text-muted"),
        dbc.Row(
            [dbc.Col(first_card), dbc.Col(second_card)],
            justify="start",
            #className="mt-5"  # Add className instead of class_name
        ),

        dcc.Store(id='tree_root_filename') # Align the column contents in the center
    ], fluid=True
)


def load_dataset(name: str, as_frame=False):
    if name.lower() == 'iris':
        data = load_iris(as_frame=as_frame)
        return data
    
    return np.loadtxt("data/" + name + ".csv", delimiter=",")

@app.callback(
    Output("tree_root_filename", "data"),
    [Input("fit-button", "n_clicks")],
    [State("dataset-dropdown", "value"), State("max-depth-input", "value"), State("min-samples-leaf-input", "value")],
)
def fit_tree(n_clicks, dataset_name: str, max_depth: int, min_samples_leaf: int) -> List[Node]:
    if n_clicks == 0:
        return None

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

    return filename


@app.callback(
    [Output("data_table_viz", "data"), Output("data_table_viz", "columns")],
    Input("show-button", "n_clicks"),
    State("dataset-dropdown", "value")
)
def show_data_table(n_clicks, dataset_name: str):
    if n_clicks == 0:
        return [], []
    dataset_loaded = load_dataset(dataset_name, as_frame=True)
    targets = dataset_loaded.target
    target_names =dataset_loaded.target_names
    df =dataset_loaded.data.sample(20)

    df["class"] = targets.apply(lambda class_: target_names[class_])

    columns = [{"name": str(col), "id": str(col)} for col in df.columns]

    return df.to_dict('records'),columns


@app.callback(
    Output('tree-graph', 'figure'),
    Output('show-button', 'disabled'),
    [Input('show-button', 'n_clicks')],
    State('tree_root_filename', 'data')
)
def update_tree(n_clicks, tree_filename: str):
    # Create the plotly figure for the tree
    figure = go.Figure()
    figure.update_layout(
        showlegend=False,
        xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
        yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': True},
        #width=900,
        height=900,
        autosize=True,
        margin=dict(l=20, r=20, t=30, b=20),
    )

    if n_clicks == 0:
        return figure, False

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

    nodes = []
    levels = []
    hoverdata = []

    # BFS to generate tree nodes in a list
    queue = [(tree, 0, 0)]
    id_counter = 0
    while queue:
        node, level, node_id = queue.pop(0)
        nodes.append(node)
        levels.append(level)

        hovertext =f"""<span style='color:brown'>Feature: {node.feature_name}</span>
        <br><span style='color:blue'>Threshold={node.threshold}</span>
        <br><span style='color:green'>Gini={node.gini:.4f}</span>
        <br><span style='color:orange'>Samples={node.n_sample}</span>
        <br><span style='color:black'>Class={node.class_name}</span>"""
        hoverdata.append(hovertext)

        parent_id = node_id
        if node.left:
            id_counter += 1
            queue.append((node.left, level + 1, id_counter))
            adj_list[parent_id].append(id_counter)  # Add a connection from the current node to the left child
        if node.right:
            id_counter += 1
            queue.append((node.right, level + 1, id_counter))
            adj_list[parent_id].append(id_counter)  # Add a connection from the current node to the right child
    
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

    figure.add_traces(line_traces + [scatter])

    return figure, False

if __name__ == "__main__":
    app.run_server(debug=True)
import dash
from dash import html, dcc, Input, Output, State
import plotly.graph_objects as go
import numpy as np
from node import Node, LeafNode
from tree import DecisionTree
from sklearn.datasets import load_iris
from typing import List

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
                graph
            ],
            style={"width": "50%"},
        ),
    ]
)

def fit_tree(dataset_name: str, max_depth: int, min_samples_leaf: int) -> List[Node]:
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

    return tree.root

def update_graph(root, feature_names=None) -> go.Figure:
    fig = go.Figure()

    frames = []
    def traverse(node, x, y):
        if node is None:
            return
        scatter_trace = go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(size=25, color='blue'),
            text=[f"Threshold: {node.threshold} \n Feature:{feature_names[node.feature_idx] if feature_names else node.feature_idx}"],
            textposition="middle center",
            hovertemplate="%{text}<extra></extra>",
            textfont=dict(color='blue') 
        )
        # Add current node to the figure
        #fig.add_trace(scatter_trace)

        # Determine the positions of child nodes
        if node.has_left_child():
            left_x = x - 5
            left_y = y - 5
            # Add a line connecting the current node and the left child
            line_trace = go.Scatter(
                x=[x, left_x],
                y=[y, left_y],
                mode='lines',
                line=dict(color='black')
            )
            #fig.add_trace(line_trace)

            frames.append(go.Frame(data=[scatter_trace, line_trace]))
            traverse(node.left, left_x, left_y)
            

        if node.has_right_child():
            right_x = x + 5
            right_y = y - 5
            # Add a line connecting the current node and the right child
            line_trace = go.Scatter(
                x=[x, right_x],
                y=[y, right_y],
                mode='lines',
                line=dict(color='green')
            )
            #fig.add_trace(line_trace)

            frames.append(go.Frame(data=[scatter_trace, line_trace]))
            traverse(node.right, right_x, right_y)



    # Starting position for the root node
    root_x = 100
    root_y = 200

    # Traverse the binary tree and update the figure
    traverse(root, root_x, root_y)

    fig.update_layout(
        showlegend=False,
        height=800,
        width=1000,
        updatemenus=[dict(type='buttons', showactive=False,
                          buttons=[dict(label='Play',
                                        method='animate',
                                        args=[None, {'frame': {'duration': 1000, 'redraw': True}, 'fromcurrent': True, 'transition': {'duration': 500}}]
                                        )]
                          )]
    )

    # Add frames to the figure
    fig.frames = frames

    return fig

# Create a callback to fit the tree to the dataset when the fit button is clicked.
@app.callback(
    Output("tree-graph", "figure"),
    [Input("fit-button", "n_clicks")],
    [State("dataset-dropdown", "value"), State("max-depth-input", "value"), State("min-samples-leaf-input", "value")],
)
def update_figure(n_clicks: int, dataset_name: str, max_depth: int, min_samples_leaf: int) -> go.Figure:
    if n_clicks == 0:
        fig = go.Figure([])
        fig.update_layout(
            showlegend=False,
            height=800,
            width=1000)
        return fig
    
    tree_structure = fit_tree(dataset_name, max_depth, min_samples_leaf)
    return update_graph(tree_structure)

# Run the app.
app.run_server(debug=True)

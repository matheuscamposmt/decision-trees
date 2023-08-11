import igraph
import plotly.graph_objects as go
import pickle
from dash import dcc, Input, Output, State, callback, callback_context
from components.fit_interface import load_dataset
import pandas as pd


def build_tree_viz(figure):
    with open('tree.pkl', 'rb') as tree_file:
        tree = pickle.load(tree_file)

    adj_list = dict()

    nodes = []
    levels = []
    hoverdata = []
    criterion = "Gini Impurity" if tree.tree_type == 'classification' else 'MSE'
    result_type = "Class" if tree.tree_type == 'classification' else "Value"
    # BFS to generate tree nodes in a list
    queue = [(tree.root, 0, 0)]
    id_counter = 0
    while queue:
        node, level, node_id = queue.pop(0)
        nodes.append(node)
        levels.append(level)

        hovertext =f"""<span style='color:red'>Condition: {node.feature_name} <= {node.threshold}</span>
                    <br><span style='color:green'>{criterion}={node.criterion_value:.4f}</span>
                    <br><span style='color:blue'>{result_type}={node.class_name}</span>
                    <br><span style='color:black'>Samples={node.n_sample}</span>"""
        
        # The reason I'm getting this error: TypeError: unsupported format string passed to NoneType.__format__
        

        if not (node.left or node.right):
            hovertext = f"""<span style='color:red'>PREDICTION NODE</span>
            <br><span style='color:green'>{criterion}={node.criterion_value:.4f}</span>
            <br><span style='color:blue;font-weight:bold'>{result_type}={node.class_name}</span>
            <br><span style='color:black'>Samples={node.n_sample}</span>"""
        
        hoverdata.append(hovertext)

        parent_id = node_id
        adj_list[parent_id] = list()
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
    for vertex in adj_list.keys():
        G.add_vertex(vertex)

    # Add edges to the graph
    for vertex, neighbors in adj_list.items():
        for neighbor in neighbors:
            G.add_edge(vertex, neighbor)

    # Calculate the 'rt' layout based on the levels
    layout = G.layout_reingold_tilford(mode="in", root=[0])
    coords = layout.coords
    x_s = [coord[0] for coord in coords]
    y_s = [max_level - coord[1] for coord in coords]

    line_traces = []
    annotations = []
    for node_i, node_connections in adj_list.items():
        for conn in node_connections:
            line_trace = go.Scatter(
                x=[x_s[node_i], x_s[conn]],
                y=[y_s[node_i], y_s[conn]],
                mode='lines',
                line=dict(color='black')
            )
            line_traces.append(line_trace)

            # add annotation
            x0 = x_s[node_i]
            x1 = x_s[conn]
            y0 = y_s[node_i]
            y1 = y_s[conn]
            x = (x0 + x1) / 2
            y = (y0 + y1) / 2
            annotations.append(
                dict(
                    x=x,
                    y=y,
                    xref="x",
                    yref="y",
                    text='False' if x0 < x1 else 'True',
                    showarrow=False,
                    font=dict(
                        family="Courier New, monospace",
                        size=16,
                        color="#ffffff"
                    ),
                    align="center",
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="#636363",
                    ax=0,
                    ay=-30,
                    bordercolor="#c7c7c7",
                    borderwidth=2,
                    borderpad=4,
                    bgcolor="#ff7f0e",
                    opacity=0.9
                )
            )
        
    # Create scatter trace for the tree nodes
    scatter = go.Scatter(
        x=x_s, y=y_s,
        mode='markers',
        fillcolor='#05004a',  # dark blue
        line=dict(color='white', width=2),
        marker=dict(size=40, color='#1f77b4'),
        hoverinfo='text',
        text=hoverdata,
        hovertemplate="%{text}<extra></extra>",
        # improve the hover aesthetics
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
        showlegend=False

    )


    return line_traces+[scatter], annotations

# TODO
def predict_tree(n_clicks, actual_figure):
    pass


@callback(
    Output('tree-graph', 'figure'),
    [Input('show-button', 'n_clicks'), Input('tree-graph', 'clickData'), Input('tree-graph', 'figure')]
)
def update_graph(show_clicks, click_data, figure):
    ctx = callback_context
    input_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if not figure:
        figure = go.Figure()
        figure.update_layout(
            showlegend=False,
            xaxis={'showgrid': False, 'zeroline': False, 'showticklabels': False},
            yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': True},
            autosize=True,
            height=800,
            width=1100,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        if show_clicks == 0:
            return figure
    if input_id == 'show-button':
        print('building tree viz...')
        graph_objects, annotations = build_tree_viz(show_clicks)
        figure['data'] = graph_objects
        figure['layout']['annotations'] =annotations

    # highlight the node clicked on
    elif input_id == 'tree-graph':
        point = click_data['points'][0]
        curves = figure['data']

    return figure



# create a callback function for the user to click on a node and see more information about it
@callback(
        Output('data-table', 'data'),
        [Input('show-button', 'n_clicks'),Input('tree-graph', 'clickData'), Input('data-table', 'data')],
        State('dataset-dropdown', 'value')
)
def update_datatable(show_clicks, click_data, data, dataset_name):
    ctx = callback_context
    input_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if input_id == 'show-button':
        if show_clicks == 0:
            return data
        else:
            data = load_dataset(dataset_name, as_frame=True)
            df = data.frame
            return df.to_dict('records')
        
    elif input_id == 'tree-graph':
        if click_data is None:
            return data
        else:
            return update_annotation(click_data, data)
    if click_data:
        return update_annotation(click_data, data)
    
    return data


def update_annotation(click_data, data):

    point = click_data['points'][0]
    point_index = point['pointIndex']
    with open('tree.pkl', 'rb') as tree_file:
        tree = pickle.load(tree_file)

    nodes = tree.get_tree_structure_bfs()
    node = nodes[point_index]

    subdata = node.data
    subdf = pd.DataFrame(subdata, columns=list(data[0].keys()))
    return subdf.to_dict('records')
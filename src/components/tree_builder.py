import igraph
import plotly.graph_objects as go
import pickle
from dash import dcc, Input, Output, State, callback, ctx


def build_tree(n_clicks):
    # Create the plotly figure for the tree
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
    if n_clicks == 0:
        return figure

    with open('tree.pkl', 'rb') as tree_file:
        tree = pickle.load(tree_file)


    adj_list = dict()

    nodes = []
    levels = []
    hoverdata = []
    criterion = "Gini Impurity" if tree.tree_type == 'classification' else 'MSE'
    # BFS to generate tree nodes in a list
    queue = [(tree.root, 0, 0)]
    id_counter = 0
    while queue:
        node, level, node_id = queue.pop(0)
        nodes.append(node)
        levels.append(level)

        hovertext =f"""<span style='color:red'>{node.feature_name} <= {node.threshold}</span>
                    <br><span style='color:green'>{criterion}={node.criteria_value:.4f}</span>
                    <br><span style='color:orange'>Class={node.class_name}</span>
                    <br><span style='color:black'>Samples={node.n_sample}</span>"""
        
        # The reason I'm getting this error: TypeError: unsupported format string passed to NoneType.__format__
        

        if not (node.left or node.right):
            hovertext = f"""<span style='color:red'>PREDICTION NODE</span>
            <br><span style='color:green'>{criterion}={node.criteria_value:.4f}</span>
            <br><span style='color:orange'>Class={node.class_name}</span>
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
    for node_i, node_connections in adj_list.items():
        for conn in node_connections:
            line_trace = go.Scatter(
                x=[x_s[node_i], x_s[conn]],
                y=[y_s[node_i], y_s[conn]],
                mode='lines+text',
                line=dict(color='black'),
                text="True" if x_s[node_i] > x_s[conn] else "False"
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

    return figure


def predict_tree(n_clicks, actual_figure):
    if n_clicks == 0:
        return None
    
    traces = actual_figure['data']
    traces.append(go.Scatter(x=[1], y=[1], mode='markers'))

    return {'data':traces, 'layout': actual_figure['layout']}


@callback(
    Output('tree-graph', 'figure'),
    Input('show-button', 'n_clicks'),
    Input('tree-graph', 'clickData'),
    [State('tree-graph', 'figure')]
)
def update_graph(show_clicks, clickData, actual_figure):
    triggered_id = ctx.triggered_id

    if triggered_id == 'show-button':
        return build_tree(show_clicks)
    
    elif triggered_id == 'tree-graph':
        return update_annotation(clickData, actual_figure)

    return build_tree(show_clicks)


# create a callback function for the user to click on a node and see more information about it

def update_annotation(click_data, actual_fig):

    if click_data is not None:
        point = click_data['points'][0]
        point_index = click_data['points'][0]['pointIndex']
        annotation = {
            'x': point['x'],
            'y': point['y'],  # Adjust the y-coordinate for the annotation position
            'text': f"Node number {point_index + 1}",
            'ax': 0,
            'ay': -40
        }

        actual_fig['layout']['annotations'] = [annotation]
        return actual_fig
    return actual_fig
from dash import dcc
from dash import dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc

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
# ---------------------------- BUTTONS ----------------------------- ##

# Create a button to fit the tree to the dataset.
fit_button = dbc.Button("Fit", id="fit-button", 
                        color="primary", className="mr-3", 
                        n_clicks=0)
show_button = dbc.Button("Show", id="show-button", 
                         color="secondary", className="btn btn-outline-primary", 
                         n_clicks=0, disabled=True)
predict_button = dbc.Button("Predict", id="predict-button",
                            color="secondary", className="btn btn-outline-primary",
                            n_clicks=0, disabled=True)

@callback(
    Output('show-button', 'disabled'),
    [Input('fit-button', 'n_clicks')]
)
def enable_button_1(n_clicks_fit):
    if n_clicks_fit > 0:
        return False
    else:
        return True


graph = dcc.Graph(id='tree-graph', config=dict(watermark=False))
max_depth_input = dbc.Input(
    type='number',
    id='max-depth-input',
    value=5
    )
min_samples_leaf_input = dbc.Input(
    type='number',
    id='min-samples-split-input',
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
                "Minimum samples for a node to split",
                html_for='min-samples-split-slider',
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
graph_card = dbc.Card(dbc.CardBody([hyperparameters_input, graph]), style={"height": 950, "width": 1200})


"""@callback(
    Output('tree-graph', 'figure'),
    
)
def show_predict(n_clicks, data_table):
    if n_clicks == 0:
        return graph.figure
    figure_data = graph.figure['data']

    if not any(figure_data):
        return graph.figure"""


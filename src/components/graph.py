from dash import dcc
from dash import dcc, Input, Output, State, callback, html, dash_table
import dash_bootstrap_components as dbc

dataset_options = [
    {'label': "Iris", 'value': "iris"},
    {'label': "Wine", 'value': "wine"},
    {'label': "Diabetes", 'value': "diabetes"}
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
show_button = dcc.Loading(
    id="show-button-loading",
    type="circle",
    children=[
        dbc.Button("Show", id="show-button", 
                   color="secondary", className="btn btn-outline-primary", 
                   n_clicks=0, disabled=True)
    ]
)
predict_button = dbc.Button("Predict", id="predict-button",
                            color="secondary", className="btn btn-outline-primary",
                            n_clicks=0, disabled=True)

graph = dcc.Graph(id='tree-graph', config=dict(watermark=False))

max_depth_input = dbc.Input(
    type='number',
    id='max-depth-input',
    value=5
    )
min_samples_split_input = dbc.Input(
    type='number',
    id='min-samples-split-input',
    value=4
)

min_samples_leaf_input = dbc.Input(
    type='number',
    id='min-samples-leaf-input',
    value=2
)
hyperparameters_input = dbc.Form(
    dbc.Row(
        [
            dbc.Col(
                [dbc.Label(
                "Maximum levels", 
                html_for='max-depth-input', 
                className="mr-2"),
                max_depth_input]
            ),
            dbc.Col(
                [dbc.Label(
                "Minimum samples for a node to split",
                html_for='min-samples-split-input',
                className="mr-2"),
                min_samples_split_input]
            ),
            dbc.Col(
                [dbc.Label(
                "Minimum samples for a leaf node",
                html_for='min-samples-leaf-input',
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

display_data = dbc.Row(
    [
        dcc.Markdown("""
            #### Data
            Click on the nodes in the graph to display its data.
        """, id='data-markdown'),

        dash_table.DataTable(id='data-table',
                             style_cell={'textAlign': 'left'},
                             style_header={'fontWeight': 'bold'},
                             style_table={'overflowX': 'auto'},
                             page_size=25)
    ]
)

graph_card = dbc.Card(dbc.CardBody([hyperparameters_input, graph]), style={"height": 950, "width": 1100})
display_card = dbc.Card(dbc.CardBody(display_data), style={"height": 950, "width": 600})


"""from dash import Input, Output, State, dash_table, html, callback
import dash_bootstrap_components as dbc
from components.fit_interface import load_dataset

data_table = dash_table.DataTable(id='data_table_viz', data=[])
table_card = dbc.Card(dbc.CardBody([html.H5("Random sample with 20 observations from the data", className="text-center mt-5 mb-3"), data_table]))

@callback(
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

    return df.to_dict('records'),columns"""
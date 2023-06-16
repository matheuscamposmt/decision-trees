import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from components import graph_card


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.ZEPHYR])
layout = dbc.Container(
    [
        html.H1("Decision Tree Visualization", className="text-center mt-5 mb-3"),
        html.H4("Explore and understand decision trees with a interactive visualization", className="text-center text-muted"),
        dbc.Row(
            [graph_card],
            justify="center",
            className="mt-5"  # Add className instead of class_name
        ),

         # Align the column contents in the center
    ], fluid=True
)

app.layout = layout
server= app.server

# Define the callback function to enable button_1
if __name__ == "__main__":
    app.run_server(debug=True)
from dash import dcc, html
from ..config import *
# import ..config

def generate_description_card():
    """

    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.H5("Visualization Group 19 Dashboard"),
            html.Div(
                id="intro",
                children="You can use this as a basic template for your JBI100 visualization project.",
            ),
        ],
    )


def generate_control_card():
    """

    :return: A Div containing controls for graphs.
    """
    return html.Div(
        id="control-card",
        children=[
            html.Label("Mapbox type"),
            dcc.Dropdown(
                id="select-mapbox-type",
                options=[{"label": i, "value": i} for i in data_type],
                value=data_type[0],
            ),
            html.Br(), #a little space between te dropdowns
            html.Label("Color scatterplot 2"),
            dcc.Dropdown(
                id="select-color-scatter-2",
                options=[{"label": i, "value": i} for i in color_list2],
                value=color_list2[0],
            ),
            html.Br(), #a little space between te dropdowns
            html.Label("Color scatterplot 3"),
            dcc.Dropdown(
                id="select-attribute",
                options=[{"label": i, "value": i} for i in attributes_list],
                value=attributes_list[0],
            ),
        ], style={"textAlign": "float-left"}
    )




def make_menu_layout():
    return [generate_control_card()]

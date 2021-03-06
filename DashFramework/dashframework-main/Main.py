import dash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
import dash_bootstrap_components as dbc
import sqlite3
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
import time

from DC2_app.views.menu import make_menu_layout
from DC2_app.views.scatterplot import Scatterplot
from DC2_app.views.scatterplot import Scatterplot
from DC2_app.data import get_amount_data
from DC2_app.data import get_subselection
from DC2_app.Models import get_ARIMA
from DC2_app.Models import get_RNN
from DC2_app.Models import get_Mape
from DC2_app.data import get_amount_of_crime_options_region_year_type_xaxis_month
from DC2_app.data import get_amount_of_crime_options_year_type_xaxis_region
from DC2_app.data import get_amount_of_crime_options_in_year_region_xaxis_CrimType


# Since we're adding callbacks to elements that don't exist in the app.layout,
# # Dash will raise an exception to warn us that we might be
# # doing something wrong.
# # In this case, we're adding the elements through a callback, so we can ignore
# # the exception.
con = sqlite3.connect('DC2.db')

query_regions = f'''SELECT * FROM Region'''
sql_query = pd.read_sql_query(query_regions, con)
Regions = pd.DataFrame(sql_query)

query_types = f'''SELECT * FROM Crime_Type'''
sql_query = pd.read_sql_query(query_types, con)
Types = pd.DataFrame(sql_query)

Year_distinct = pd.DataFrame(['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019'], columns=["Years"])

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(
    #top Header
    children=[
        html.Div(
            children=[
                html.P(children="🚗", style={"font-size": "48px", "margin": "0 auto", "text-align": "center"}),
                html.H1(
                    children="Crime Uncovered", style={"color": "#FFFFFF", "font-size": "48px", "font-weight": "bold",
                                                        "text-align": "center", "margin": "0 auto"}),
                html.P(children="Analyze the crime spread in england",
                       style={"color": "#CFCFCF", "margin": "4px auto",
                              "text-align": "center", "max-width": "384px"},
                       ),
            ], style={"background-color": "#222222", "height": "288px", "padding": "16px 0 0 0"}
        ),

        html.Div(
            children=[
                # Idea for this selection box originated from https://realpython.com/python-dash/, has been edited.
            html.Div(
                    children=[
                        html.Div(children="Region",
                                 style={"margin-bottom": "6px", "font-weight": "bold", "color": "#079A82"}),
                        dcc.Dropdown(id="Region_filter", style={"height": "48px", "width": "290px"},
                                     options=[
                                         {"label": Region, "value": Region}
                                         for Region in np.sort(list(Regions['Falls within']))
                                     ],
                                     value=[""],
                                     clearable=False,
                                     multi=False,
                                     className="dropdown",
                                     ),
                    ],
                ),
                html.Div(
                    children=[
                        html.Div(children="Crime_Type",
                                 style={"margin-bottom": "6px", "font-weight": "bold", "color": "#079A82"}),
                        dcc.Dropdown(
                            id="Crime_Type_filter", style={"height": "48px", "width": "290px"},
                            options=[
                                {"label": Type, "value": Type}
                                for Type in list(Types['Crime type'])
                            ],
                            value="",
                            clearable=False,
                            searchable=False,
                            className="dropdown",
                        ),
                    ],
                ),
                html.Div(
                    children=[
                        html.Div(children="Years",
                                 style={"margin-bottom": "6px", "font-weight": "bold", "color": "#079A82"}),
                        dcc.Dropdown(id="Year_filter", style={"height": "48px", "width": "290px"},
                                     options=[
                                         {"label": Year_distinct, "value": Year_distinct}
                                         for Year_distinct in np.sort(list(Year_distinct['Years']))
                                     ],
                                     value=[""],
                                     clearable=False,
                                     multi=False,
                                     className="dropdown",
                                     ),
                    ],
                ),
                html.Div([
                            html.Button('Load', id='Trigger', n_clicks=0, style={"margin-top": "38px"})
                    ],
                ),
            ], style={"height": "112px", "width": "1000px", "display": "flex", "justify-content": "space-evenly",
                      "padding-top": "24px", "margin": "-80px auto 0 auto", "background-color": "#FFFFFF",
                      "box-shadow": "0 4px 6px 0 rgba(0, 0, 0, 0.18)"},
        ),
        html.Div(
            children=[
                html.Div(id="Combined-container1", children=[
                    html.Div(id='ModelVisSARIMA', style={'display': 'inline-block', 'vertical-align': 'top', 'width': '90%', }),
                    html.Div(id='MAPE_ARIMA', style={'display': 'inline-block', 'vertical-align': 'top', 'width': '10%', })
                ], style={"margin": "10px auto 0 auto"}),
                html.Div(id="Combined-container2", children=[
                    html.Div(id='ModelVisRNN', style={'display': 'inline-block', 'vertical-align': 'top', 'width': '90%', }),
                    html.Div(id='MAPE_RNN', style={'display': 'inline-block', 'vertical-align': 'top', 'width': '10%', })
                ], style={"margin": "10px auto 0 auto"}),
                html.Div(id="Combined-container3", children=[
                    html.Div(id='Amount_Month', style={'display': 'inline-block', 'vertical-align': 'top', 'width': '40%', }),
                    html.Div(id='Amount_Region', style={'display': 'inline-block', 'vertical-align': 'top', 'width': '60%', }),
                ], style={"margin": "10px auto 0 auto"}),
                html.Div(id="Combined-container4", children=[
                    html.Div(id='Amount_Type', style={'display': 'inline-block', 'vertical-align': 'top', 'width': '100%', })
                ], style={"margin": "10px auto 0 auto"}),
            ],
        ),
    ], )


@app.callback(
    Output("ModelVisSARIMA", "children"),
    Output("ModelVisRNN", "children"),
    Output("MAPE_ARIMA", "children"),
    Output("MAPE_RNN", "children"),
    Output("Amount_Month", "children"),
    Output("Amount_Region", "children"),
    Output("Amount_Type", "children"),

    Input('Trigger', 'n_clicks'),
    State('Region_filter', 'value'),
    State('Crime_Type_filter', 'value'),
    State('Year_filter', 'value')
)
def update_charts(n_clicks, Region, Type, Year):
    if Region != "" and Type != "" and Year != "":
        df_amount = get_amount_data(Type, Region)
        train, test, predicted = get_ARIMA(df_amount)
        train_RNN, test_RNN, predicted_RNN = get_RNN(df_amount)
        #df_Subselect = get_subselection(Type, Region)

        # scatterplot showing number of casualties
        fig = go.Figure()
        fig.update_layout(title="SARIMA")
        fig.add_trace(go.Scatter(x=list(train.reset_index()["Date"]) , y=train["Crime_Number"],
                                 mode='lines+markers', name='Train'))
        fig.add_trace(go.Scatter(x=list(test.reset_index()["Date"]) , y=test["Crime_Number"],
                                 mode='lines+markers', name='Test'))
        fig.add_trace(go.Scatter(x=list(predicted.reset_index()["Date"]) , y=predicted["Crime_Number"],
                                 mode='lines+markers', name='Predicted'))

        fig1 = go.Figure()
        fig1.update_layout(title="RNN")
        fig1.add_trace(go.Scatter(x=list(train_RNN.reset_index()["Date"]) , y=train_RNN["Crime_Number"],
                                 mode='lines+markers', name='Train'))
        fig1.add_trace(go.Scatter(x=list(test_RNN.reset_index()["Date"]) , y=test_RNN["Crime_Number"],
                                 mode='lines+markers', name='Test'))
        fig1.add_trace(go.Scatter(x=list(predicted_RNN.reset_index()["Date"]) , y=predicted_RNN["Crime_Number"],
                                 mode='lines+markers', name='Predicted'))

        fig2 = px.bar(get_amount_of_crime_options_region_year_type_xaxis_month(Region, Year, Type), x='Month',
                      y='COUNT(`Crime type`)',
                      labels={
                          "COUNT(`Crime type`)": "Amount of crime",
                          "Month": "Month",
                      },
                      title=f"{Region}, {Year}, {Type}")

        fig3 = px.bar(get_amount_of_crime_options_in_year_region_xaxis_CrimType(Year, Region), x='Crime type',
                      y='COUNT(`Crime type`)',
                      labels={
                          "COUNT(`Crime type`)": "Amount of crime",
                          "Crime type": "Crime type",
                      },
                      title=f"{Region}, {Year}")

        fig4 = px.bar(get_amount_of_crime_options_year_type_xaxis_region(Year, Type), x='Falls within',
                      y='COUNT(`Crime type`)',
                      labels={
                          "COUNT(`Crime type`)": "Amount of crime",
                          "Falls within": "Region",
                      },
                      title=f"{Year}, {Type}")
    return dcc.Graph(figure=fig), dcc.Graph(figure=fig1),\
               "MAPE: " + str(get_Mape(test, predicted)),\
               "MAPE: " + str(get_Mape(test_RNN, predicted_RNN)), \
               dcc.Graph(figure=fig2), dcc.Graph(figure=fig3), dcc.Graph(figure=fig4)


if __name__ == '__main__':
    app.run_server(debug=True)

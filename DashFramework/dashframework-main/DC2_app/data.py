import plotly.express as px
import pandas as pd
import numpy as np
import time
import sqlite3

def get_amount_data(Selected_Crime_Types, Selected_Regions):
    con = sqlite3.connect('DC2_DB.db')
    query = f'''SELECT Year, Month, `Falls within`, COUNT(*) AS "Crime_Number"
                FROM Street_fixed
                WHERE `Crime type` = "{Selected_Crime_Types}" AND `Falls within` = "{Selected_Regions}"
                GROUP BY Year, Month, `Falls within`
             '''
    sql_query = pd.read_sql_query(query, con)
    df = pd.DataFrame(sql_query)
    return df

def get_subselection(Selected_Crime_Types, Selected_Regions):
    con = sqlite3.connect('DC2_DB.db')
    query = f'''SELECT *
                FROM Street_fixed
                WHERE `Crime type` = "{Selected_Crime_Types}" AND `Falls within` = "{Selected_Regions}"
             '''
    sql_query = pd.read_sql_query(query, con)
    df = pd.DataFrame(sql_query)
    return df
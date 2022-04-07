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


def get_amount_of_crime_options_region_year_type_xaxis_month(Selected_Region, Selected_Year, Selected_Crime_type):
    con = sqlite3.connect('DC2_DB.db')
    sql_query = f'''
    SELECT COUNT(`Crime type`), Month, `Falls within`
    FROM Street_fixed
    WHERE `Falls within`="{Selected_Region}" AND Year={Selected_Year} AND `Crime type`={Selected_Crime_type}
    GROUP BY Month
    '''
    sql_query = pd.read_sql_query(sql_query, con)
    df = pd.DataFrame(sql_query)
    return df    

def get_amount_of_crime_options_year_type_xaxis_region(Selected_Year, Selected_Crime_type):
    con = sqlite3.connect('DC2_DB.db')
    sql_query = f'''
    SELECT COUNT(`Crime type`), Month, `Falls within`, Year, `Crime type`
    FROM Street_fixed
    WHERE  Year={Selected_Year} AND `Crime type`="{Selected_Crime_type}"
    GROUP BY `Falls within`
    '''
    sql_query = pd.read_sql_query(sql_query, con)
    df = pd.DataFrame(sql_query)
    return df   

def get_amount_of_crime_options_in_year_region_xaxis_CrimType(Selected_Year, Selected_Region):
    con = sqlite3.connect('DC2_DB.db')
    sql_query = f'''
    SELECT COUNT(`Crime type`), Month, `Falls within`, Year, `Crime type`
    FROM Street_fixed
    WHERE  Year={Selected_Year} AND `Falls within`="{Selected_Region}"
    GROUP BY `Crime type`
    '''
    sql_query = pd.read_sql_query(sql_query, con)
    df = pd.DataFrame(sql_query)
    return df   


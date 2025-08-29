import numpy as np
import pandas as pd


def preprocess(df, date_col_name='Date', target_col_name='Close/Last'):
    """
    Takes in a time series (pd.Dataframe) of stock prices and processes it for further evaluation.
    The following steps are taken:
    - Remove unnecessary columns.
    - Replace currency symbols.
    - Introduce the dates as the index of the pd.Dataframe.
    """
    df = df[[date_col_name, target_col_name]].copy()

    df[target_col_name] = df[target_col_name].replace({r'\$': ''}, regex=True).astype(float)

    df[date_col_name] = pd.to_datetime(df[date_col_name], format='%m/%d/%Y')
    df = df.set_index(date_col_name).sort_index()

    return df


def take_log(df, target_col_name='Close/Last'):
    """
    Takes the logs of a column.
    Useful to reduce variance and improve forecasts.
    """
    df[target_col_name] = np.log(df[target_col_name])

    return df

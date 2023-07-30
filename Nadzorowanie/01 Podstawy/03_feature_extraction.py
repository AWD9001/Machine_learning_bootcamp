# Import bibliotek
import numpy as np
import pandas as pd
import sklearn

# Załadowanie danych


def fetch_financial_data(company='AMZN'):
    """
    This function fetches stock market quotations.
    """
    import pandas_datareader.data as web
    return web.DataReader(name=company, data_source='stooq')


df_raw = fetch_financial_data()
df_raw.head()

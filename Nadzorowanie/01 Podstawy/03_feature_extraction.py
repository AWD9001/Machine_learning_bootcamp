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
print(df_raw.head())

# Utworzenie kopii danych
df = df_raw.copy()
df = df[:5]
print(df.info())

# Generowanie nowych zmiennych
print(df.index.month)
df['day'] = df.index.day
df['month'] = df.index.month
df['year'] = df.index.year
print(df)

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

# Dyskretyzacja zmiennej ciągłej

df = pd.DataFrame(data={'height': [175., 178.5, 185., 191., 184.5, 183., 168.]})
print(df)

df['height_cat'] = pd.cut(x=df.height, bins=3)
print(df)

df['height_cat'] = pd.cut(x=df.height, bins=(160, 175, 180, 195))
print(df)

df['height_cat'] = pd.cut(x=df.height, bins=(160, 175, 180, 195), labels=['small', 'medium', 'high']
                          )
print(df)

pd.get_dummies(df, drop_first=True, prefix='height')
print(df)

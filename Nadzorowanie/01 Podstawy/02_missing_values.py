# Import bibliotek
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import plotly.express as px
import seaborn as sns
import sklearn
from sklearn.impute import SimpleImputer

print(sklearn.__version__)

# Wygenerowanie danych
data = {
    'size': ['XL', 'L', 'M', np.nan, 'M', 'M'],
    'color': ['red', 'green', 'blue', 'green', 'red', 'green'],
    'gender': ['female', 'male', np.nan, 'female', 'female', 'male'],
    'price': [199.0, 89.0, np.nan, 129.0, 79.0, 89.0],
    'weight': [500, 450, 300, np.nan, 410, np.nan],
    'bought': ['yes', 'no', 'yes', 'no', 'yes', 'no']
}

df_raw = pd.DataFrame(data=data)
print(df_raw)

# Utworzenie kopii danych
df = df_raw.copy()
df.info()

# Sprawdzenie braków
print(df.isnull())
print(df.isnull().sum())
print(df.isnull().sum().sum())
print(df.isnull().sum() / len(df))

# Uzupełnienie braków - SimpleImputer
print(df[['weight']])

# strategy: 'mean', 'median', 'most_frequent', 'constant'
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(df[['weight']])

print(imputer.statistics_)

imputer.transform(df[['weight']])

df['weight'] = imputer.transform(df[['weight']])
print(df)

imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=99.0)
print(imputer.fit_transform(df[['price']]))

imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='L')
print(imputer.fit_transform(df[['size']]))

print(df)

imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer.fit_transform(df[['size']])

df = df_raw.copy()
print(df)

print(df.isnull().sum())
print(pd.isnull(df['weight']))
print(df[pd.isnull(df['weight'])])
print(df[~pd.isnull(df['weight'])])

pd.notnull(df['weight'])

print(df[pd.notnull(df['weight'])])

print(df)

df.fillna(value='brak')
df.fillna(value=0.0)

print(df['size'])

df['size'].fillna(value='L', inplace=True)
print(df)

df.dropna()

df = df.dropna()
print(df)

# Braki danych w szeregach czasowych
data = {'price': [108, 109, 110, 110, 109, np.nan, np.nan, 112, 111, 111]}
date_range = pd.date_range(start='01-01-2020 09:00', end='01-01-2020 18:00', periods=10)

df = pd.DataFrame(data=data, index=date_range)
print(df)

register_matplotlib_converters()
sns.set()

plt.figure(figsize=(10, 4))
plt.title('Braki danych')
_ = plt.plot(df.price)

df_plotly = df.reset_index()
px.line(df_plotly, 'index', 'price', width=600, height=400,
        title='Szeregi czasowe - braki danych')
# Usunięcie braków

df_plotly = df_plotly.dropna()
px.line(df_plotly, 'index', 'price', width=600, height=400,
        title='Szeregi czasowe - braki danych')
# Wypełnienie braków stałą wartością 0

df_plotly = df.reset_index()
df_plotly['price_fill'] = df_plotly['price'].fillna(0)
px.line(df_plotly, 'index', 'price_fill', width=600, height=400,
        title='Szeregi czasowe - braki danych - wstawienie 0')
# Wypełnienie braków średnią

df_plotly = df.reset_index()
df_plotly['price_fill'] = df_plotly['price'].fillna(df_plotly['price'].mean())
px.line(df_plotly, 'index', 'price_fill', width=600, height=400,
        title='Szeregi czasowe - braki danych - wstawienie średniej')
# Zastosowanie interpolacji

df_plotly = df.reset_index()
df_plotly['price_fill'] = df_plotly['price'].interpolate()
px.line(df_plotly, 'index', 'price_fill', width=600, height=400,
        title='Szeregi czasowe - braki danych - interpolacja')
# Wypełnienie braków metodą forward fill

df_plotly = df.reset_index()
df_plotly['price_fill'] = df_plotly['price'].fillna(method='ffill')
px.line(df_plotly, 'index', 'price_fill', width=600, height=400,
        title='Szeregi czasowe - braki danych - forward fill')
# Wypełnienie braków metodą backward fill

df_plotly = df.reset_index()
df_plotly['price_fill'] = df_plotly['price'].fillna(method='bfill')
px.line(df_plotly, 'index', 'price_fill', width=600, height=400,
        title='Szeregi czasowe - braki danych')

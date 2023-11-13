# Import bibliotek
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

sns.set()
np.random.seed(42)
np.set_printoptions(precision=4, suppress=True)
print(f'sklearn version: {sklearn.__version__}')

# Wczytanie danych
df_raw = pd.read_csv('https://storage.googleapis.com/esmartdata-courses-files/'
                     'ml-course/insurance.csv')
print(df_raw.head())

# Utworzenie kopii danych
df = df_raw.copy()
print(df.info())

# Eksplorcja i wstępne przygotowanie danych
print(df[df.duplicated()])
print(df[df['charges'] == 1639.5631])

df = df.drop_duplicates()
print(df.info())

cat_cols = [col for col in df.columns if df[col].dtype == 'O']
print(cat_cols)

cat_cols = [col for col in df.columns if df[col].dtype == 'O']
print(cat_cols)

for col in cat_cols:
    df[col] = df[col].astype('category')
df.info()

print(df.describe().T)
print(df.describe(include=['category']).T)

print(df.isnull().sum())
print(df.sex.value_counts())
df.sex.value_counts().plot(kind='pie')

print(df.smoker.value_counts())
print(df.region.value_counts())
df.charges.plot(kind='hist', bins=30)

import plotly.express as px

px.histogram(df, x='charges', width=700, height=400, nbins=50, facet_col='smoker', facet_row='sex')
px.histogram(df, x='smoker', facet_col='sex', color='sex', width=700, height=400)
df_dummies = pd.get_dummies(df, drop_first=True)
print(df_dummies)

corr = df_dummies.corr()
print(corr)

sns.set(style="white")
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(8, 6))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5,
            cbar_kws={"shrink": .5})

df_dummies.corr()['charges'].sort_values(ascending=False)

sns.set()
df_dummies.corr()['charges'].sort_values()[:-1].plot(kind='barh')

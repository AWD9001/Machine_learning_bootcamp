# Import bibliotek
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
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

data = df_dummies.copy()
target = data.pop('charges')
data.head()

target.head()

# Podział danych na zbiór treningowy i testowy
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

print(f'X_trian shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_trian shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')

# Regresja liniowa
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

print(f'R2 score: {regressor.score(X_test, y_test):.4f}')

y_pred = regressor.predict(X_test)
print(y_pred[:10])

y_true = y_test.copy()
predictions = pd.DataFrame(data={'y_true': y_true, 'y_pred': y_pred})
predictions['error'] = predictions['y_true'] - predictions['y_pred']
print(predictions.head())

predictions.error.plot(kind='hist', bins=30)

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_true, y_pred)
print(f'MAE wynosi: {mae:.2f}')

print(regressor.intercept_)
print(regressor.coef_)
print(data.columns)

# Dobór cech modelu - eliminacja wsteczna

import statsmodels.api as sm

X_train_ols = X_train.copy()
X_train_ols = X_train_ols.values
X_train_ols = sm.add_constant(X_train_ols)
print(X_train_ols)

ols = sm.OLS(endog=y_train, exog=X_train_ols).fit()
predictors = ['const'] + list(X_train.columns)
print(ols.summary(xname=predictors))

X_selected = X_train_ols[:, [0, 1, 2, 3, 5, 6, 7, 8]]
predictors.remove('sex_male')

ols = sm.OLS(endog=y_train, exog=X_selected).fit()
print(ols.summary(xname=predictors))

X_selected = X_train_ols[:, [0, 1, 2, 3, 5, 7, 8]]
predictors.remove('region_northwest')

ols = sm.OLS(endog=y_train, exog=X_selected).fit()
print(ols.summary(xname=predictors))

X_selected = X_train_ols[:, [0, 1, 2, 3, 5, 7]]
predictors.remove('region_southwest')

ols = sm.OLS(endog=y_train, exog=X_selected).fit()
print(ols.summary(xname=predictors))

X_selected = X_train_ols[:, [0, 1, 2, 3, 5]]
predictors.remove('region_southeast')

ols = sm.OLS(endog=y_train, exog=X_selected).fit()
print(ols.summary(xname=predictors))

# Eliminacja wsteczna - automatyzacja
X_train_numpy = X_train.values
X_train_numpy = sm.add_constant(X_train_numpy)
num_vars = len(X_train_numpy[0])

predictors = ['const'] + list(X_train.columns)
sl = 0.05

for i in range(0, num_vars):
    ols = sm.OLS(endog=y_train, exog=X_train_numpy).fit()
    max_pval = max(ols.pvalues.astype('float'))
    if max_pval > sl:
        for j in range(0, num_vars - i):
            if ols.pvalues[j].astype('float') == max_pval:
                X_train_numpy = np.delete(X_train_numpy, j, axis=1)
                predictors.remove(predictors[j])

print(ols.summary(xname=predictors))

# Zapisanie końcowego modelu
ols.save('model.pickle')

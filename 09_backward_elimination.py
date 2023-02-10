import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm



sns.set()
np.random.seed(42)
np.set_printoptions(precision=4, suppress=True)

# Wczytanie danych
df_raw = pd.read_csv(
    'https://storage.googleapis.com/esmartdata-courses-files/ml-course/insurance.csv')
print('Pięć pierwszych wierszy:\n', df_raw.head())

# Utworzenie kopii
df = df_raw.copy()
print('info:')
print(df.info())

# Eksplorcja i wstępne przygotowanie danych
print('Pokaż wiersze zduplikowane:\n', df[df.duplicated()])
print('weryfikacja:')
print(df[df['charges'] == 1639.5631])
df = df.drop_duplicates()

# Zmiana typów danych w kolumnach z object na category
cat_cols = [col for col in df.columns if df[col].dtype == 'O']
print('kolumny:', cat_cols)
for col in cat_cols:
    df[col] = df[col].astype('category')
print(df.info())

# Opisy danych
print(df.describe().T)
print(df.describe(include=['category']).T)
print(df.isnull().sum())
print('sex:', df.sex.value_counts())
df.sex.value_counts().plot(kind='pie')
plt.show()
print('smoker:', df.smoker.value_counts())
print('region:', df.region.value_counts())
df.charges.plot(kind='hist', bins=30)
plt.show()

p = px.histogram(df, x='charges', width=700, height=400, nbins=50, facet_col='smoker',
                 facet_row='sex')
q = px.histogram(df, x='smoker', facet_col='sex', color='sex', width=700, height=400)
p.show()
q.show()

df_dummies = pd.get_dummies(df, drop_first=True)
print(df_dummies)

corr = df_dummies.corr()
print(corr)

sns.set(style="white")
mask = np.zeros_like(corr, dtype=np.bool_)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(8, 6))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()

print(df_dummies.corr()['charges'].sort_values(ascending=False))


sns.set()
df_dummies.corr()['charges'].sort_values()[:-1].plot(kind='barh')
plt.show()


data = df_dummies.copy()
target = data.pop('charges')
print(data.head())
print(target.head())

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

# Regresja liniowa
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
plt.show()

mae = mean_absolute_error(y_true, y_pred)
print(f'MAE wynosi: {mae:.2f}')

print(regressor.intercept_)
print(regressor.coef_)
print(data.columns)


# Dobór cech modelu - eliminacja wsteczna

X_train_ols = X_train.copy()
X_train_ols = X_train_ols.values
X_train_ols = sm.add_constant(X_train_ols)
print('X_train_ols:')
print(X_train_ols)

ols = sm.OLS(endog=y_train, exog=X_train_ols).fit()
predictors = ['const'] + list(X_train.columns)
print('ols.summary(xname=predictors):')
print(ols.summary(xname=predictors))

X_selected = X_train_ols[:, [0, 1, 2, 3, 5, 6, 7, 8]]
predictors.remove('sex_male')

ols = sm.OLS(endog=y_train, exog=X_selected).fit()
print('ols.summary(xname=predictors): remove_sex_male')
print(ols.summary(xname=predictors))

X_selected = X_train_ols[:, [0, 1, 2, 3, 5, 7, 8]]
predictors.remove('region_northwest')

ols = sm.OLS(endog=y_train, exog=X_selected).fit()
print('ols.summary(xname=predictors): remove_region_northwest')
print(ols.summary(xname=predictors))


X_selected = X_train_ols[:, [0, 1, 2, 3, 5, 7]]
predictors.remove('region_southwest')

ols = sm.OLS(endog=y_train, exog=X_selected).fit()
print('ols.summary(xname=predictors): remove_region_southwest')
print(ols.summary(xname=predictors))

X_selected = X_train_ols[:, [0, 1, 2, 3, 5]]
predictors.remove('region_southeast')

ols = sm.OLS(endog=y_train, exog=X_selected).fit()
print('ols.summary(xname=predictors): remove_region_southeast')
print(ols.summary(xname=predictors))

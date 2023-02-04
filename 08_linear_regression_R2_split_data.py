import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

np.random.seed(42)
np.set_printoptions(precision=6, suppress=True, edgeitems=30, linewidth=120,
                    formatter=dict(float=lambda x: f'{x:.2f}'))
sns.set(font_scale=1.3)

# Wygenerowanie danych
data, target = make_regression(n_samples=100, n_features=1, n_targets=1, noise=30.0,
                               random_state=42)

print(f'data shape: {data.shape}')
print(f'target shape: {target.shape}')

print(data[:5])

print(target[:5])

plt.figure(figsize=(8, 6))
plt.title('Regresja liniowa')
plt.xlabel('cecha x')
plt.ylabel('zmienna docelowa')
plt.scatter(data, target, label='cecha x')
plt.legend()
plt.plot()

# Regresja Liniowa przy użyciu scikit - learn

regressor = LinearRegression()

# metoda fit() dopasowuje model liniowy do danych
regressor.fit(data, target)

LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None)

# metoda score() dokonuje oceny modelu na przekazanych danych (wynik R2 score)
regressor.score(data, target)

# metoda predict() dokonuje predykcji na podstawie modelu
y_pred = regressor.predict(data)
print(y_pred)


# Wizualizacja graficzna modelu
plt.figure(figsize=(8, 6))
plt.title('Regresja liniowa')
plt.xlabel('cecha x')
plt.ylabel('zmienna docelowa')
plt.scatter(data, target, label='cecha x')
plt.plot(data, y_pred, color='red', label='model')
plt.legend()
plt.show()

print('regressor.score(data, target):', regressor.score(data, target))

print("item for item in dir(regressor) if not item.startswith('_')")
print([item for item in dir(regressor) if not item.startswith('_')])

print('regressor.coef_:', regressor.coef_)
print('regressor.intercept_:', regressor.intercept_)

# Końcowa postać modelu
plt.figure(figsize=(8, 6))
plt.title('Regresja liniowa')
plt.xlabel('cecha x')
plt.ylabel('zmienna docelowa')
plt.scatter(data, target, label='cecha x')
plt.plot(data, regressor.intercept_ + regressor.coef_[0] * data, color='red', label='model')
plt.legend()
plt.show()

# Regresja z podziałem na zbiór treningowy oraz testowy

data, target = make_regression(n_samples=1000, n_features=1, n_targets=1, noise=15.0,
                               random_state=42)

print(f'data shape: {data.shape}')
print(f'target shape: {target.shape}')


X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25)

print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_train shape: {y_train.shape}')

# Wykres punktów zbioru
plt.figure(figsize=(8, 6))
plt.title('Regresja liniowa train vs. test')
plt.xlabel('cecha x')
plt.ylabel('zmienna docelowa')
plt.scatter(X_train, y_train, label='zbiór treningowy', color='gray', alpha=0.5)
plt.scatter(X_test, y_test, label='zbiór testowy', color='gold', alpha=0.5)
plt.legend()
plt.plot()

regressor = LinearRegression()
regressor.fit(X_train, y_train)

regressor.score(X_train, y_train)

regressor.score(X_test, y_test)

# Regresja liniowa - zbiór treningowy - wizualizacja
plt.figure(figsize=(8, 6))
plt.title('Regresja liniowa: zbior treningowy')
plt.xlabel('cecha x')
plt.ylabel('zmienna docelowa')
plt.scatter(X_train, y_train, label='zbiór treningowy', color='gray', alpha=0.5)
plt.plot(X_train, regressor.intercept_ + regressor.coef_[0] * X_train, color='red')
plt.legend()
plt.plot()

# Regresja liniowa - zbiór testowy - wizualizacja
plt.figure(figsize=(8, 6))
plt.title('Regresja liniowa: zbior testowy')
plt.xlabel('cecha x')
plt.ylabel('zmienna docelowa')
plt.scatter(X_test, y_test, label='zbiór testowy', color='gold', alpha=0.5)
plt.plot(X_test, regressor.intercept_ + regressor.coef_[0] * X_test, color='red')
plt.legend()
plt.plot()

# Predykcja na podstawie modelu
y_pred = regressor.predict(X_test)

predictions = pd.DataFrame(data={'y_true': y_test, 'y_pred': y_pred})
print(predictions.head())

predictions['error'] = predictions['y_true'] - predictions['y_pred']
print(predictions.head())

_ = predictions['error'].plot(kind='hist', bins=50, figsize=(8, 6))

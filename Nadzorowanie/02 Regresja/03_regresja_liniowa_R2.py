# Import bibliotek
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
import seaborn as sns
import sklearn

np.random.seed(42)
np.set_printoptions(precision=6, suppress=True, edgeitems=30, linewidth=120,
                    formatter=dict(float=lambda x: f'{x:.2f}'))
sns.set(font_scale=1.3)
print(f'sklearn.__version__: {sklearn.__version__}\n')

# Wygenerowanie danych
from sklearn.datasets import make_regression

data, target = make_regression(n_samples=100, n_features=1, n_targets=1, noise=30.0,
                               random_state=42)

print(f'data shape: {data.shape}\n')
print(f'target shape: {target.shape}')

print(f'data[:5]:\n{data[:5]}\n')
print(f'target[:5]:\n{target[:5]}\n')

plt.figure(figsize=(8, 6))
plt.title('Regresja liniowa')
plt.xlabel('cecha x')
plt.ylabel('zmienna docelowa')
plt.scatter(data, target, label='cecha x')
plt.legend()
plt.plot()

# Regresja Liniowa przy użyciu scikit-learn
# Y = w0 + w1X1

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

# metoda fit() dopasowuje model liniowy do danych
print(f'regressor.fit(data, target):\n{regressor.fit(data, target)}\n')

# metoda score() dokonuje oceny modelu na przekazanych danych (wynik R2 score)
print(f'regressor.score(data, target): {regressor.score(data, target)}\n')

# metoda predict() dokonuje predykcji na podstawie modelu
y_pred = regressor.predict(data)
print(f'y_pred = regressor.predict(data):\n{y_pred}\n')

# Wizualizacja graficzna modelu

plt.figure(figsize=(8, 6))
plt.title('Regresja liniowa')
plt.xlabel('cecha x')
plt.ylabel('zmienna docelowa')
plt.scatter(data, target, label='cecha x')
plt.plot(data, y_pred, color='red', label='model')
plt.legend()
plt.show()


# Współczynnik R^2
print(f'regressor.score(data, target): {regressor.score(data, target)}\n')
print(f"[item for item in dir(regressor) if not item.startswith('_')]"
      f"\n{[item for item in dir(regressor) if not item.startswith('_')]}\n")

print(f'regressor.coef_: {regressor.coef_}')
print(f'regressor.intercept_: {regressor.intercept_}\n')

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
print(f'target shape: {target.shape}\n')

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25)

print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_train shape: {y_train.shape}\n')

plt.figure(figsize=(8, 6))
plt.title('Regresja liniowa train vs. test')
plt.xlabel('cecha x')
plt.ylabel('zmienna docelowa')
plt.scatter(X_train, y_train, label='zbiór treningowy', color='gray', alpha=0.5)
plt.scatter(X_test, y_test, label='zbiór testowy', color='gold', alpha=0.5)
plt.legend()
plt.plot()

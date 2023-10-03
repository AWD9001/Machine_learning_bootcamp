# Import bibliotek
import numpy as np
import pandas as pd
import plotly.express as px

np.random.seed(42)

# Wygenerowanie danych
X1 = np.array([1, 2, 3, 4, 5, 6])
Y = np.array([3000, 3250, 3500, 3750, 4000, 4250])
m = len(X1)

print(f'Lata pracy: {X1}')
print(f'Wynagrodzenie: {Y}')
print(f'Liczba próbek: {m}\n')

# Przygotowanie danych
X1 = X1.reshape(m, 1)
Y = Y.reshape(-1, 1)
print(f'Lata pracy.reshape(m = len(X1), 1):\n{X1}\n')
print(f'Lata pracy.shape:\n{X1.shape}\n')
print(f'Wynagrodzenie.reshape(-1, 1):\n{Y}\n')

bias = np.ones((m, 1))
print(f'bias = np.ones((m, 1)):\n{bias}\n')
print(f'bias.shape:\n{bias.shape}\n')

X = np.append(bias, X1, axis=1)
print(f'X = np.append(bias, X1, axis=1):\n{X}\n')
print(f'X.shape:\n{X.shape}\n')

# Losowa inicjalizacja parametrów
eta = 0.01
print(f'eta: {eta}\n')

weights = np.random.randn(2, 1)
print(f'weights = np.random.randn(2, 1):\n{weights}\n')

# Metoda gradientu prostego
intercept = []
coef = []

for i in range(3000):
    gradient = (2 / m) * X.T.dot(X.dot(weights) - Y)
    weights = weights - eta * gradient
    intercept.append(weights[0][0])
    coef.append(weights[1][0])

print('Metoda gradientu prostego:')
print(f'weights:\n{weights}\n')

df = pd.DataFrame(data={'intercept': intercept, 'coef': coef})
print("df = pd.DataFrame(data={'intercept': intercept, 'coef': coef})")
print(f'df.head():\n{df.head()}')

# Wizualizacja dopasowania
px.line(df, y='intercept', width=800, title='Dopasowanie: intercept')
px.line(df, y='coef', width=800, title='Dopasowanie: coef')

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
print(f'Liczba próbek: {m}')

# Przygotowanie danych
X1 = X1.reshape(m, 1)
Y = Y.reshape(-1, 1)
print('\nX1:\n', X1)
print('X1.shape:', X1.shape)

bias = np.ones((m, 1))
print('bias:\n', bias)
print('bias.shape:', bias.shape)

X = np.append(bias, X1, axis=1)
print('\nMacierz X po połączeniu z macierzą jedynkową:')
print(X)
print('X.shape:', X.shape)

# Losowa inicjalizacja parametrów
eta = 0.01

weights = np.random.randn(2, 1)
print('\nX:\n', X)
print('weights:\n', weights)

# Metoda gradientu prostego
intercept = []
coef = []

for i in range(3000):
    gradient = (2 / m) * X.T.dot(X.dot(weights) - Y)
    weights = weights - eta * gradient
    intercept.append(weights[0][0])
    coef.append(weights[1][0])

print('\nwagi po dopasowaniu:')
print(weights)

df = pd.DataFrame(data={'intercept': intercept, 'coef': coef})
print('\n', df.head())

# Wizualizacja dopasowania

fig1 = px.line(df, y='intercept', width=800, title='Dopasowanie: intercept')
fig2 = px.line(df, y='coef', width=800, title='Dopasowanie: coef')
fig1.show()
fig2.show()

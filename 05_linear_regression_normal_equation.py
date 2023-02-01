import numpy as np
from sklearn.linear_model import LinearRegression

# Przygotowanie danych

X1 = np.array([1, 2, 3, 4, 5, 6])
Y = np.array([3000, 3250, 3500, 3750, 4000, 4250])
m = len(X1)

print(f'Lata pracy: {X1}')
print(f'Wynagrodzenie: {Y}')
print(f'Liczba próbek: {m}', '\n')

X1 = X1.reshape(m, 1)
print('X1:')
print(X1)
print(X1.shape, '\n')

print('bias:')
bias = np.ones((m, 1))
print(bias)
print(bias.shape, '\n')

print('X:')
X = np.append(bias, X1, axis=1)
print(X)
print(X.shape, '\n')

# Równanie normalne:
print('Równanie normalne:')
print(np.dot(X.T, X))

L = np.linalg.inv(np.dot(X.T, X))
P = np.dot(X.T, Y)
print('L:', '\n', L, '\n')
print('P:', '\n', P)

print(np.dot(L, P))

# Regresja liniowa przy pomocy scikit-learn
print('Regresja liniowa przy pomocy scikit-learn:')

regression = LinearRegression()
regression.fit(X1, Y)

print(regression.intercept_)
print(regression.coef_[0])

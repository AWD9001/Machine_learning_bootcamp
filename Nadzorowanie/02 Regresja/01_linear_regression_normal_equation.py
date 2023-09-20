# Import bibliotek
import numpy as np
from sklearn.linear_model import LinearRegression

# Przygotowanie danych
X1 = np.array([1, 2, 3, 4, 5, 6])
Y = np.array([3000, 3250, 3500, 3750, 4000, 4250])
m = len(X1)

print('\nDane wejściowe:')
print(f'    Lata pracy: {X1}')
print(f'    Wynagrodzenie: {Y}')
print(f'    Liczba próbek: {m}\n')

X1 = X1.reshape(m, 1)
print(f'X1:\n{X1}\n')
print(f'X1.shape:\n{X1.shape}\n')

bias = np.ones((m, 1))
print(bias)
print(bias.shape)

X = np.append(bias, X1, axis=1)
print(X)
print(X.shape)

# Równanie normalne
print(np.dot(X.T, X))

L = np.linalg.inv(np.dot(X.T, X))
print(L)

P = np.dot(X.T, Y)
print(P)

print(np.dot(L, P))

# Regresja liniowa przy pomocy scikit-learn
print('\n   Regresja liniowa przy pomocy scikit-learn:')
regression = LinearRegression()
regression.fit(X1, Y)

print('regression.intercept_:')
print(regression.intercept_)
print('regression.coef_[0]:')
print(regression.coef_[0])

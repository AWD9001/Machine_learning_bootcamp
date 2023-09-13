# Import bibliotek
import numpy as np

# Przygotowanie danych
X1 = np.array([1, 2, 3, 4, 5, 6])
Y = np.array([3000, 3250, 3500, 3750, 4000, 4250])
m = len(X1)

print(f'Lata pracy: {X1}')
print(f'Wynagrodzenie: {Y}')
print(f'Liczba próbek: {m}')

X1 = X1.reshape(m, 1)
print(X1)
print(X1.shape)

bias = np.ones((m, 1))
print(bias)
print(bias.shape)

X = np.append(bias, X1, axis=1)
print(X)
print(X.shape)

# Równanie normalne
np.dot(X.T, X)

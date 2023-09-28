# Import bibliotek
import numpy as np
# import pandas as pd
# import plotly.express as px

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

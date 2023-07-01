# Import bibliotek
import numpy as np
import pandas as pd
import sklearn

print(sklearn.__version__)

# Wygenerowanie danych
data = {
    'size': ['XL', 'L', 'M', np.nan, 'M', 'M'],
    'color': ['red', 'green', 'blue', 'green', 'red', 'green'],
    'gender': ['female', 'male', np.nan, 'female', 'female', 'male'],
    'price': [199.0, 89.0, np.nan, 129.0, 79.0, 89.0],
    'weight': [500, 450, 300, np.nan, 410, np.nan],
    'bought': ['yes', 'no', 'yes', 'no', 'yes', 'no']
}

df_raw = pd.DataFrame(data=data)
print(df_raw)

# Utworzenie kopii danych
df = df_raw.copy()
df.info()

# Sprawdzenie braków
print(df.isnull())
print(df.isnull().sum())
print(df.isnull().sum().sum())
print(df.isnull().sum() / len(df))

# Import bibliotek

# import numpy as np
import pandas as pd
import sklearn

print(sklearn.__version__)

# Wygenerowanie danych
data = {
    'size': ['XL', 'L', 'M', 'L', 'M'],
    'color': ['red', 'green', 'blue', 'green', 'red'],
    'gender': ['female', 'male', 'male', 'female', 'female'],
    'price': [199.0, 89.0, 99.0, 129.0, 79.0],
    'weight': [500, 450, 300, 380, 410],
    'bought': ['yes', 'no', 'yes', 'no', 'yes']
}

df_raw = pd.DataFrame(data=data)
print(df_raw)

# Utworzenie kopii danych
df = df_raw.copy()
print(df.info())

# Zmiana typu danych i wstępna eksploracja
for col in ['size', 'color', 'gender', 'bought']:
    df[col] = df[col].astype('category')

df['weight'] = df['weight'].astype('float')

print("Info:\n", df.info(), "\n")
print("Opis:\n", df.describe(), "\n")
print("Transpozycja:\n", df.describe().T, "\n")

print("Podział na kategorie:\n", df.describe(include=['category']).T, "\n")
print(df)
# Import bibliotek

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

# LabelEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df['bought'])
le.transform(df['bought'])

le.fit_transform(df['bought'])
print(le.classes_)

df['bought'] = le.fit_transform(df['bought'])
print(df)

le.inverse_transform(df['bought'])

df['bought'] = le.inverse_transform(df['bought'])
print(df)

# OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
encoder.fit(df[['size']])

encoder.transform(df[['size']])
print(encoder.categories_)

encoder = OneHotEncoder(drop='first', sparse=False)
encoder.fit(df[['size']])
encoder.transform(df[['size']])
print(df)

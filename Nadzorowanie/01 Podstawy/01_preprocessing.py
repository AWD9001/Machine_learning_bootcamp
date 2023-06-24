# Import bibliotek

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import scale

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
encoder = OneHotEncoder(sparse=False)
encoder.fit(df[['size']])

encoder.transform(df[['size']])
print(encoder.categories_)

encoder = OneHotEncoder(drop='first', sparse=False)
encoder.fit(df[['size']])
encoder.transform(df[['size']])
print(df)

# Pandas get_dummies()
print(pd.get_dummies(data=df))
print(pd.get_dummies(data=df, drop_first=True))
print(pd.get_dummies(data=df, drop_first=True, prefix='new'))
print(pd.get_dummies(data=df, drop_first=True, prefix_sep='-'))
print(pd.get_dummies(data=df, drop_first=True, columns=['size']))

# Standaryzacja - StandardScaler
# std() - pandas nieobciążony
# std() - numpy obciążony

print((df['price'] - df['price'].mean()) / df['price'].std())


def standardize(x):
    return (x - x.mean()) / x.std()


standardize(df['price'])

scale(df['price'])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df[['price']])
scaler.transform(df[['price']])

scaler = StandardScaler()
df[['price', 'weight']] = scaler.fit_transform(df[['price', 'weight']])
print(df)

# Przygotowanie danych do modelu
df = df_raw.copy()
print(df)

le = LabelEncoder()

df['bought'] = le.fit_transform(df['bought'])

scaler = StandardScaler()
df[['price', 'weight']] = scaler.fit_transform(df[['price', 'weight']])

df = pd.get_dummies(data=df, drop_first=True)
print(df)

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


data = {
    'size': ['XL', 'L', 'M', np.nan, 'L', 'M'],
    'color': ['red', 'green', 'blue', 'green', 'red', 'green'],
    'gender': ['female', 'male', np.nan, 'male', 'female', 'female'],
    'price': [199.0, 89.0, np.nan, 99.0, 129.0, 79.0],
    'weight': [500, 450, 300, np.nan, 410, np.nan],
    'bought': ['yes', 'no', 'yes', 'no', 'yes', 'no']
}

df_raw = pd.DataFrame(data=data)

df = df_raw.copy()  # kopia zapasowa
print('Informacje o danych:')
print(df.info(), '\n')

print('Czy jest pusta wartość?:')
print(df.isnull(), '\n')

print('Liczba pusych danych w kolumnach:')
print(df.isnull().sum(), '\n')

print('Procent braku danych:')
print(df.isnull().sum() / len(df), '\n')

# SimpleInmputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(df[['weight']])
df['weight'] = imputer.transform(df[['weight']])
print('Kolumna "weight" do zastąpieniu pustych pozycji:')
print(df['weight'], '\n')

imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=99.0)
imputer.fit(df[['price']])
df['price'] = imputer.transform(df[['price']])
print('Kolumna "price" do zastąpieniu pustych pozycji:')
print(df['price'], '\n')

imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer.fit(df[['size']])
df['size'] = imputer.transform(df[['size']])
print('Kolumna "size" do zastąpieniu pustych pozycji:')
print(df['size'], '\n')

# badanie pustych
print(df_raw.isnull().sum(), '\n')
print(pd.isnull(df_raw['weight']), '\n')
print(df_raw[pd.isnull(df_raw['weight'])], '\n')  # wiersze w których są puste wartości
print(df_raw[~pd.isnull(df_raw['weight'])], '\n')  # odwrotność tego wyżej
print(pd.notnull(df_raw['weight']))  # to samo

# metoda fillna
df = df_raw
print(df.fillna(value='brak'), '\n')
print(df.fillna(value=0.0), '\n')
print(df.fillna(value='L', inplace=True), '\n')
print(df.dropna, '\n')

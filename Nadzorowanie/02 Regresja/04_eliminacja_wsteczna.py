# Import bibliotek
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

sns.set()
np.random.seed(42)
np.set_printoptions(precision=4, suppress=True)
print(f'sklearn version: {sklearn.__version__}')

# Wczytanie danych
df_raw = pd.read_csv('https://storage.googleapis.com/esmartdata-courses-files/'
                     'ml-course/insurance.csv')
print(df_raw.head())

# Utworzenie kopii danych
df = df_raw.copy()
print(df.info())

# Eksplorcja i wstępne przygotowanie danych
print(df[df.duplicated()])
print(df[df['charges'] == 1639.5631])

df = df.drop_duplicates()
print(df.info())

# Import bibliotek
import numpy as np
import pandas as pd
import sklearn
import seaborn as sns

np.random.seed(42)
np.set_printoptions(edgeitems=30, linewidth=100000,
                    formatter=dict(float=lambda x: f'{x:.2f}'))
sns.set()
print(sklearn.__version__)

# Załadowanie danych
from sklearn.datasets import load_iris

raw_data = load_iris()
raw_data_copy = raw_data.copy()
raw_data_copy.keys()

print(raw_data_copy['DESCR'])

# Przygotowanie danych
data = raw_data_copy['data']
target = raw_data_copy['target']

print(f'{data[:5]}\n')
print(target[:5])

# połączenie atrybutów ze zmienną docelową
all_data = np.c_[data, target]
print(all_data[:5])

# budowa obiektu DataFrame
df = pd.DataFrame(data=all_data, columns=raw_data.feature_names + ['target'])
print(df.head())
print(df.tail())
print(df.info())
print(df.describe())
print(df.describe().T.apply(lambda x: round(x, 2)))
print(df.target.value_counts())
df.target.value_counts().plot(kind='pie')

data = df.copy()
target = data.pop('target')
print(data.head())
print(target.head())

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

sns.set()
np.random.seed(42)
np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: f'{x:.2f}'))

# Załadowanie danych
raw_data = load_iris()
raw_data_copy = raw_data.copy()
raw_data_copy.keys()
print(raw_data_copy['DESCR'])

# Przygotowanie danych
data = raw_data_copy['data']
target = raw_data_copy['target']
print('\n', f'{data[:5]}\n')
print(target[:5])
# połączenie atrybutów ze zmienną docelową
all_data = np.c_[data, target]
print('\n', all_data[:5])
# budowa obiektu DataFrame
df = pd.DataFrame(data=all_data, columns=raw_data.feature_names + ['target'])
print('\n', df.head())

print('\n', df.info())
print(df.describe())
print(df.describe().T.apply(lambda x: round(x, 2)))  # zaokrąglenie do 2-óch miejsc po przecinku
print(df.target.value_counts())

df.target.value_counts().plot(kind='pie')


data = df.copy()
target = data.pop('target')
print('\n', data.head())
print('target:', target.head())

# Podział danych na zbiór treningowy i testowy - iris data
X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=42)
print(f'X_train shape {X_train.shape}')
print(f'y_train shape {y_train.shape}')
print(f'X_test shape {X_test.shape}')
print(f'y_test shape {y_test.shape}')
print(f'\nTest ratio: {len(X_test) / len(data):.2f}')
print(f'\ny_train:\n{y_train.value_counts()}')
print(f'\ny_test:\n{y_test.value_counts()}')
print(f'X_train shape {X_train.shape}')
print(f'y_train shape {y_train.shape}')
print(f'X_test shape {X_test.shape}')
print(f'y_test shape {y_test.shape}')
print(f'\nTest ratio: {len(X_test) / len(data):.2f}')
print(f'\ny_train:\n{y_train.value_counts()}')
print(f'\ny_test:\n{y_test.value_counts()}')


# Import bibliotek
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
# from IPython.display import Image

sns.set(font_scale=1.3)
np.random.seed(42)

# Załadowanie danych
raw_data = load_iris()
all_data = raw_data.copy()

data = all_data['data']
target = all_data['target']
feature_names = [name.replace(' ', '_')[:-5] for name in all_data['feature_names']]
target_names = all_data['target_names']

print(f'Liczba próbek: {len(data)}')
print(f'Kształt danych: {data.shape}')
print(f'Nazwy zmiennych objaśniających: {feature_names}')
print(f'Nazwy kategorii kosaćca: {target_names}')

# Eksploracja danych
df = pd.DataFrame(data=np.c_[data, target], columns=feature_names + ['target'])
print(df.head())
plt.figure(figsize=(8, 6))
_ = sns.scatterplot('sepal_length', hue='target', data=df, legend='full',
                    palette=sns.color_palette()[:3])
print(df['target'].value_counts())

# Przygotowanie danych do modelu
data = df.copy()
data = data[['sepal_length', 'sepal_width', 'target']]
target = data.pop('target')

print(data.head())
print(target.head())

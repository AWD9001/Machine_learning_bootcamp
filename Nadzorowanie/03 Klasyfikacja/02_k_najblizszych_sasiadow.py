# Import bibliotek
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.datasets import load_iris

np.random.seed(42)
sns.set(font_scale=1.3)

# Wczytanie danych
raw_data = load_iris()
print(raw_data.keys())

all_data = raw_data.copy()

data = all_data['data']
target = all_data['target']

print(f'{data[:5]}\n')
print(f'{target[:5]}')

print(all_data['target_names'])

df = pd.DataFrame(data=np.c_[data, target], columns=all_data['feature_names'] + ['class'])
print(df.head())

print(df.info())
print(df.describe().T)
print(df['class'].value_counts())

# Wykres Rozproszenia

print(df.columns)

_ = sns.pairplot(df, vars=all_data['feature_names'], hue='class')
print(df.corr())

data = data[:, :2]

print('data shape:', data.shape)
print('target shape:', target.shape)

plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c=target, cmap='viridis')
plt.title('Wykres punktowy')
plt.xlabel('cecha_1: sepal_length')
plt.ylabel('cecha_2: sepal_width')
plt.show()

df = pd.DataFrame(data=np.c_[data, target], columns=['sepal_length', 'sepal_width', 'class'])
px.scatter(df, x='sepal_length', y='sepal_width', color='class', width=800)

# K-nearest Neighbour Algorithm - Algorytm K-najbliższych sąsiadów
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(data, target)

# Wykres granic decyzyjnych
x_min, x_max = data[:, 0].min() - 0.5, data[:, 0].max() + 0.5
y_min, y_max = data[:, 1].min() - 0.5, data[:, 1].max() + 0.5

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
mesh = np.c_[xx.ravel(), yy.ravel()]
Z = classifier.predict(mesh)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.pcolormesh(xx, yy, Z, cmap='gnuplot', alpha=0.1)
plt.scatter(data[:, 0], data[:, 1], c=target, cmap='gnuplot', edgecolors='r')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title('3-class classification k=5')
plt.show()

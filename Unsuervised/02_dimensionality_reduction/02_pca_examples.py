# Import bibliotek

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.datasets import mnist

np.set_printoptions(precision=4, suppress=True, linewidth=150)

# Załadowanie danych - breast cancer
raw_data = load_breast_cancer()
all_data = raw_data.copy()
data = all_data['data']
target = all_data['target']
print(data[:3])

print(target[:30])

print(data.shape)

# Standaryzacja
scaler = StandardScaler()
data_std = scaler.fit_transform(data)
print(data_std[:3])

# PCA - 2 komponenty
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_std)
print(data_pca[:5])

pca_2 = pd.DataFrame(data={'pca_1': data_pca[:, 0], 'pca_2': data_pca[:, 1], 'class': target})
pca_2.replace(0, 'Benign', inplace=True)
pca_2.replace(1, 'Malignant', inplace=True)
print(pca_2.head())

results = pd.DataFrame(data={'explained_variance_ratio': pca.explained_variance_ratio_})
results['cumulative'] = results['explained_variance_ratio'].cumsum()
results['component'] = results.index + 1
print(results)

px.scatter(pca_2, 'pca_1', 'pca_2', color=pca_2['class'], width=950, template='plotly_dark')

# PCA - 3 komponenty
pca = PCA(n_components=3)
data_pca = pca.fit_transform(data_std)
print(data_pca[:5])

pca_3 = pd.DataFrame(data={'pca_1': data_pca[:, 0], 'pca_2': data_pca[:, 1],
                           'pca_3': data_pca[:, 2], 'class': target})
pca_3.replace(0, 'Benign', inplace=True)
pca_3.replace(1, 'Malignant', inplace=True)
print(pca_3.head())

results = pd.DataFrame(data={'explained_variance_ratio': pca.explained_variance_ratio_})
results['cumulative'] = results['explained_variance_ratio'].cumsum()
results['component'] = results.index + 1
print(results)

fig = go.Figure(data=[go.Bar(x=results['component'], y=results['explained_variance_ratio'],
                             name='explained_variance_ratio'),
                      go.Scatter(x=results['component'], y=results['cumulative'],
                                 name='cumulative')],
                layout=go.Layout(title='PCA - 3 components', width=950, template='plotly_dark'))
fig.show()

px.scatter_3d(pca_3, x='pca_1', y='pca_2', z='pca_3', color='class', symbol='class',
              opacity=0.7, size_max=10, width=950, template='plotly_dark')

# Zbiór danych MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')

X_train = X_train[:5000]
y_train = y_train[:5000]

print(X_train[0])
print(y_train[:5])

plt.figure(figsize=(12, 8))
for i in range(8):
    plt.subplot(240 + i + 1)
    plt.imshow(X_train[i], cmap='gray_r')
    plt.title(y_train[i], color='white', fontsize=17)
    plt.axis('off')
plt.show()

X_train = X_train / 255.
X_test = X_test / 255.
print(X_train.shape)

X_train = X_train.reshape(-1, 28 * 28)
print(X_train.shape)


pca = PCA(n_components=3)

X_train_pca = pca.fit_transform(X_train)
print(X_train_pca[:5])


results = pd.DataFrame(data={'explained_variance_ratio': pca.explained_variance_ratio_})
results['cumulative'] = results['explained_variance_ratio'].cumsum()
results['component'] = results.index + 1
print(results)

fig = go.Figure(data=[go.Bar(x=results['component'], y=results['explained_variance_ratio'],
                             name='explained_variance_ratio'),
                      go.Scatter(x=results['component'], y=results['cumulative'],
                                 name='cumulative')],
                layout=go.Layout(title='PCA - 2 components', width=950, template='plotly_dark'))
fig.show()

px.scatter(pca_2, 'pca_1', 'pca_2', color=pca_2['class'], width=950, template='plotly_dark')

# Zbiór danych Cifar
from tensorflow.keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')


X_train = X_train[:5000]
y_train = y_train[:5000]
print(X_train[0].shape)

print(y_train[:5])

targets = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
           5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

plt.imshow(X_train[1])
plt.title(targets[y_train[1][0]], color='white', fontsize=17)
plt.axis('off')
plt.show()

plt.figure(figsize=(12, 8))
for i in range(8):
    plt.subplot(240 + i + 1)
    plt.imshow(X_train[i])
    plt.title(targets[y_train[i][0]], color='white', fontsize=17)
    plt.axis('off')
plt.show()

X_train = X_train / 255.
X_test = X_test / 255.
print(X_train.shape)

X_train = X_train.reshape(-1, 32 * 32 * 3)
print(X_train.shape)

print(X_train[:5])

pca = PCA(n_components=3)

X_train_pca = pca.fit_transform(X_train)
print(X_train_pca[:5])

results = pd.DataFrame(data={'explained_variance_ratio': pca.explained_variance_ratio_})
results['cumulative'] = results['explained_variance_ratio'].cumsum()
results['component'] = results.index + 1
print(results)

fig = go.Figure(data=[go.Bar(x=results['component'], y=results['explained_variance_ratio'],
                             name='explained_variance_ratio'),
                      go.Scatter(x=results['component'], y=results['cumulative'],
                                 name='cumulative')],
                layout=go.Layout(title='PCA - 3 components', width=950, template='plotly_dark'))
fig.show()

X_train_pca_df = pd.DataFrame(np.c_[X_train_pca, y_train],
                              columns=['pca_1', 'pca_2', 'pca_3', 'class'])
X_train_pca_df['name'] = X_train_pca_df['class'].map(targets)
X_train_pca_df['class'] = X_train_pca_df['class'].astype('str')
print(X_train_pca_df.head())


px.scatter_3d(X_train_pca_df, x='pca_1', y='pca_2', z='pca_3', color='name',
              symbol='name', opacity=0.7, size_max=10, width=950, height=700,
              title='PCA - CIFAR dataset', template='plotly_dark')

pca = PCA(n_components=0.95)

X_train_pca = pca.fit_transform(X_train)
print(X_train_pca[:5])

print(pca.n_components_)

print(pca.explained_variance_ratio_)

results = pd.DataFrame(data={'explained_variance_ratio': pca.explained_variance_ratio_})
results['cumulative'] = results['explained_variance_ratio'].cumsum()
results['component'] = results.index + 1
print(results.head())

fig = go.Figure(data=[go.Bar(x=results['component'], y=results['explained_variance_ratio'],
                             name='explained_variance_ratio'),
                      go.Scatter(x=results['component'], y=results['cumulative'],
                                 name='cumulative')],
                layout=go.Layout(title=f'PCA - {pca.n_components_} components', width=950,
                                 template='plotly_dark'))
fig.show()

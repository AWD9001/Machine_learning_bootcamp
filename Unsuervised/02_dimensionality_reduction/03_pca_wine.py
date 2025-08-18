# Import bibliotek
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

np.set_printoptions(precision=4, suppress=True, edgeitems=5, linewidth=200)

# Załadowanie danych

df_raw = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
                     header=None)
df = df_raw.copy()
df.head()


data = df.iloc[:, 1:]
target = df.iloc[:, 0]
data.head()

target.value_counts()

# Podział na zbiór treningowy i testowy
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, target)

print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')

# Standaryzacja

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)
print(X_train_std[:5])

# PCA

from sklearn.decomposition import PCA

pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
print(X_train_pca.shape)

results = pd.DataFrame(data={'explained_variance_ratio': pca.explained_variance_ratio_})
results['cumulative'] = results['explained_variance_ratio'].cumsum()
results['component'] = results.index + 1
print(results)

fig = go.Figure(data=[go.Bar(x=results['component'], y=results['explained_variance_ratio'],
                             name='explained variance ratio'),
                      go.Scatter(x=results['component'], y=results['cumulative'],
                                 name='cumulative explained variance')],
                layout=go.Layout(title=f'PCA - {pca.n_components_} components', width=950,
                                 template='plotly_dark'))
fig.show()

X_train_pca_df = pd.DataFrame(data=np.c_[X_train_pca, y_train],
                              columns=['pca1', 'pca2', 'pca3', 'target'])
print(X_train_pca_df.head())

px.scatter_3d(X_train_pca_df, x='pca1', y='pca2', z='pca3', color='target',
              template='plotly_dark', width=950)

print(X_train_pca[:5])

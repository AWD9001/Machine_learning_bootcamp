# Import bibliotek
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

np.set_printoptions(precision=8, suppress=True, edgeitems=5, linewidth=200)

# Wygenerowanie danych
raw_data = load_iris()
data = raw_data['data']
target = raw_data['target']
feature_names = list(raw_data['feature_names'])
feature_names = [name.replace(' ', '_')[:-5] for name in feature_names]
df = pd.DataFrame(data=np.c_[data, target], columns=feature_names + ['class'])
df['class'] = df['class'].map({0.0: 'setosa', 1.0: 'versicolor', 2.0: 'virginica'})

# Wizualizacja danych

px.scatter_3d(df, x='sepal_length', y='petal_length', z='petal_width', template='plotly_dark',
              title='Iris data - wizualizacja 3D (sepal_length, petal_length, petal_width)',
              color='class', symbol='class', opacity=0.5, width=950, height=700)

# Standaryzacja
X = df.iloc[:, [0, 2, 3]]
y = df.iloc[:, -1]

scaler = StandardScaler()
X_std = scaler.fit_transform(X)
print(X_std[:5])

# Implementacja PCA

# macierz kowariancji
cov_mat = np.cov(X_std, rowvar=False)
print(cov_mat)

# wektory własne i odpowiadające nim wartości własne macierzy kowariancji
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print(f'Wartości własne:\n{eig_vals}\n')
print(f'Wektory własne:\n{eig_vecs}')

# posortowanie wektorów według wartości własnych
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
eig_pairs.sort(reverse=True)
print(eig_pairs)

# obliczenie wartości procentowej wyjaśnionej wariancji
total = sum(eig_vals)
explained_variance_ratio = [(i / total) for i in sorted(eig_vals, reverse=True)]
print(explained_variance_ratio)

cumulative_explained_variance = np.cumsum(explained_variance_ratio)
print(cumulative_explained_variance)

results = pd.DataFrame(data={'explained_variance_ratio': explained_variance_ratio})
results['cumulative'] = results['explained_variance_ratio'].cumsum()
results['component'] = results.index + 1
print(results)

fig = go.Figure(data=[go.Bar(x=results['component'], y=results['explained_variance_ratio'],
                             name='explained variance ratio'),
                      go.Scatter(x=results['component'], y=results['cumulative'],
                                 name='cumulative explained variance')],
                layout=go.Layout(title='PCA - 3 components', width=950, template='plotly_dark'))
fig.show()

print(eig_pairs)

# obliczenie wartości procentowej wyjaśnionej wariancji
total = sum(eig_vals)
explained_variance_ratio = [(i / total) for i in sorted(eig_vals, reverse=True)]
print(explained_variance_ratio)

cumulative_explained_variance = np.cumsum(explained_variance_ratio)
print(cumulative_explained_variance)

results = pd.DataFrame(data={'explained_variance_ratio': explained_variance_ratio})
results['cumulative'] = results['explained_variance_ratio'].cumsum()
results['component'] = results.index + 1
print(results)

fig = go.Figure(data=[go.Bar(x=results['component'], y=results['explained_variance_ratio'],
                             name='explained variance ratio'),
                      go.Scatter(x=results['component'], y=results['cumulative'],
                                 name='cumulative explained variance')],
                layout=go.Layout(title='PCA - 3 components', width=950, template='plotly_dark'))
fig.show()

print(eig_pairs)

# 2 komponenty, W - macierz składająca się z 2 wektorów własnych mających największą wartość własną
W = np.hstack((eig_pairs[0][1].reshape(3, 1), eig_pairs[1][1].reshape(3, 1)))
print(W)

X_pca = X_std.dot(W)
pca_df = pd.DataFrame(data=X_pca, columns=['pca_1', 'pca_2'])
pca_df['class'] = df['class']
pca_df['pca_2'] = - pca_df['pca_2']
print(pca_df)

px.scatter(pca_df, 'pca_1', 'pca_2', color='class', width=950, template='plotly_dark')

# PCA - scikit-learn
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)
pca_df = pd.DataFrame(data=X_pca, columns=['pca_1', 'pca_2'])
pca_df['class'] = df['class']
print(pca_df)

px.scatter(pca_df, 'pca_1', 'pca_2', color='class', width=950, template='plotly_dark')

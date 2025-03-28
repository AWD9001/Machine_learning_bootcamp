# Import bibliotek
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
import plotly.express as px

sns.set(font_scale=1.2)

# Wygenerowanie danych
from sklearn.datasets import make_blobs

data = make_blobs(n_samples=10, centers=2, cluster_std=1.0,
                  center_box=(-8.0, 8.0), random_state=42)[0]
df = pd.DataFrame(data, columns=['x1', 'x2'])
df.head()

# Wizualizacja danych
plt.figure(figsize=(14, 7))
plt.scatter(data[:, 0], data[:, 1])

for label, x, y in zip(range(1, 11), data[:, 0], data[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(-3, 3), textcoords='offset points',
                 ha='right', va='bottom')
plt.title('Grupowanie hierarchiczne')
plt.show()

# Wizualizacja - dendrogram
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage

# przeprowadzenie grupowania hierarchicznego
linked = linkage(data)

# wizualizacja grupowania hierarchicznego jako dendrogram
plt.figure(figsize=(14, 7))
dendrogram(linked, orientation='top', labels=range(1, 11),
           distance_sort='descending', show_leaf_counts=True)
plt.title('Grupowanie hierarchiczne - dendrogram')
plt.show()

plt.figure(figsize=(14, 7))
dendrogram(linked, orientation='right', labels=range(1, 11),
           distance_sort='descending', show_leaf_counts=True)
plt.title('Grupowanie hierarchiczne - dendrogram')
plt.show()

# Grupowanie hierarchiczne
# bottom-up approach
cluster = AgglomerativeClustering(n_clusters=2)
cluster.fit_predict(data)

# Wizualizacja klastrów
df = pd.DataFrame(data, columns=['x1', 'x2'])
df['cluster'] = cluster.labels_

fig = px.scatter(df, 'x1', 'x2', 'cluster', width=950, height=500, template='plotly_dark',
                 title='Grupowanie hierarchiczne', color_continuous_midpoint=0.6)
fig.update_traces(marker_size=12)
fig.show()

# Porównanie metryk (euklidesowa, Manhattan, kosinusowa)
data = make_blobs(n_samples=1000, centers=4, cluster_std=1.5,
                  center_box=(-8.0, 8.0), random_state=42)[0]
df = pd.DataFrame(data, columns=['x1', 'x2'])

px.scatter(df, 'x1', 'x2', width=950, height=500,
           title='Grupowanie hierarchiczne', template='plotly_dark')

# Odległość euklidesowa
cluster_euclidean = AgglomerativeClustering(n_clusters=4)
cluster_euclidean.fit_predict(data)

df_euclidean = pd.DataFrame(data, columns=['x1', 'x2'])
df_euclidean['cluster'] = cluster_euclidean.labels_
fig = px.scatter(df_euclidean, 'x1', 'x2', 'cluster', width=950, height=500,
                 template='plotly_dark', title='Grupowanie hierarchiczne - metryka euklidesowa',
                 color_continuous_midpoint=0.6)
fig.show()

# Metryka Manhattan
cluster_manhattan = AgglomerativeClustering(n_clusters=4, affinity='manhattan', linkage='complete')
cluster_manhattan.fit_predict(data)

df_manhattan = pd.DataFrame(data, columns=['x1', 'x2'])
df_manhattan['cluster'] = cluster_manhattan.labels_
fig = px.scatter(df_manhattan, 'x1', 'x2', 'cluster', width=950, height=500, template='plotly_dark',
                 title='Grupowanie hierarchiczne - metryka Manhattan',
                 color_continuous_midpoint=0.6)
fig.show()

# Odległość kosinusowa
cluster_cosine = AgglomerativeClustering(n_clusters=4, affinity='cosine', linkage='complete')
cluster_cosine.fit_predict(data)

df_cosine = pd.DataFrame(data, columns=['x1', 'x2'])
df_cosine['cluster'] = cluster_cosine.labels_
fig = px.scatter(df_cosine, 'x1', 'x2', 'cluster', width=950, height=500,
                 template='plotly_dark', title='Grupowanie hierarchiczne - metryka kosinusowa',
                 color_continuous_midpoint=0.6)
fig.show()

from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=3, column_titles=['euclidean', 'Manhattan', 'cosine'])

fig1 = px.scatter(df_euclidean, 'x1', 'x2', 'cluster')['data'][0]
fig2 = px.scatter(df_manhattan, 'x1', 'x2', 'cluster')['data'][0]
fig3 = px.scatter(df_cosine, 'x1', 'x2', 'cluster')['data'][0]

fig.add_trace(fig1, row=1, col=1)
fig.add_trace(fig2, row=1, col=2)
fig.add_trace(fig3, row=1, col=3)
fig.update_layout(template='plotly_dark')
fig.show()

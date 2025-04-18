# Import bibliotek
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

# Wygenerowanie danych i wizualizacja
data = make_blobs(n_samples=1000, centers=3, cluster_std=1.2,
                  center_box=(-8.0, 8.0), random_state=42)[0]
df = pd.DataFrame(data, columns=['x1', 'x2'])
px.scatter(df, 'x1', 'x2', width=950, height=500, title='Klasteryzacja',
           template='plotly_dark')

# DBSCAN
cluster = DBSCAN(eps=0.5, min_samples=5)
cluster.fit(data)

print(cluster.labels_[:10])

# DBSCAN - wizualizacja

df['cluster'] = cluster.labels_
px.scatter(df, 'x1', 'x2', 'cluster', width=950, height=500, title='DBSCAN(eps=0.5, min_samples=5)',
           template='plotly_dark', color_continuous_midpoint=0)

cluster = DBSCAN(eps=0.5, min_samples=7)
cluster.fit(data)

df['cluster'] = cluster.labels_
px.scatter(df, 'x1', 'x2', 'cluster', width=950, height=500, title='DBSCAN(eps=0.5, min_samples=7)',
           template='plotly_dark')

cluster = DBSCAN(eps=0.8, min_samples=5)
cluster.fit(data)

df['cluster'] = cluster.labels_
px.scatter(df, 'x1', 'x2', 'cluster', width=950, height=500, title='DBSCAN(eps=0.7, min_samples=5)',
           template='plotly_dark')

# BSCAN - 4 klastry
data = make_blobs(n_samples=1000, centers=4, cluster_std=1.2, center_box=(-8.0, 8.0),
                  random_state=43)[0]
df = pd.DataFrame(data, columns=['x1', 'x2'])
px.scatter(df, 'x1', 'x2', width=950, height=500, title='DBSCAN', template='plotly_dark')

cluster = DBSCAN(eps=0.5, min_samples=5)
cluster.fit(data)

df['cluster'] = cluster.labels_
px.scatter(df, 'x1', 'x2', 'cluster', width=950, height=500, title='DBSCAN(eps=0.5, min_samples=5)',
           template='plotly_dark')

df.cluster.value_counts()

cluster = DBSCAN(eps=0.8, min_samples=5)
cluster.fit(data)

df['cluster'] = cluster.labels_
px.scatter(df, 'x1', 'x2', 'cluster', width=950, height=500, title='DBSCAN(eps=0.8, min_samples=5)',
           template='plotly_dark')

cluster = DBSCAN(eps=0.6, min_samples=6)
cluster.fit(data)

df['cluster'] = cluster.labels_
px.scatter(df, 'x1', 'x2', 'cluster', width=950, height=500, title='DBSCAN(eps=0.6, min_samples=6)',
           template='plotly_dark')

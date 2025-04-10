# Import bibliotek
import numpy as np
import pandas as pd
import plotly.express as px

# Wygenerowanie danych i wizualizacja
from sklearn.datasets import make_blobs

data = make_blobs(n_samples=1000, centers=3, cluster_std=1.2,
                  center_box=(-8.0, 8.0), random_state=42)[0]
df = pd.DataFrame(data, columns=['x1', 'x2'])
px.scatter(df, 'x1', 'x2', width=950, height=500, title='Klasteryzacja',
           template='plotly_dark')

# DBSCAN
from sklearn.cluster import DBSCAN

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

# Import bibliotek
import numpy as np
import pandas as pd
import plotly.express as px

# Wygenerowanie danych i wizualizacja
from sklearn.datasets import make_blobs

blobs_data = make_blobs(n_samples=1000, cluster_std=0.7, random_state=24, center_box=(-4.0, 4.0))[0]
blobs = pd.DataFrame(blobs_data, columns=['x1', 'x2'])
px.scatter(blobs, 'x1', 'x2', width=950, height=500, title='blobs data', template='plotly_dark')

from sklearn.datasets import make_circles

circle_data = make_circles(n_samples=1000, factor=0.5, noise=0.05)[0]
circle = pd.DataFrame(circle_data, columns=['x1', 'x2'])
px.scatter(circle, 'x1', 'x2', width=950, height=500, title='circle data', template='plotly_dark')

random_data = np.random.rand(1500, 2)
random = pd.DataFrame(random_data, columns=['x1', 'x2'])
px.scatter(random, 'x1', 'x2', width=950, height=500, title='random data', template='plotly_dark')

# Porównanie algorytmów - blobs data - 3 klastry
from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.01)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)
kmeans.fit(blobs_data)
clusters = kmeans.predict(blobs_data)
blobs['cluster'] = clusters
trace1 = px.scatter(blobs, 'x1', 'x2', 'cluster', width=800, height=500)['data'][0]
fig.add_trace(trace1, row=1, col=1)

from sklearn.cluster import AgglomerativeClustering

agglo = AgglomerativeClustering(n_clusters=3, affinity='euclidean')
clusters = agglo.fit_predict(blobs_data)
blobs['cluster'] = clusters
trace2 = px.scatter(blobs, 'x1', 'x2', 'cluster', width=800, height=500)['data'][0]
fig.add_trace(trace2, row=1, col=2)

from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(blobs_data)
clusters = dbscan.labels_
blobs['cluster'] = clusters
trace3 = px.scatter(blobs, 'x1', 'x2', 'cluster', width=800, height=500)['data'][0]
fig.add_trace(trace3, row=1, col=3)

fig.update_layout(title='KMeans vs. Agglomerative Clustering vs. DBSCAN - blobs data',
                  template='plotly_dark', coloraxis={'colorscale': 'viridis'})
fig.show()

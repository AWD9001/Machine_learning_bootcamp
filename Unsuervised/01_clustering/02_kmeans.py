# Import bibliotek
import pandas as pd
import plotly.express as px
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Wygenerowanie danych
data = make_blobs(n_samples=1000, centers=None, cluster_std=1.0,
                  center_box=(-4.0, 4.0), random_state=42)[0]
df = pd.DataFrame(data, columns=['x1', 'x2'])
df.head()

# Wizualizacja danych
px.scatter(df, 'x1', 'x2', width=950, height=500, title='Klasteryzacja - Algorytm K-średnich')

# Algorytm K-średnich
y_kmeans = KMeans(n_clusters=4)
y_kmeans.fit(data)

df['y_kmeans'] = y_kmeans
df.head()

# Wizualizacja klastrów
px.scatter(df, 'x1', 'x2', 'y_kmeans', width=950, height=500,
           title='Algorytm K-średnich - 3 klastry')

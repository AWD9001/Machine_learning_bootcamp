# Import bibliotek
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=1.3)

# Wygenerowanie danych
from sklearn.datasets import make_blobs

data = make_blobs(n_samples=1000, centers=4, cluster_std=1.5,
                  center_box=(-8.0, 8.0), random_state=42)[0]
df = pd.DataFrame(data, columns=['x1', 'x2'])

px.scatter(df, 'x1', 'x2', width=950, height=500, title='Algorytm K-średnich',
           template='plotly_dark')

# Algorytm K-średnich
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5)
kmeans.fit(data)

# wcss
print(kmeans.inertia_)

y_kmeans = kmeans.predict(data)
df['y_kmeans'] = y_kmeans
df.head()

px.scatter(df, 'x1', 'x2', 'y_kmeans', width=950, height=500,
           title='Algorytm K-średnich - 5 klastrów', template='plotly_dark')

# WCSS - Within - Cluster Sum - of - Squared
wcss = []
for i in range(2, 10):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)

print(wcss)

wcss = pd.DataFrame(wcss, columns=['wcss'])
wcss = wcss.reset_index()
wcss = wcss.rename(columns={'index': 'clusters'})
wcss['clusters'] += 1
wcss.head()

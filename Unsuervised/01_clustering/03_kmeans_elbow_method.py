# Import bibliotek
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

sns.set(font_scale=1.3)

# Wygenerowanie danych
data = make_blobs(n_samples=1000, centers=4, cluster_std=1.5,
                  center_box=(-8.0, 8.0), random_state=42)[0]
df = pd.DataFrame(data, columns=['x1', 'x2'])

px.scatter(df, 'x1', 'x2', width=950, height=500, title='Algorytm K-średnich',
           template='plotly_dark')

# Algorytm K-średnich
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

# Metoda "łokcia" - Elbow Method
px.line(wcss, x='clusters', y='wcss', width=950, height=500,
        title='Within-Cluster-Sum of Squared Errors (WCSS)', template='plotly_dark')

# Wizualizacja klastrów
kmeans = KMeans(n_clusters=3)
kmeans.fit(data)

y_kmeans = kmeans.predict(data)
df['y_kmeans'] = y_kmeans

px.scatter(df, 'x1', 'x2', 'y_kmeans', width=950, height=500,
           title='Algorytm K-średnich - 3 klastry',  template='plotly_dark')

centers = pd.DataFrame(data=kmeans.cluster_centers_, columns=['c1', 'c2'])
print(centers)

fig = px.scatter(df, 'x1', 'x2', 'y_kmeans', width=950, height=500,
                 title='Algorytm K-średnich - 3 klastry', template='plotly_dark')
fig.add_trace(go.Scatter(x=centers['c1'], y=centers['c2'],
                         mode='markers', marker={'size': 12, 'color': 'LightSkyBlue',
                                                 'line': {'width': 2, 'color': 'tomato'}},
                         showlegend=False))
fig.show()

from mlxtend.plotting import plot_decision_regions

plt.figure(figsize=(15, 8))
plt.title('Granice decyzyjne')
plot_decision_regions(data, y_kmeans, clf=kmeans, legend=1)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='yellow')
plt.show()

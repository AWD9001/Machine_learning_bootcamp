# Import bibliotek
import numpy as np
from numpy.linalg import norm
import pandas as pd
import random
import plotly.express as px
import plotly.graph_objects as go

np.random.seed(42)
np.set_printoptions(precision=6)
random.seed(41)

# Wygenerowanie danych
from sklearn.datasets import make_blobs

data = make_blobs(n_samples=40, centers=2, cluster_std=1.0, center_box=(-4.0, 4.0), random_state=42)[0]
df = pd.DataFrame(data, columns=['x1', 'x2'])
df.head()

# Wizualizacja danych
fig = px.scatter(df, 'x1', 'x2', width=950, height=500, title='Algorytm K-średnich')
fig.update_traces(marker_size=12)

# Implementacja algorytmu K-średnich

# wyznaczenie wartości brzegowych
x1_min = df.x1.min()
x1_max = df.x1.max()

x2_min = df.x2.min()
x2_max = df.x2.max()

print(x1_min, x1_max)
print(x2_min, x2_max)

# losowe wygnererowanie współrzędnych centroidów
centroid_1 = np.array([random.uniform(x1_min, x1_max), random.uniform(x2_min, x2_max)])
centroid_2 = np.array([random.uniform(x1_min, x1_max), random.uniform(x2_min, x2_max)])
print(centroid_1)
print(centroid_2)

# wizualizacja tzw. punktów startowych centroidów
fig = px.scatter(df, 'x1', 'x2', width=950, height=500, title='Algorytm K-średnich - inicjalizacja centroidów')
fig.add_trace(go.Scatter(x=[centroid_1[0]], y=[centroid_1[1]], name='centroid 1', mode='markers', marker_line_width=3))
fig.add_trace(go.Scatter(x=[centroid_2[0]], y=[centroid_2[1]], name='centroid 2', mode='markers', marker_line_width=3))
fig.update_traces(marker_size=12, showlegend=False)

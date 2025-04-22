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

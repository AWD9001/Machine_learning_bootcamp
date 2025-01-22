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

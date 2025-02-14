# Import bibliotek
import pandas as pd
# import plotly.express as px

# Wygenerowanie danych
from sklearn.datasets import make_blobs

data = make_blobs(n_samples=1000, centers=None, cluster_std=1.0,
                  center_box=(-4.0, 4.0), random_state=42)[0]
df = pd.DataFrame(data, columns=['x1', 'x2'])
df.head()

# Import bibliotek
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

sns.set(font_scale=1.2)

# Wygenerowanie danych
from sklearn.datasets import make_blobs

data = make_blobs(n_samples=10, centers=2, cluster_std=1.0,
                  center_box=(-8.0, 8.0), random_state=42)[0]
df = pd.DataFrame(data, columns=['x1', 'x2'])
df.head()

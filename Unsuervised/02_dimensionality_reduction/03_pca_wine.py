# Import bibliotek
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

np.set_printoptions(precision=4, suppress=True, edgeitems=5, linewidth=200)

# Załadowanie danych

df_raw = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
                     header=None)
df = df_raw.copy()
df.head()


data = df.iloc[:, 1:]
target = df.iloc[:, 0]
data.head()

target.value_counts()


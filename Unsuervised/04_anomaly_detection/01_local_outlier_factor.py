# Import bibliotek
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
import plotly.express as px
import plotly.graph_objects as go

sns.set(font_scale=1.2)
np.random.seed(10)

# Wygenerowanie danych
data = make_blobs(n_samples=300, cluster_std=2.0, random_state=10)[0]
data[:5]

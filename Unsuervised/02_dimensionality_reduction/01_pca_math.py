# Import bibliotek
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

np.set_printoptions(precision=8, suppress=True, edgeitems=5, linewidth=200)

# Wygenerowanie danych
from sklearn.datasets import load_iris

raw_data = load_iris()
data = raw_data['data']
target = raw_data['target']
feature_names = list(raw_data['feature_names'])
feature_names = [name.replace(' ', '_')[:-5] for name in feature_names]
df = pd.DataFrame(data=np.c_[data, target], columns=feature_names + ['class'])
df['class'] = df['class'].map({0.0: 'setosa', 1.0: 'versicolor', 2.0: 'virginica'})

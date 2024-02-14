# Import bibliotek
# import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
# import plotly.express as px
import seaborn as sns
from sklearn.datasets import load_iris

np.random.seed(42)
sns.set(font_scale=1.3)

# Wczytanie danych
raw_data = load_iris()
print(raw_data.keys())

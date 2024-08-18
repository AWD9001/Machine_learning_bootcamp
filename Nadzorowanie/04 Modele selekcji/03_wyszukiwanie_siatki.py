# Grid search

# Import bibliotek

# import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import make_moons
# import pandas as pd
# import plotly.express as px

np.random.seed(42)
sns.set(font_scale=1.3)

# Wygenerowanie danych
raw_data = make_moons(n_samples=2000, noise=0.25, random_state=42)
data = raw_data[0]
target = raw_data[1]

print(f'{data[:5]}\n')
print(f'{target[:5]}')

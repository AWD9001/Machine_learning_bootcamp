import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

np.random.seed(42)

# Wczytanie danych

from sklearn.datasets import load_iris

raw_data = load_iris()
print(raw_data.keys())

all_data = raw_data.copy()

data = all_data['data']
target = all_data['target']

print(f'{data[:5]}\n')
print(f'{target[:5]}')

print(all_data['target_names'])

df = pd.DataFrame(data=np.c_[data, target], columns=all_data['feature_names'] + ['class'])
print(df.head())
print(df.info())
print(df.describe().T)
print(df['class'].value_counts())

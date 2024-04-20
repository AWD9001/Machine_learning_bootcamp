# Import bibliotek
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import seaborn as sns
# import plotly.express as px

sns.set(font_scale=1.3)
np.random.seed(42)

# Załadowanie danych
from sklearn.datasets import load_iris

raw_data = load_iris()
all_data = raw_data.copy()

data = all_data['data']
target = all_data['target']
feature_names = all_data['feature_names']
target_names = all_data['target_names']

df = pd.DataFrame(data=np.c_[data, target], columns=feature_names + ['target'])
df.head()

data = data[:, [0, 1]]
target = df['target'].apply(int).values

print(f'{data[:5]}\n')
print(f'{target[:5]}')

# Las losowy
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(data, target)

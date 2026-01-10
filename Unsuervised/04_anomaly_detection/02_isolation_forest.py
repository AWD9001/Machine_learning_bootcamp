# Import bibliotek
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# Wczytanie danych
data = pd.read_csv('https://storage.googleapis.com/esmartdata-courses-files/ml-course/factory.csv')
data.head()

data.describe()

# Wizualizacja danych
px.scatter(data, x='item_length', y='item_width', width=950, template='plotly_dark',
           title='Isolation Forest')

# Isolation Forset
from sklearn.ensemble import IsolationForest

# contamination in [0, 0.05]
outlier = IsolationForest(n_estimators=100, contamination=0.05)
print(outlier.fit(data))

y_pred = outlier.predict(data)
print(y_pred[:30])

# Isolation Forset - wizualizacja
data['outlier_flag'] = y_pred
px.scatter(data, x='item_length', y='item_width', color='outlier_flag', width=950,
           template='plotly_dark', color_continuous_midpoint=-1, title='Isolation Forest')

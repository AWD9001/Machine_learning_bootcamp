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
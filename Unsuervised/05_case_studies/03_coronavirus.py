# Import bibliotek
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

np.random.seed(42)

# Wczytanie danych

# dane od 22.01.2020 do 17.02.2020
url = 'https://storage.googleapis.com/esmartdata-courses-files/ml-course/coronavirus.csv'
data = pd.read_csv(url, parse_dates=['Date', 'Last Update'])
data.head()

# Eksploracja i przygotowanie danych
data.info()
data.isnull().sum()

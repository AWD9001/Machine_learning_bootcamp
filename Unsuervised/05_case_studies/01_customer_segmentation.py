# Import bibliotek

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Załadowanie danych

url = 'https://storage.googleapis.com/esmartdata-courses-files/ml-course/OnlineRetail.csv'
raw_data = pd.read_csv(url, encoding='latin', parse_dates=['InvoiceDate'])
data = raw_data.copy()
data.head(3)

# Eksploracja danych

data.info()

data.describe()

data.describe(include=['object'])

data.describe(include=['datetime'])

data.isnull().sum()

# usunięcie braków
data = data.dropna()
data.isnull().sum()

data['Country'].value_counts()

tmp = data['Country'].value_counts().reset_index()
tmp.columns = ['Country', 'Count']
tmp.query("Count > 200", inplace=True)
px.bar(tmp, x='Country', y='Count', template='plotly_dark', color_discrete_sequence=['#03fcb5'],
       title='Częstotliwość zakupów ze względu na kraj')

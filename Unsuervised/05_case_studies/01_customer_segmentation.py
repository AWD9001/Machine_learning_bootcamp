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

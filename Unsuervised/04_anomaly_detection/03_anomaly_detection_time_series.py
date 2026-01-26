# Import bibliotek
import numpy as np
import pandas as pd
import plotly.express as px
import fbprophet

np.random.seed(41)
fbprophet.__version__

# Załadowanie danych
df = pd.read_csv('https://storage.googleapis.com/esmartdata-courses-files/ml-course/traffic.csv',
                 parse_dates=['timestamp'])
df.head()

df.info()

# Wizualizacja danych
px.line(df, x='timestamp', y='count', title='Anomaly Detection - web traffic', width=950,
        height=500, template='plotly_dark', color_discrete_sequence=['#42f5d4'])

px.scatter(df, x='timestamp', y='count', title='Anomaly Detection - web traffic', width=950,
           height=500, template='plotly_dark', color_discrete_sequence=['#42f5d4'])

# Przygotowanie danych
df.head(3)

data = df.copy()
data.columns = ['ds', 'y']
data.head(3)

# Prophet - budowa modelu
from fbprophet import Prophet

print(Prophet?)

model = Prophet(daily_seasonality=True, yearly_seasonality=False, weekly_seasonality=False,
                interval_width=0.99, changepoint_range=0.8)

model.fit(data)
forecast = model.predict(data)

forecast.head(3)

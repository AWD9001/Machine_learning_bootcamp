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

forecast[['ds', 'trend', 'yhat', 'yhat_lower', 'yhat_upper']].head(3)

forecast['real'] = data['y']
forecast['anomaly'] = 1
forecast.loc[forecast['real'] > forecast['yhat_upper'], 'anomaly'] = -1
forecast.loc[forecast['real'] < forecast['yhat_lower'], 'anomaly'] = -1
forecast.head(3)

# Wizualizacja działania modelu
px.scatter(forecast, x='ds', y='real', color='anomaly', color_continuous_scale='Bluyl',
           title='Anomaly Detection in Time Series', template='plotly_dark', width=950, height=500)

future = model.make_future_dataframe(periods=1440, freq='Min')
print(future)

forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()

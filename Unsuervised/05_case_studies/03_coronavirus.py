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

# brak Province/State -> Country
data['Province/State'] = np.where(data['Province/State'].isnull(), data['Country'],
                                  data['Province/State'])
data.isnull().sum()

data['Country'].value_counts().nlargest(10)

data['Country'] = np.where(data['Country'] == 'Mainland China', 'China', data['Country'])
data['Country'].value_counts().nlargest(10)

tmp = data['Country'].value_counts().nlargest(15).reset_index()
tmp.columns = ['Country', 'Count']
tmp = tmp.sort_values(by=['Count', 'Country'], ascending=[False, True])
tmp['iso_alpha'] = ['CHN', 'USA', 'AUS', 'CAN', 'JPN', 'KOR', 'THA', 'HKG',
                    np.nan, 'SGP', 'TWN', 'VNM', 'FRA', 'MYS', 'NPL']

print(tmp)

px.scatter_geo(tmp, locations='iso_alpha', size='Count', size_max=40, template='plotly_dark',
               color='Count', text='Country', projection='natural earth',
               color_continuous_scale='reds', width=950,
               title='Liczba przypadków Koronawirusa na świcie - TOP15')

px.scatter_geo(tmp, locations='iso_alpha', size='Count', size_max=40, template='plotly_dark',
               color='Count', text='Country', projection='natural earth',
               color_continuous_scale='reds', scope='asia', width=950,
               title='Liczba przypadków Koronawirusa - Azja (z TOP15 global)')

px.bar(tmp, x='Country', y='Count', template='plotly_dark', width=950,
       color_discrete_sequence=['#42f5c8'],
       title='Liczba przypadków Koronawirusa w rozbiciu na kraje')


px.bar(tmp.query("Country != 'China'"), x='Country', y='Count', template='plotly_dark',
       width=950, color_discrete_sequence=['#42f5c8'],
       title='Liczba przypadków Koronawirusa w rozbiciu na kraje (poza Chinami)')

tmp = data.groupby(by=data['Date'].dt.date)[['Confirmed', 'Deaths',
                                             'Recovered']].sum().reset_index()

print(tmp)

fig = go.Figure()

trace1 = go.Scatter(x=tmp['Date'], y=tmp['Confirmed'], mode='markers+lines', name='Confirmed')
trace2 = go.Scatter(x=tmp['Date'], y=tmp['Deaths'], mode='markers+lines', name='Deaths')
trace3 = go.Scatter(x=tmp['Date'], y=tmp['Recovered'], mode='markers+lines', name='Recovered')

fig.add_trace(trace1)
fig.add_trace(trace2)
fig.add_trace(trace3)

fig.update_layout(template='plotly_dark', width=950, title='Koronawirus (22.01-17.02.2020)')

data_confirmed = tmp[['Date', 'Confirmed']]
data_confirmed.columns = ['ds', 'y']
data_confirmed.head()

fig = go.Figure()

fig.add_trace(go.Scatter(x=data_confirmed['ds'], y=data_confirmed['y'], mode='markers+lines',
                         name='Confirmed', fill='tozeroy'))
fig.update_layout(template='plotly_dark', width=950,
                  title='Liczba potwierdzonych przypadków (22.01-12.102)')

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

print(data.info())

print(data.describe())

print(data.describe(include=['object']))

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

# obcięcie tylko do United Kingdom
data_uk = data.query("Country == 'United Kingdom'").copy()
data_uk.head()


# utworzenie nowej zmiennej Sales
data_uk['Sales'] = data_uk['Quantity'] * data_uk['UnitPrice']
data_uk.head()

# częstotliwość zakupów ze względu na datę
tmp = data_uk.groupby(data_uk['InvoiceDate'].dt.date)['CustomerID'].count().reset_index()
tmp.columns = ['InvoiceDate', 'Count']
tmp.head()

from plotly.subplots import make_subplots

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)

trace1 = px.line(tmp, x='InvoiceDate', y='Count', template='plotly_dark',
                 color_discrete_sequence=['#03fcb5'])['data'][0]
trace2 = px.scatter(tmp, x='InvoiceDate', y='Count', template='plotly_dark',
                    color_discrete_sequence=['#03fcb5'])['data'][0]

fig.add_trace(trace1, row=1, col=1)
fig.add_trace(trace2, row=2, col=1)
fig.update_layout(template='plotly_dark', title='Częstotliwość zakupów ze względu na datę',
                  width=950)
fig.show()

data_uk.head()


# Łączna sprzedaż ze względu na datę
tmp = data_uk.groupby(data_uk['InvoiceDate'].dt.date)['Sales'].sum().reset_index()
tmp.columns = ['InvoiceDate', 'Sales']
tmp.head()

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)

trace1 = px.line(tmp, x='InvoiceDate', y='Sales', template='plotly_dark',
                 color_discrete_sequence=['#03fcb5'])['data'][0]
trace2 = px.scatter(tmp, x='InvoiceDate', y='Sales', template='plotly_dark',
                    color_discrete_sequence=['#03fcb5'])['data'][0]

fig.add_trace(trace1, row=1, col=1)
fig.add_trace(trace2, row=2, col=1)
fig.update_layout(template='plotly_dark', title='Łączna sprzedaż ze względu na datę', width=950)
fig.show()

# Wyznacznie retencji klienta
# wydobycie unikalnych wartości CustomerID
data_user = pd.DataFrame(data['CustomerID'].unique(), columns=['CustomerID'])
data_user.head(3)

# wydobycie daty ostatniego zakupu dla każdego klienta
last_purchase = data_uk.groupby('CustomerID')['InvoiceDate'].max().reset_index()
last_purchase.columns = ['CustomerID', 'LastPurchaseDate']
last_purchase.head()

last_purchase['LastPurchaseDate'].max()

last_purchase['LastPurchaseDate'].min()

# wyznaczenie retencji jako liczby dni od daty ostatniego
# kupna klienta do maksymalnej (ostatniej) daty kupna w danych
last_purchase['Retention'] = (last_purchase['LastPurchaseDate'].max() -
                              last_purchase['LastPurchaseDate']).dt.days
last_purchase.head()

last_purchase['Retention'].value_counts()

px.histogram(last_purchase, x='Retention', template='plotly_dark',
             width=950, height=500, title='Retention', nbins=100,
             color_discrete_sequence=['#03fcb5'])

# połaczenie CustomerID oraz retencji
data_user = pd.merge(data_user, last_purchase, on='CustomerID')
data_user = data_user[['CustomerID', 'Retention']]
data_user.head()

px.scatter(data_user, x='CustomerID', y='Retention', template='plotly_dark',
           width=950, color_discrete_sequence=['#03fcb5'])

data_retention = data_user[['Retention']]
data_retention.head()

# standaryzacja danych
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_user['RetentionScaled'] = scaler.fit_transform(data_retention)
print(data_user.head())

px.scatter(data_user, x='CustomerID', y='RetentionScaled', template='plotly_dark', width=950,
           color_discrete_sequence=['#03fcb5'])

data_retention_scaled = data_user[['RetentionScaled']]
data_retention_scaled.head()

# Retencja - KMeans
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, max_iter=1000)
    kmeans.fit(data_retention_scaled)
    wcss.append(kmeans.inertia_)

wcss = pd.DataFrame(data=np.c_[range(1, 10), wcss], columns=['NumberOfClusters', 'WCSS'])
print(wcss)

px.line(wcss, x='NumberOfClusters', y='WCSS', template='plotly_dark', title='WCSS',
        width=950, color_discrete_sequence=['#03fcb5'])

kmeans = KMeans(n_clusters=3, max_iter=1000)
kmeans.fit(data_retention_scaled)

data_user['Cluster'] = kmeans.labels_
print(data_user.head())

tmp = data_user.groupby('Cluster')['Retention'].describe()
print(tmp)


tmp = tmp['mean'].reset_index()
tmp.columns = ['Cluster', 'MeanRetention']
px.bar(tmp, x='Cluster', y='MeanRetention', template='plotly_dark', width=950,
       height=400, color_discrete_sequence=['#03fcb5'])

px.scatter(data_user, x='CustomerID', y='Retention', color='Cluster', template='plotly_dark',
           width=950, title='Wizualizacja klastrów')

tmp = data_user['Cluster'].value_counts().reset_index()
tmp.columns = ['Cluster', 'Count']
px.bar(tmp, x='Cluster', y='Count', template='plotly_dark', color_discrete_sequence=['#03fcb5'],
       width=950,
       title='Rozkład częstości klastrów')

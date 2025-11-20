# Import bibliotek
import pandas as pd

pd.set_option('display.float_format', lambda x: f'{x:.2f}')

# Załadownaie danych
!wget https://storage.googleapis.com/esmartdata-courses-files/ml-course/products.csv
!wget https://storage.googleapis.com/esmartdata-courses-files/ml-course/orders.csv

products = pd.read_csv('products.csv', usecols=['product_id', 'product_name'])
print(products.head())


orders = pd.read_csv('orders.csv', usecols=['order_id', 'product_id'])
print(orders.head())


# Przygotowanie danych
data = pd.merge(orders, products, how='inner', on='product_id', sort=True)
data = data.sort_values(by='order_id')
data.head()

data.describe()

# rozkład produktów
data['product_name'].value_counts()

# liczba transakcji
data['order_id'].nunique()

transactions = data.groupby(by='order_id')['product_name'].apply(lambda name: ','.join(name))
print(transactions)

transactions = transactions.str.split(',')
print(transactions)

# Kodowanie transakcji
from mlxtend.preprocessing import TransactionEncoder

encoder = TransactionEncoder()
encoder.fit(transactions)
transactions_encoded = encoder.transform(transactions, sparse=True)
print(transactions_encoded)

transactions_encoded_df = pd.DataFrame(transactions_encoded.toarray(), columns=encoder.columns_)
print(transactions_encoded_df)

# Algorytm Apriori
from mlxtend.frequent_patterns import apriori, association_rules

supports = apriori(transactions_encoded_df, min_support=0.01, use_colnames=True, n_jobs=-1)
supports = supports.sort_values(by='support', ascending=False)
supports.head(10)

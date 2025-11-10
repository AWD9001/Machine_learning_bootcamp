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

# Import bibliotek
import pandas as pd

pd.set_option('display.float_format', lambda x: f'{x:.2f}')

# Załadownaie danych
!wget https://storage.googleapis.com/esmartdata-courses-files/ml-course/products.csv
!wget https://storage.googleapis.com/esmartdata-courses-files/ml-course/orders.csv


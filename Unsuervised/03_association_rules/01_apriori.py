# Import bibliotek
import numpy as np
import pandas as pd

# Wygenerowanie danych
data = {'produkty': ['chleb jajka mleko', 'mleko ser', 'chleb masło ser', 'chleb jajka']}

transactions = pd.DataFrame(data=data, index=[1, 2, 3, 4])
print(transactions)

# Przygotowanie danych
# rozwinięcie kolumny do obiektu DataFrame
expand = transactions['produkty'].str.split(expand=True)
print(expand)


# wydobycie nazw wszystkich produktów
products = []
for col in expand.columns:
    for product in expand[col].unique():
        if product is not None and product not in products:
            products.append(product)

products.sort()
print(products)

transactions_encoded = np.zeros((len(transactions), len(products)), dtype='int8')
print(transactions_encoded)

# kodowanie 0-1
for row in zip(range(len(transactions)), transactions_encoded, expand.values):
    for idx, product in enumerate(products):
        if product in row[2]:
            transactions_encoded[row[0], idx] = 1

print(transactions_encoded)

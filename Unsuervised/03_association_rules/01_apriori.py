# Import bibliotek
import numpy as np
import pandas as pd

# Wygenerowanie danych
data = {'produkty': ['chleb jajka mleko', 'mleko ser', 'chleb masło ser', 'chleb jajka']}

transactions = pd.DataFrame(data=data, index=[1, 2, 3, 4])
print(transactions)

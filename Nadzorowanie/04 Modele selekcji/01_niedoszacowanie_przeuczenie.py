# Import bibliotek
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons
# import plotly.express as px

np.random.seed(42)
sns.set(font_scale=1.3)

# Wygenerowanie danych
# n_samples=700
raw_data = make_moons(n_samples=2000, noise=0.25, random_state=42)
data = raw_data[0]
target = raw_data[1]

print(f'{data[:5]}\n')
print(f'{target[:5]}')

df = pd.DataFrame(data=np.c_[data, target], columns=['x1', 'x2', 'target'])
print(df.head())

# Wizualizacja danych
plt.figure(figsize=(10, 8))
plt.scatter(data[:, 0], data[:, 1], c=target, cmap='viridis')
plt.title('Klasyfikacja - dane do modelu')
plt.show()

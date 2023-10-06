# Import bibliotek
# import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
import seaborn as sns
import sklearn

np.random.seed(42)
np.set_printoptions(precision=6, suppress=True, edgeitems=30, linewidth=120,
                    formatter=dict(float=lambda x: f'{x:.2f}'))
sns.set(font_scale=1.3)
print(f'sklearn.__version__: {sklearn.__version__}\n')

# Wygenerowanie danych
from sklearn.datasets import make_regression

data, target = make_regression(n_samples=100, n_features=1, n_targets=1, noise=30.0,
                               random_state=42)

print(f'data shape: {data.shape}\n')
print(f'target shape: {target.shape}')

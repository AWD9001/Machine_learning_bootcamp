# Import bibliotek
import numpy as np
# import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.datasets import make_regression

sns.set(font_scale=1.3)
np.random.seed(42)
np.set_printoptions(precision=6, suppress=True)
print(sklearn.__version__)

# Wygenerowanie danych
data, target = make_regression(n_samples=200, n_features=1, noise=20)
target = target ** 2

print(f'{data[:5]}\n')
print(target[:5])

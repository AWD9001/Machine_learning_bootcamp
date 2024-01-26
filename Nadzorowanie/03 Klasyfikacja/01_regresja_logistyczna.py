# Import bibliotek
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
# import plotly.figure_factory as ff
import seaborn as sns
import sklearn

sns.set(font_scale=1.3)
np.set_printoptions(precision=6, suppress=True, edgeitems=10, linewidth=100000,
                    formatter=dict(float=lambda x: f'{x:.2f}'))
np.random.seed(42)
print(sklearn.__version__)

# Regresja Logistyczna (Logistic Regression) - wprowadzenie


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


X = np.arange(-5, 5, 0.1)
y = sigmoid(X)

plt.figure(figsize=(8, 6))
plt.plot(X, y)
plt.title('Funkcja Sigmoid')
plt.show()

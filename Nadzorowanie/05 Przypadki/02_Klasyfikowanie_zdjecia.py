# Import bibliotek
# %tensorflow_version 2.x
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets.fashion_mnist import load_data

np.set_printoptions(precision=12, suppress=True, linewidth=150)
pd.options.display.float_format = '{:.6f}'.format
sns.set(font_scale=1.3)

# Załadowanie danych i wstępna eksploracja
(X_train, y_train), (X_test, y_test) = load_data()

print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}')
print(f'X_train[0] shape: {X_train[0].shape}')

print(X_train[0])

print(y_train[:10])

plt.imshow(X_train[0], cmap='gray_r')
plt.axis('off')

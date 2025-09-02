# Import bibliotek
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from keras.datasets import mnist

# Wygenerowanie danych
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')

X_train = X_train[:5000]
y_train = y_train[:5000]

plt.figure(figsize=(12, 8))
for i in range(8):
    plt.subplot(240 + i + 1)
    plt.imshow(X_train[i], cmap='gray_r')
    plt.title(y_train[i], color='white', fontsize=17)
    plt.axis('off')
plt.show()

X_train = X_train.reshape(-1, 28 * 28)
print(X_train.shape)

X_train = X_train / 255.

# PCA

from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)
print(X_train_pca.shape)

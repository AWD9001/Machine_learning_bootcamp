import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import plotly.figure_factory as ff
from sklearn import datasets
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix

sns.set(font_scale=1.3)
np.random.seed(42)

# Wczytanie danych
raw_digits = datasets.load_digits()
digits = raw_digits.copy()
digits.keys()

images = digits['images']
targets = digits['target']
print(f'images shape: {images.shape}')
print(f'targets shape: {targets.shape}')

print(images[0])

# Wyświetlenie kilku obrazów
plt.figure(figsize=(12, 10))
for index, (image, target) in enumerate(list(zip(images, targets))[:6]):
    plt.subplot(2, 6, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap='Greys')
    plt.title(f'Label: {target}')

# Przygotowanie danych do modelu

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(images, targets)

print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

print()
print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')

print(X_train[0])

print(y_train[0])

# Trenowanie modelu kernel = 'linear'

from sklearn.svm import SVC

classifier = SVC(gamma=0.001, kernel='linear')
classifier.fit(X_train, y_train)

print(classifier.score(X_test, y_test))

# Trenowanie modelu kernel = 'rbf'
classifier = SVC(gamma=0.001, kernel='rbf')
classifier.fit(X_train, y_train)

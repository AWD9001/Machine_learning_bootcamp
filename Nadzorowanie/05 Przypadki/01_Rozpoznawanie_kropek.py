import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
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

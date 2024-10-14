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

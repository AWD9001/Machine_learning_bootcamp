# Import bibliotek
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image

sns.set(font_scale=1.3)
np.random.seed(42)

# Wskaźnik Giniego
Image('https://storage.googleapis.com/esmartdata-courses-files/ml-course/Picture1.png')

# Gini dla korzenia(root) - poziom0
print(1 - (50 / 150) ** 2 - (50 / 150) ** 2 - (50 / 150) ** 2)
# Poziom 1, węzeł lewy
print(1 - (45 / 52)**2 - (6 / 52)**2 - (1 / 52)**2)
# Poziom 1, węzeł prawy
print(1 - (5 / 98)**2 - (44 / 98)**2 - (49 / 98)**2)

# Entropia
Image('https://storage.googleapis.com/esmartdata-courses-files/ml-course/Picture2.png')

# Entropia dla korzenia (root) - poziom 0
print(-((50 / 150) * np.log2(50 / 150) + (50 / 150) * np.log2(50 / 150) + (50 / 150) *
        np.log2(50 / 150)))
# Poziom 1, węzeł lewy
print(-((47 / 59) * np.log2(47 / 59) + (11 / 59) * np.log2(11 / 59) + (1 / 59) * np.log2(1 / 59)))
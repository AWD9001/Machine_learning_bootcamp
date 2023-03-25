# drzewa decyzyjne

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image

sns.set(font_scale=1.3)

# wskaźnik Giniego

print(1 - (50 / 150)**2 - (50 / 150)**2 - (50 / 150)**2)  # dla korzenia - poziom 0

print(1 - (45 / 52)**2 - (6 / 52)**2 - (1 / 52)**2)  # poziom 1 - węzeł lewy

print(1 - (5 / 98)**2 - (44 / 98)**2 - (49 / 98)**2)  # poziom 1 - węzeł prawy


# Entropia

print(-((50 / 150) * np.log2(50 / 150) + (50 / 150) * np.log2(50 / 150) + (50 / 150) *
        np.log2(50 / 150)))

print(-((47 / 59) * np.log2(47 / 59) + (11 / 59) * np.log2(11 / 59) + (1 / 59) * np.log2(1 / 59)))

print(-((3 / 91) * np.log2(3 / 91) + (39 / 91) * np.log2(39 / 91) + (49 / 91) * np.log2(49/ 91)))

from scipy.stats import entropy

print(entropy([0.5, 0.5], base=2))
print(entropy([0.8, 0.2], base=2))
print(entropy([0.95, 0.05], base=2))

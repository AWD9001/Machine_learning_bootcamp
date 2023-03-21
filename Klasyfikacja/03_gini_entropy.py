# drzewa decyzyjne

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image

sns.set(font_scale=1.3)

# wskaźnik Giniego


1 - (50 / 150)**2 - (50 / 150)**2 - (50 / 150)**2  # dla korzenia - poziom 0

1 - (45 / 52)**2 - (6 / 52)**2 - (1 / 52)**2  # poziom 1 - węzeł lewy

1 - (5 / 98)**2 - (44 / 98)**2 - (49 / 98)**2  # poziom 1 - węzeł prawy
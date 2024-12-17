# Import bibliotek
import numpy as np
import pandas as pd
import plotly.express as px
import sklearn

np.random.seed(42)
np.set_printoptions(precision=6, suppress=True, edgeitems=10, linewidth=1000,
                    formatter=dict(float=lambda x: f'{x:.2f}'))
print(sklearn.__version__)

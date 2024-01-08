# Import bibliotek
import numpy as np
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go

np.random.seed(42)

y_true = 100 + 20 * np.random.randn(50)
print(y_true)

y_pred = y_true + 10 * np.random.randn(50)
print(y_pred)

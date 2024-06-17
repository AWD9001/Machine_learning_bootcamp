# Import bibliotek
import numpy as np
import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

# Metryki - Klasyfikacja binarna
# Accuracy - Dokładność klasyfikacji
# Accuracy = (correct predictions / total predictions) * 100
y_true = np.array([1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1,
                   1, 0, 1])
y_pred = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1,
                   1, 0, 1])

from sklearn.metrics import accuracy_score

accuracy_score(y_true, y_pred)

results = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
print(results)

results = results.sort_values(by='y_true')
results = results.reset_index(drop=True)
results['sample'] = results.index + 1
print(results)

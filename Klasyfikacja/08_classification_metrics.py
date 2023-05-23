# Import bibliotek
import numpy as np
import pandas as pd
import plotly.graph_objects as go
# import plotly.express as px
from plotly.subplots import make_subplots

# Metryki - Klasyfikacja binarna
y_true = np.array([1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1,
                   1, 0, 1])
y_pred = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1,
                   1, 0, 1])

from sklearn.metrics import accuracy_score
accuracy_score(y_true, y_pred)

results = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
print(results, '\n')

results = results.sort_values(by='y_true')
results = results.reset_index(drop=True)
results['sample'] = results.index + 1
print(results)

fig = make_subplots(rows=2, cols=1)
fig.add_trace(go.Scatter(x=results['sample'], y=results['y_true'], mode='markers', name='y_true'),
              row=1, col=1)
fig.add_trace(go.Scatter(x=results['sample'], y=results['y_pred'], mode='markers', name='y_pred'),
              row=2, col=1)
fig.update_layout(width=800, height=600, title='Klasyfikator binarny')
fig.show()

# Macierz konfuzji/pomyłek
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)
print(cm)

import plotly.figure_factory as ff


def plot_confusion_matrix(cmm):
    cmm = cmm[::-1]
    cmm = pd.DataFrame(cmm, columns=['pred_0', 'pred_1'], index=['true_1', 'true_0'])

    figu = ff.create_annotated_heatmap(z=cmm.values, x=list(cmm.columns), y=list(cmm.index),
                                       colorscale='ice', showscale=True, reversescale=True)
    figu.update_layout(width=400, height=400, title='Confusion Matrix', font_size=16)
    figu.show()


plot_confusion_matrix(cm)

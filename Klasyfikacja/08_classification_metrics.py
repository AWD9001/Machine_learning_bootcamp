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

cm = confusion_matrix(y_true, y_pred)
cm_df = pd.DataFrame(cm, columns=['pred_0', 'pred_1'], index=['true_0', 'true_1'])
print(cm_df)

tn, fp, fn, tp = cm.ravel()
print(f'TN - True Negative: {tn}')
print(f'FP - False Positive: {fp}')
print(f'FN - False Negative: {fn}')
print(f'TP - True Positive: {tp}')

# False Positive Rate - Type I error
fpr = fp / (fp + tn)
print(fpr)

# False Negative Rate - Type II error
fnr = fn / (fn + tp)
print(fnr)

# Precision - ile obserwacji przewidywanych jako pozytywne są w rzeczywistości pozytywne
precision = tp / (tp + fp)
print(precision)

# Recall - jak wiele obserwacji z wzystkich poytywnych sklasyfikowaliśmy jako pozytywne
recall = tp / (tp + fn)
print(recall)

# raport klasyfikacji
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))

# Krzywa ROC
from sklearn.metrics import roc_curve

fpr, tpr, tresh = roc_curve(y_true, y_pred, pos_label=1)

roc = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
print(roc)


def plot_roc_curve(y_true, y_pred):
    # Binary classification
    from sklearn.metrics import roc_curve
    fpr, tpr, tresh = roc_curve(y_true, y_pred, pos_label=1)

    fig = go.Figure(
        data=[
            go.Scatter(x=roc['fpr'], y=roc['tpr'],
                    line_color='red',
                    name='ROC Curve'),
            go.Scatter(x=[0, 1], y=[0, 1],
                    mode='lines',
                    line_dash='dash',
                    line_color='navy')
        ],
        layout=go.Layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
                         title='ROC Curve',
                         showlegend=False,
                         width=700))
    fig.show()


plot_roc_curve(y_true, y_pred)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

np.random.seed(42)

y_true = 100 + 20 * np.random.randn(50)
print(y_true)

y_pred = y_true + 10 * np.random.randn(50)
print(y_pred)


results = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
print(results.head())

results['error'] = results['y_true'] - results['y_pred']
results['error_squared'] = results['error'] ** 2
print(results.head())

# Interpretacja graficzna


def plot_regression_results(y_true, y_pred):
    results = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    min = results[['y_true', 'y_pred']].min().min()
    max = results[['y_true', 'y_pred']].max().max()

    fig = go.Figure(data=[go.Scatter(x=results['y_true'], y=results['y_pred'], mode='markers'),
                    go.Scatter(x=[min, max], y=[min, max])],
                    layout=go.Layout(showlegend=False, width=800, xaxis='y_true',
                                     yaxis='y_pred', title='Regresja: y_true vs. y_pred'))
    fig.show()


plot_regression_results(y_true, y_pred)

y_true = 100 + 20 * np.random.randn(1000)
y_pred = y_true + 10 * np.random.randn(1000)

results = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
results['error'] = results['y_true'] - results['y_pred']

px.histogram(results, x='error', nbins=50, width=800)

# Mean Absolute Error - Średni błąd bezwzględny


def mean_absolute_error(y_true, y_pred):
    return abs(y_true - y_pred).sum() / len(y_true)


mean_absolute_error(y_true, y_pred)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_true, y_pred)

# Root Mean Squared Error - RMSE - Pierwiastek błędu średniokwadratowego


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(((y_true - y_pred) ** 2).sum() / len(y_true))


root_mean_squared_error(y_true, y_pred)
np.sqrt(root_mean_squared_error(y_true, y_pred))

# Max Error - Błąd maksymalny


def max_errory(y_true, y_pred):
    return abs(y_true - y_pred).max()


max_errory(y_true, y_pred)


from sklearn.metrics import max_error

max_error(y_true, y_pred)

# R2 score - współczynnik determinacji
from sklearn.metrics import r2_score

r2_score(y_true, y_pred)


def r2_score(y_true, y_pred):
    numerator = ((y_true - y_pred) ** 2).sum()
    denominator = ((y_true - y_true.mean()) ** 2).sum()
    try:
        r2 = 1 - numerator / denominator
    except ZeroDivisionError:
        print('Dzielenie przez zero')
    return r2


t = r2_score(y_true, y_pred)
print(t)

# Import bibliotek
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

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

print(f"MAE - mean absolute error: {results['error'].abs().sum() / len(results):.4f}")
print(f"MSE - mean squared error: {results['error_squared'].sum() / len(results):.4f}")
print(f"RMSE - root mean squared error: "
      f"{np.sqrt(results['error_squared'].sum() / len(results)):.4f}")

# Interpretacja graficzna


# noinspection PyTypeChecker
def plot_regression_results(y_true1, y_pred1):

    results1 = pd.DataFrame({'y_true': y_true1, 'y_pred': y_pred1})
    min1 = results[['y_true', 'y_pred']].min().min()
    max1 = results[['y_true', 'y_pred']].max().max()

    fig = go.Figure(data=[go.Scatter(x=results1['y_true'], y=results1['y_pred'], mode='markers'),
                    go.Scatter(x=[min1, max1], y=[min1, max1])],
                    layout=go.Layout(showlegend=False, width=800,
                                     xaxis_title='y_true',
                                     yaxis_title='y_pred',
                                     title='Regresja: y_true vs. y_pred'))
    fig.show()


plot_regression_results(y_true, y_pred)

y_true = 100 + 20 * np.random.randn(1000)
y_pred = y_true + 10 * np.random.randn(1000)

results = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
results['error'] = results['y_true'] - results['y_pred']

px.histogram(results, x='error', nbins=50, width=800)

# Mean Absolute Error - Średni błąd bezwzględny


def mean_absolut_error(y_true1, y_pred1):
    return abs(y_true1 - y_pred1).sum() / len(y_true1)


mean_absolut_error(y_true, y_pred)

mean_absolute_error(y_true, y_pred)

# Mean Squared Error - MSE - Błąd średniokwadratowy


def mean_squared_def_error(y_true1, y_pred1):
    return ((y_true1 - y_pred1) ** 2).sum() / len(y_true1)


mean_squared_def_error(y_true, y_pred)

mean_squared_error(y_true, y_pred)

# Root Mean Squared Error - RMSE - Pierwiastek błędu średniokwadratowego


def root_mean_squared_error(y_true1, y_pred1):
    return np.sqrt(((y_true1 - y_pred1) ** 2).sum() / len(y_true1))


root_mean_squared_error(y_true, y_pred)

np.sqrt(mean_squared_error(y_true, y_pred))

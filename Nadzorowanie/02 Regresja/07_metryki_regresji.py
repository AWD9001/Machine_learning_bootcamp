# Import bibliotek
import numpy as np
import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go

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

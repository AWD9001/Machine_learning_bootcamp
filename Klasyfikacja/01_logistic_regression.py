import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import seaborn as sns
import sklearn

sns.set(font_scale=1.3)
np.set_printoptions(precision=6, suppress=True, edgeitems=10, linewidth=100000,
                    formatter=dict(float=lambda x: f'{x:.2f}'))
np.random.seed(42)

'''
Regresja Logistyczna (Logistic Regression) - wprowadzenie
Pomimo nazwy jest to liniowy model do zadań klasyfikacyjnych. Inna nazwa Logit Regression.

Przykłady zastosowań:
przewidywanie czy mail jest spamem, czy też nie
przewidywanie czy użytkownik kliknie w reklamę
przewidywanie czy nowotwór jest złośliwy czy też nie
przewidywanie czy dłużnik spłaci wierzycielowi dług, czy też zajdzie zdarzenie default
przewidywanie czy transakcja jest próbą oszustwa
Przy budowie modelu regresji logistycznej wykorzystamy funkcję sigmoid. Definiuje się ją wzorem:
sigmoid(x) = 1/(1+e^-x)
'''

sigmoid = lambda x: 1 / (1 + np.exp(-x))
X = np.arange(-5, 5, 0.1)
y = sigmoid(X)

plt.figure(figsize=(8, 6))
plt.plot(X, y)
plt.title('Funkcja Sigmoid')
plt.show()

'''
Następnie rozważmy funkcję liniową y = w0 + w1x. Podstawiając to do funkcji sigmoid otrzymujemy:
p(x) = 1/(1 + e^(-(w0 + w1x)))
Dzięki temu przekształceniu regresja logistyczna zwraca nam wartości z przedziału (0, 1) co możemy
interpretować jako prawdopodobieństwo i na podstawie tych prawdopodobieństw przewidywać
poszczególne klasy.
'''

# Załadowanie danych
from sklearn.datasets import load_breast_cancer

raw_data = load_breast_cancer()
print(raw_data.keys(), '\n')
print(raw_data.DESCR)


all_data = raw_data.copy()

data = all_data['data']
target = all_data['target']
print(f'rozmiar data: {data.shape}')
print(f'rozmiar target: {target.shape}')

# Podział danych na zbiór treningowy i testowy
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, target)

print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}')
print(X_train)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print(X_train)
print(scaler.mean_)
print(scaler.scale_)

# Dopasowanie modelu
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predykcja na podstawie modelu
y_pred = log_reg.predict(X_test)
print(y_pred[:30])

y_prob = log_reg.predict_proba(X_test)
print(y_prob[:30])

# Ocena modelu
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_confusion_matrix

cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm)

print(f'Accuracy: {accuracy_score(y_test, y_pred)}')


def plot_confusion_matrix(cm):
    # klasyfikacja binarna
    cm = cm[::-1]
    cm = pd.DataFrame(cm, columns=['pred_0', 'pred_1'], index=['true_1', 'true_0'])

    fig = ff.create_annotated_heatmap(z=cm.values, x=list(cm.columns), y=list(cm.index),
                                      colorscale='ice', showscale=True, reversescale=True)
    fig.update_layout(width=500, height=500, title='Confusion Matrix', font_size=16)
    fig.show()


plot_confusion_matrix(cm)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
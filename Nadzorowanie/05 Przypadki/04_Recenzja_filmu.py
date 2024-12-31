# Import bibliotek
import numpy as np
import pandas as pd
import plotly.express as px
import sklearn

np.random.seed(42)
np.set_printoptions(precision=6, suppress=True, edgeitems=10, linewidth=1000,
                    formatter=dict(float=lambda x: f'{x:.2f}'))
print(sklearn.__version__)

# Pobranie danych
!wget https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/\
              corpora/movie_reviews.zip

!unzip -q movie_reviews.zip

!pwd
!ls

from sklearn.datasets import load_files

raw_movie = load_files('movie_reviews')
movie = raw_movie.copy()
movie.keys()

# Eksploracja i przygotowanie danych

print(movie['data'][:10])
print(movie['target'][:10])
print(movie['target_names'])
print(movie['filenames'][:2])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(movie['data'], movie['target'], random_state=42)

print(f'X_train: {len(X_train)}')
print(f'X_test: {len(X_test)}')

print(X_train[0])

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=3000)
X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)

print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')

# Trenowanie modelu
from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()
classifier.fit(X_train, y_train)
classifier.score(X_test, y_test)

# Ocena modelu
from sklearn.metrics import confusion_matrix

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

import plotly.figure_factory as ff


def plot_confusion_matrix(cm1):
    cm1 = cm1[::-1]
    cm1 = pd.DataFrame(cm1, columns=['negative', 'positive'], index=['positive', 'negative'])

    fig = ff.create_annotated_heatmap(z=cm1.values, x=list(cm1.columns), y=list(cm1.index),
                                      colorscale='ice', showscale=True, reversescale=True)
    fig.update_layout(width=400, height=400, title='Confusion Matrix', font_size=16)
    fig.show()


plot_confusion_matrix(cm)

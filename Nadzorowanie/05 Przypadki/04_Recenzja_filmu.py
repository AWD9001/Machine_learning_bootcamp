# Import bibliotek
import io
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import plotly.figure_factory as ff
import requests


np.random.seed(42)
np.set_printoptions(precision=6, suppress=True, edgeitems=10, linewidth=1000,
                    formatter=dict(float=lambda x: f'{x:.2f}'))
print(sklearn.__version__)

# Pobranie danych
url = "https://raw.githubusercontent.com/nltk/" \
      "nltk_data/gh-pages/packages/corpora/movie_reviews.zip"

response = requests.get(url)

zip_data = io.BytesIO(response.content)

# Zapisywanie pliku .zip na dysku
with open("movie_reviews.zip", "wb") as file:
    file.write(response.content)

# Rozpakowywanie pliku .zip
with zipfile.ZipFile("movie_reviews.zip", "r") as zip_ref:
    zip_ref.extractall()

raw_movie = load_files('movie_reviews.zip')
movie = raw_movie.copy()
movie.keys()

# Eksploracja i przygotowanie danych
print(movie['data'][:10])
print(movie['target'][:10])
print(movie['target_names'])
print(movie['filenames'][:2])

X_train, X_test, y_train, y_test = train_test_split(movie['data'], movie['target'], random_state=42)

print(f'X_train: {len(X_train)}')
print(f'X_test: {len(X_test)}')

print(X_train[0])

tfidf = TfidfVectorizer(max_features=3000)
X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)

print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')

# Trenowanie modelu
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
classifier.score(X_test, y_test)

# Ocena modelu
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)


def plot_confusion_matrix(cm1):
    cm1 = cm1[::-1]
    cm1 = pd.DataFrame(cm1, columns=['negative', 'positive'], index=['positive', 'negative'])

    fig = ff.create_annotated_heatmap(z=cm1.values, x=list(cm1.columns), y=list(cm1.index),
                                      colorscale='ice', showscale=True, reversescale=True)
    fig.update_layout(width=400, height=400, title='Confusion Matrix', font_size=16)
    fig.show()


plot_confusion_matrix(cm)

print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))

# Predykcja na podstawie modelu
new_reviews = ['It was awesome! Very interesting story.',
               'I cannot recommend this film. Short and awful.',
               'Very long and boring. Don\'t waste your time.',
               'Well-organized and quite interesting.']

new_reviews_tfidf = tfidf.transform(new_reviews)
print(new_reviews_tfidf)
print(new_reviews_tfidf.toarray())

new_reviews_pred = classifier.predict(new_reviews_tfidf)
print(new_reviews_pred)

new_reviews_prob = classifier.predict_proba(new_reviews_tfidf)
print(new_reviews_prob)

np.argmax(new_reviews_prob, axis=1)

print(movie['target_names'])

for review, target, prob in zip(new_reviews, new_reviews_pred, new_reviews_prob):
    print(f"{review} -> {movie['target_names'][target]} -> {prob[target]:.4f}")

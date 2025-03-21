# Import bibliotek
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

np.random.seed(42)
np.set_printoptions(precision=6, suppress=True, edgeitems=10, linewidth=1000,
                    formatter=dict(float=lambda x: f'{x:.2f}'))
print(sklearn.__version__)

# Wygenerowanie danych
documents = [
    'Today is Friday',
    'I like Friday',
    'Today I am going to learn Python.',
    'Friday, Friday!!!'
]
print(documents)

# Wektoryzacja tekstu
vectorizer = CountVectorizer()
vectorizer.fit_transform(documents)

vectorizer.fit_transform(documents).toarray()
print(vectorizer.get_feature_names())

df = pd.DataFrame(data=vectorizer.fit_transform(documents).toarray(),
                  columns=vectorizer.get_feature_names())

print(df)
print(vectorizer.vocabulary_)

vectorizer.transform(['Friday morning']).toarray()

# Wektoryzacja tekstu - bigramy
bigram = CountVectorizer(ngram_range=(1, 2), min_df=1)    # min_df=2
bigram.fit_transform(documents).toarray()

print(bigram.vocabulary_)

df = pd.DataFrame(data=bigram.fit_transform(documents).toarray(),
                  columns=bigram.get_feature_names())
print(df)

# TFIDF Transformer
documents = [
    'Friday morning',
    'Friday chill',
    'Friday - morning',
    'Friday, Friday morning!!!'
]

print(documents)

counts = vectorizer.fit_transform(documents).toarray()
print(counts)

df = pd.DataFrame(data=vectorizer.fit_transform(documents).toarray(),
                  columns=vectorizer.get_feature_names())
print(df)

tfidf = TfidfTransformer()
tfidf.fit_transform(counts).toarray()

# TFIDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit_transform(documents).toarray()

print(tfidf_vectorizer.idf_)

# Przygotowanie danych tekstowych - przykład
raw_data = fetch_20newsgroups(subset='train', categories=['comp.graphics'], random_state=42)
raw_data.keys()

all_data = raw_data.copy()
print(all_data['data'][:5])

print(print(all_data['data'][0]))

print(all_data['target_names'])

print(all_data['target'][:10])

tfidf = TfidfVectorizer()
tfidf.fit_transform(all_data['data']).toarray()

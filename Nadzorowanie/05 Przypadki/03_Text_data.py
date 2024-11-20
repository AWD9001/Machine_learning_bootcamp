# Import bibliotek
import numpy as np
import pandas as pd
import plotly.express as px
import sklearn

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
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
vectorizer.fit_transform(documents)

vectorizer.fit_transform(documents).toarray()
vectorizer.get_feature_names()

df = pd.DataFrame(data=vectorizer.fit_transform(documents).toarray(),
                  columns=vectorizer.get_feature_names())

print(df)
print(vectorizer.vocabulary_)

vectorizer.transform(['Friday morning']).toarray()
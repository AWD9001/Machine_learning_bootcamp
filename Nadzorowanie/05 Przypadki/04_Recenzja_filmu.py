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


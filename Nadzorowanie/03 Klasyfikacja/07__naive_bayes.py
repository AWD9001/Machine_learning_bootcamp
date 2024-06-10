# Import bibliotek
import numpy as np
import pandas as pd
import sklearn
from sklearn.naive_bayes import GaussianNB

np.random.seed(42)
np.set_printoptions(precision=6, suppress=True)
print(sklearn.__version__)

# Wygenerowanie danych
pogoda = ['słonecznie', 'deszczowo', 'pochmurno', 'deszczowo', 'słonecznie', 'słonecznie',
          'pochmurno', 'pochmurno', 'słonecznie']
temperatura = ['ciepło', 'zimno', 'ciepło', 'ciepło', 'ciepło', 'umiarkowanie',
               'umiarkowanie', 'ciepło', 'zimno']

spacer = ['tak', 'nie', 'tak', 'nie', 'tak', 'tak', 'nie', 'tak', 'nie']

raw_df = pd.DataFrame(data={'pogoda': pogoda, 'temperatura': temperatura, 'spacer': spacer})
df = raw_df.copy()
print(df)

# Przygotowanie danych do modelu
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df['spacer'] = encoder.fit_transform(spacer)
print(df)

df = pd.get_dummies(df, columns=['pogoda', 'temperatura'], drop_first=True)
print(df)

data = df.copy()
target = data.pop('spacer')
print(data)

print(target)

# Klasyfikator bayesowski
model = GaussianNB()
model.fit(data, target)

model.score(data, target)

print(data.iloc[[0]])

print(model.predict(data.iloc[[0]]))

print(encoder.classes_)

print(encoder.classes_[model.predict(data.iloc[[0]])[0]])

print(model.predict_proba(data.iloc[[0]]))

# Import bibliotek
import numpy as np
import pandas as pd
import sklearn

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
df['spacer']= encoder.fit_transform(spacer)
print(df)

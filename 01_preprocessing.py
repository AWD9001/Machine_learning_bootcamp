import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = {
    'size': ['XL', 'L', 'M', 'L', 'M'],
    'color': ['red', 'green', 'blue', 'green', 'red'],
    'gender': ['female', 'male', 'male', 'female', 'female'],
    'price': [199.0, 89.0, 99.0, 129.0, 79.0],
    'weight': [500, 4500, 300, 380, 410],
    'bought': ['yes', 'no', 'yes', 'no', 'yes']
}

df_raw = pd.DataFrame(data=data)
# print(df_raw)

'''
---tworzymy kopie danych---
                            '''
df = df_raw.copy()
# print(df.info())

'''
---zmiana typów danych i wstępna eksploracja
                                            '''
for col in ['size', 'color', 'gender', 'bought']:
    df[col] = df[col].astype('category')
df['weight'] = df['weight'].astype('float')
# print(df.info())
# print(df.describe())
# print(df.describe().T)
# print(df.describe(include='category'))

'''
---Label encoder--- zmienia yes, no na 1 i 0
                    '''
le = LabelEncoder()
le.fit(df['bought'])
le.transform(df['bought'])
# print(le.classes_)
df['bought'] = le.fit_transform(df['bought'])

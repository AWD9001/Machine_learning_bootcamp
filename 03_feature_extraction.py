import pandas as pd
import pandas_datareader.data as web


def fetch_financial_data(company='AMZN'):
    # Ta funkcja pobiera oferty giełdowe.
    return web.DataReader(name=company, data_source='stooq')


df_raw = fetch_financial_data()
print(df_raw.head())

# Utworzenie kopii danych
df = df_raw.copy()
df = df[:5]
print('\n', df.info())

# Generowanie nowych zmiennych
print('\n', df.index.month)

df['day'] = df.index.day
df['month'] = df.index.month
df['year'] = df.index.year
print('\n', df)

# Dyskretyzacja zmiennej ciągłej
df = pd.DataFrame(data={'height': [175., 178.5, 185., 191., 184.5, 183., 168.]})
print('\n', '\n', df)

df['height_cat'] = pd.cut(x=df.height, bins=3)
print('\n', df)


df['height_cat'] = pd.cut(x=df.height, bins=(160, 175, 180, 195))
print('\n', df)


df['height_cat'] = pd.cut(x=df.height, bins=(160, 175, 180, 195),
                          labels=['small', 'medium', 'high'])
print('\n', df)

print('\n', pd.get_dummies(df, drop_first=True, prefix='height'))

# Ekstrakcja cech
df = pd.DataFrame(data={'lang': [['PL', 'ENG'], ['GER', 'ENG', 'PL', 'FRA'], ['RUS']]})
print('\n\n', df)

df['lang_number'] = df['lang'].apply(len)
print('\n', df)


df['PL_flag'] = df['lang'].apply(lambda x: 1 if 'PL' in x else 0)
print('\n', df)

df = pd.DataFrame(data={'website': ['wp.pl', 'onet.pl', 'google.com']})
print('\n', df)
print('\n', df.website.str.split('.', expand=True))


new = df.website.str.split('.', expand=True)
df['portal'] = new[0]
df['extension'] = new[1]
print('\n', df)

'''
              Open     High    Low   Close    Volume
Date
2023-01-27  99.530  103.485  99.53  102.24  87775614
2023-01-26  98.235   99.490  96.92   99.22  68523557
2023-01-25  92.560   97.240  91.52   97.18  94261570
2023-01-24  96.930   98.090  96.00   96.32  66929452
2023-01-23  97.560   97.780  95.86   97.52  76501103
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 5 entries, 2023-01-27 to 2023-01-23
Data columns (total 5 columns):
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   Open    5 non-null      float64
 1   High    5 non-null      float64
 2   Low     5 non-null      float64
 3   Close   5 non-null      float64
 4   Volume  5 non-null      int64
dtypes: float64(4), int64(1)
memory usage: 240.0 bytes

 None

 Int64Index([1, 1, 1, 1, 1], dtype='int64', name='Date')

               Open     High    Low   Close    Volume  day  month  year
Date
2023-01-27  99.530  103.485  99.53  102.24  87775614   27      1  2023
2023-01-26  98.235   99.490  96.92   99.22  68523557   26      1  2023
2023-01-25  92.560   97.240  91.52   97.18  94261570   25      1  2023
2023-01-24  96.930   98.090  96.00   96.32  66929452   24      1  2023
2023-01-23  97.560   97.780  95.86   97.52  76501103   23      1  2023


    height
0   175.0
1   178.5
2   185.0
3   191.0
4   184.5
5   183.0
6   168.0

    height          height_cat
0   175.0  (167.977, 175.667]
1   178.5  (175.667, 183.333]
2   185.0    (183.333, 191.0]
3   191.0    (183.333, 191.0]
4   184.5    (183.333, 191.0]
5   183.0  (175.667, 183.333]
6   168.0  (167.977, 175.667]

    height  height_cat
0   175.0  (160, 175]
1   178.5  (175, 180]
2   185.0  (180, 195]
3   191.0  (180, 195]
4   184.5  (180, 195]
5   183.0  (180, 195]
6   168.0  (160, 175]

    height height_cat
0   175.0      small
1   178.5     medium
2   185.0       high
3   191.0       high
4   184.5       high
5   183.0       high
6   168.0      small

    height  height_medium  height_high
0   175.0              0            0
1   178.5              1            0
2   185.0              0            1
3   191.0              0            1
4   184.5              0            1
5   183.0              0            1
6   168.0              0            0


                   lang
0            [PL, ENG]
1  [GER, ENG, PL, FRA]
2                [RUS]

                   lang  lang_number
0            [PL, ENG]            2
1  [GER, ENG, PL, FRA]            4
2                [RUS]            1

                   lang  lang_number  PL_flag
0            [PL, ENG]            2        1
1  [GER, ENG, PL, FRA]            4        1
2                [RUS]            1        0

       website
0       wp.pl
1     onet.pl
2  google.com

         0    1
0      wp   pl
1    onet   pl
2  google  com
'''
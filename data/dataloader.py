import pandas as pd


pathx = '~/Music/wsb/data/testx.csv'
pathy = '~/Music/wsb/data/testy.csv'


def getx():
    x = pd.read_csv(pathx)
    # x.drop_duplicates()
    x = x.sort_values('created_utc')
    return x


def gety():
    return pd.read_csv(pathy)

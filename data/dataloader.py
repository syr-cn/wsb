import pandas as pd


class Dataloader:
    def __init__(self):
        self.pathx = './testx.csv'
        self.pathy = './testy.csv'

    def getx(self):
        return pd.read_csv(self.pathx)

    def gety(self):
        return pd.read_csv(self.pathy)

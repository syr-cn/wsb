
import numpy as np
import pandas as pd

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split

import math


class Preprocessing:

    def __init__(self, args):
        self.data = 'data/9000.csv'
        # self.data = 'data/ag_news.csv'
        self.max_len = args.max_len
        self.max_words = args.max_words
        self.test_size = args.test_size

    def load_data(self):
        df = pd.read_csv(self.data)
        # df.drop(['id', 'keyword', 'location'], axis=1, inplace=True)

        X = df['body'].values
        X = [str(i) for i in X]
        Y = df[['s', 'an']].values
        Y = [[0 if math.isnan(j) else (j+1)/2
              for j in i]
             for i in Y]

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            X, Y, test_size=self.test_size)

    def prepare_tokens(self):
        self.tokens = Tokenizer(num_words=self.max_words)
        self.tokens.fit_on_texts(self.x_train)

    def sequence_to_token(self, x):
        sequences = self.tokens.texts_to_sequences(x)
        return sequence.pad_sequences(sequences, maxlen=self.max_len)

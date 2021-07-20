from torch import nn
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab

import pandas as pd
import numpy as np

import utils
import data.dataloader


class SentenceExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        token = get_tokenizer('basic_english')
        counter = Counter()

        trainx = data.dataloader.getx()
        trainy = data.dataloader.gety()

        for txt in trainx['body']:
            counter.update(token(utils.filter(txt)))
        vocab = Vocab(counter, min_freq=1)
        print([vocab[t] for t in ['this', 'is', 'an', 'example']])

    def init_weight(self):
        pass

    def forward(self):
        pass

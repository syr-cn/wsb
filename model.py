from torch import nn
from collections import Counter
from torchtext.vocab import Vocab

import pandas as pd
import numpy as np
import torch

import utils
import data.dataloader


counter = Counter()

trainx = data.dataloader.getx()
trainy = data.dataloader.gety()

for txt in trainx['body']:
    counter.update(utils.filter(txt).split())
vocab = Vocab(counter, min_freq=1)


def parse(x): return [vocab[i] for i in x.split()]
# print([vocab[t] for t in "it's is an example".split()])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SentenceExtractor(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(SentenceExtractor, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.i2h = nn.Linear(embed_dim+hidden_dim, hidden_dim)
        self.i2o = nn.Linear(embed_dim+hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1)
        self.init_weight()

    def init_weight(self):
        initrange = 0.5
        self.embedding.weight.data.uniform(-initrange, initrange)
        self.i2h.weight.data.uniform(-initrange, initrange)
        self.i2h.bias.data.zero_()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def Hidden(self):
        return torch.zeros(1, self.hidden_dim)

# library imports
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
# import spacy
# import jovian
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import string
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import mean_squared_error

reviews = pd.read_csv('ag_news.csv')
print(reviews.head())

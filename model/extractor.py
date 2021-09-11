# version:
# python==3.8.5
# torch==1.9.0
# torchtext==0.10.0

import torch
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

tokenizer = get_tokenizer('basic_english')
train_iter = AG_NEWS(split='train')
# train_iter可以直接在这里改，下面yield_tokens函数也要改
# tokenizer(str)可以把字符串分割为列表，全部转化为小写，遇到符号会单独分成一个字符（很合理），不能智能识别"'s"为"is"


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])  # 预设生词为0
# vocab实现词与编号的映射


def txt2idx(s):
    return vocab(tokenizer(s))


s = 'Here\'s the it xjtusyrnb an example!,?.'
print(tokenizer(s))
print(vocab.lookup_token(0))
print(txt2idx(s))

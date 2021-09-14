# version:
# python==3.8.5
# torch==1.9.0
# torchtext==0.10.0

import torch
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import LSTM

tokenizer = get_tokenizer('basic_english')
train_iter = AG_NEWS(split='train')
train_iter = list(train_iter)[:20]
# train_iter可以直接在这里改，下面yield_tokens函数也要改
# tokenizer(str)可以把字符串分割为列表，全部转化为小写，遇到符号会单独分成一个字符（很合理），不能智能识别"'s"为"is"


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])  # 预设生词为0
# vocab实现词与编号的映射


def txt2idx(s, device):
    return torch.tensor(vocab(tokenizer(s)), device=device, dtype=torch.int32)


def tag2vec(i, device):
    vec = [0, 0, 0, 0]
    vec[i-1] = 1
    vec = torch.tensor(vec, device=device, dtype=torch.float32)
    vec = vec.view(1, -1)
    return vec


# Test
# s = 'Here\'s the it xjtusyrnb an example!,?.'
# print(tokenizer(s))
# print(vocab.lookup_token(0))
# print(txt2idx(s))


vocab_size = len(vocab)
input_dim = 6
hidden_dim = 6
num_layers = 3
output_dim = 4
epochs = 1
# 好像没法batch优化

torch.manual_seed(2174)
# 设定随机种子

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTM.LSTM(vocab_size, input_dim, hidden_dim,
                  num_layers, output_dim, device=device).to(device)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

with torch.no_grad():
    tag, txt = train_iter[0]
    txt = txt2idx(txt, device)
    tag = tag2vec(tag, device)
    print(txt.shape, tag.shape)
    print(model(txt).shape)


def train(dataloader):
    for epoch in range(1, epochs+1):
        for(tag, txt)in train_iter:
            # reset accumulates gradients
            model.zero_grad()

            # load train data
            trainx = txt2idx(txt, device)
            trainy = tag2vec(tag, device)

            # forward
            resulty = model(trainx)

            # gradient backward
            loss = loss_function(resulty, trainy)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                print('-'*100)
                print(resulty)
                print(trainy)
        print(f'epoch {epoch}:loss={loss}')


train(train_iter)

with torch.no_grad():
    tag, txt = train_iter[0]
    txt = txt2idx(txt, device)
    tag = tag2vec(tag, device)
    print(txt, tag)
    print(model(txt))

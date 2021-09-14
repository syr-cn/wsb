import torch
import torch.nn.functional as F


class LSTM(torch.nn.Module):
    def __init__(self, vocab_size, input_dim, hidden_dim, num_layers, output_dim, device, dropout=0):
        # hidden_dim aka embedding_dim
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.embedding = torch.nn.Embedding(vocab_size, input_dim, sparse=True)
        # self.hidden = torch.randn(2, num_layers, 1, hidden_dim, device=device)
        self.lstm = torch.nn.LSTM(
            input_dim, hidden_dim, num_layers, dropout=dropout)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.init_weights(0.5)

    def init_weights(self, initrange):
        self.embedding.weight.data.uniform_(-initrange, initrange)
        # lstm自带正态初始化
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text):
        embeds = self.embedding(text)
        embeds = embeds.view(embeds.shape[0], 1, -1)
        # output, hidden = self.lstm(embeds, self.hidden)
        output, _ = self.lstm(embeds)
        output = output[-1]
        # print(lstm_out.shape)
        output = output.view(len(output), -1)
        output = self.fc(output)
        # output = F.log_softmax(output, dim=1)
        # 激活函数可改
        return output

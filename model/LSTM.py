import torch
import torch.nn.functional as F

torch.manual_seed(1)
# random seed


class LSTM(torch.nn.Module):
    def __init__(self, vocab_size, input_dim, hidden_dim, output_dim):
        # hidden_dim aka embedding_dim
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.embedding = torch.nn.Embedding(vocab_size, input_dim, sparse=True)
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.init_weights(0.5)

    def init_weight(self, initrange):
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_(-initrange, initrange)

    def forward(self, text):
        embeds = self.embedding(text)
        hidden, _ = self.lstm(embeds.view(len(text), 1, -1))
        out_vec = self.fc(hidden.view(len(text), -1))
        output = F.log_softmax(out_vec, dim=1)
        return output

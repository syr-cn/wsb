import torch
import torch.nn as nn
import torch.nn.functional as F


class RedditAnalyzer(nn.ModuleList):

    def __init__(self, args):
        super(RedditAnalyzer, self).__init__()

        self.batch_size = args.batch_size
        self.hidden_dim = args.hidden_dim
        self.LSTM_layers = args.lstm_layers
        self.fc_dim = args.fc_dim
        self.input_size = args.max_words  # embedding dimention

        self.device = torch.device(
            'cuda:0'if torch.cuda.is_available() else 'cpu')
        self.dropout = nn.Dropout(0.5)
        self.embedding = nn.Embedding(
            self.input_size, self.hidden_dim, padding_idx=0, device=self.device)
        self.lstm = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim,
                            num_layers=self.LSTM_layers, batch_first=True)
        self.fc1 = nn.Linear(in_features=self.hidden_dim,
                             out_features=self.fc_dim, device=self.device)
        self.fc2 = nn.Linear(self.fc_dim, 1, device=self.device)

    def forward(self, x):
        h = torch.zeros((self.LSTM_layers, x.size(0),
                         self.hidden_dim)).to(self.device)
        c = torch.zeros((self.LSTM_layers, x.size(0),
                         self.hidden_dim)).to(self.device)

        torch.nn.init.xavier_normal_(h)
        torch.nn.init.xavier_normal_(c)

        out = self.embedding(x)
        out, (hidden, cell) = self.lstm(out, (h, c))
        out = self.dropout(out)
        # out = torch.relu_(self.fc1(out[:, -1, :]))
        out = torch.relu_(self.fc1(out[:, -1]))
        out = self.dropout(out)
        out = torch.sigmoid(self.fc2(out))
        return out

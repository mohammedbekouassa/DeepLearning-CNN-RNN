import torch.nn as nn

class RowLSTM(nn.Module):
    def __init__(self, hidden=128, layers=2, num_classes=10, bidir=True):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=28, hidden_size=hidden, num_layers=layers,
            batch_first=True, bidirectional=bidir
        )
        mult = 2 if bidir else 1
        self.fc = nn.Linear(mult * hidden, num_classes)

    def forward(self, x):        # x: [B,1,28,28]
        x = x.squeeze(1)         # -> [B,28,28]  (sequence length=28, features=28)
        out, _ = self.rnn(x)     # -> [B,28,H*mult]
        hT = out[:, -1, :]       # last timestep
        return self.fc(hT)

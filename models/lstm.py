import torch.nn as nn

class RowLSTM(nn.Module):
    """
    Traite l'image 28x28 comme séquence de 28 lignes de taille 28.
    """
    def __init__(self, hidden=128, num_layers=1, num_classes=10):
        super().__init__()
        self.lstm = nn.LSTM(input_size=28, hidden_size=hidden,
                            num_layers=num_layers, batch_first=True)
        self.cls = nn.Linear(hidden, num_classes)

    def forward(self, x):
        # x: [B,1,28,28] -> [B,28,28]
        x = x.squeeze(1)
        out, _ = self.lstm(x)      # [B,28,H]
        return self.cls(out[:, -1])  # dernier pas de temps

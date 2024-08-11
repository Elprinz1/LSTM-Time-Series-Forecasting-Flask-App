import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer, num_stacked_layers):
        super().__init__()
        self.hidden_layer = hidden_layer
        self.num_stacked_layers = num_stacked_layers
        self.lstm = nn.LSTM(input_size, hidden_layer,
                            num_stacked_layers, batch_first=True)
        self.fc = nn.Linear(hidden_layer, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size,
                         self.hidden_layer).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size,
                         self.hidden_layer).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out

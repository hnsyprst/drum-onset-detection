import torch
import torch.nn as nn

class RecurrentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=512, hidden_size=128, num_layers=2)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=16, num_layers=2)
        self.hidden = nn.Linear(in_features=16, out_features=5)
        self.relu = nn.ReLU()

    def forward(self, input):
        output, _ = self.lstm1(input)
        output, _ = self.lstm2(output)
        #flattened = torch.flatten(input)
        output = self.hidden(output)
        output = self.relu(output)
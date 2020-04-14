import numpy as np
import torch
import torch.nn as nn

class AtariCNN(nn.Module):
    def __init__(self, inputlen, recurrent=False, hidden_size=512):
        super(AtariCNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(inputlen, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU()
        )
        self.fc = nn.Sequential(nn.Linear(64 * 7 * 7, hidden_size), nn.ReLU())

    def forward(self, x):
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


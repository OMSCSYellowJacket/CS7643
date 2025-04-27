import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicDecoder(nn.Module):
    def __init__(self, latent_dim=20, hidden_dim=400, output_dim=784):
        super(BasicDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act2 = nn.Sigmoid()

    def forward(self, z):
        h = self.act1(self.fc1(z))
        return self.act2(self.fc2(h))
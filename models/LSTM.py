import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTM(nn.Module):
    def __init__(self, input_size=21, hidden_size=5, num_layers=4, output_size=1, dropout=0.2):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear_layer = nn.Linear(self.hidden_size, self.output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.linear_layer(out[:, -1, :]) 
        return out
    
    def fit(self, x, y, epochs=100, lr=0.05, VERBOSE=False):
        criterion = torch.nn.MSELoss(reduction = 'mean')
        optimiser = torch.optim.Adam(self.parameters(), lr = lr)
        if VERBOSE:
            print(self.forward(x[:, :, :1]))
            print(y)
        hist = np.zeros(epochs)
        for t in range(epochs):
            pred = self.forward(x[:, :, :1])
            loss = criterion(pred, y)
            print("Epoch ", t, "Loss: ", loss.item())
            hist[t] = loss.item()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()


# class LSTM(nn.Module):

#     def __init__(self, input_size=16, hidden_size=5, num_layers=4, output_size=1, dropout=0.2):
#         super(LSTM, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.output_size = output_size
#         self.dropout = dropout

#         self.encoder = nn.Linear(self.input_size, self.hidden_size)
#         self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, dropout=self.dropout)
#         self.decoder = nn.Linear(self.hidden_size, self.output_size)
#         self.relu = nn.ReLU()
#         self.init_weights()

#     def init_weights(self):
#         for name, param in self.lstm.named_parameters():
#             if 'bias' in name:
#                  nn.init.constant_(param, 0.0)
#             elif 'weight_ih' in name:
#                  nn.init.kaiming_normal_(param)
#             elif 'weight_hh' in name:
#                  nn.init.orthogonal_(param)

#     def forward(self, x):
#         out = self.encoder(x)
#         out = self.relu(out)
#         out, _ = self.lstm(out)
#         # out = out[:, -1, :]
#         out = out.squeeze(-1)
#         return self.decoder(out)

#     def fit(self, x, val=None, epochs=10, batch_size=60, lr=1e-4, VERBOSE=False):
#         """Simple training script for """
#         optimizer = torch.optim.Adam(self.parameters(), lr=lr)
#         dataset = TensorDataset(x)
#         dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
#         for e in range(epochs):
#             total_loss = 0
#             for batch in dataloader:
#                 inputs = batch[0]
#                 loss_fxn = nn.MSELoss()
#                 pred = self.forward(inputs[:, :, :-1])
#                 print(pred.shape)
#                 print(inputs[:, :, -1].shape)
#                 loss = loss_fxn(pred, inputs[:, :, -1])
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 total_loss += loss.item()
#             if val is not None:
#                 val_loss = nn.functional.mse_loss(self.forward(val), val)
#                 if VERBOSE:
#                     print("Epoch", str(e), "Loss:", total_loss, "Val Loss:", val_loss)
#             elif VERBOSE:
#                 print("Epoch", str(e), "Loss:", total_loss)

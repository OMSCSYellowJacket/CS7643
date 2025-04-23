import torch
import torch.nn as nn


class StackedAutoEncoder(nn.Module):

    def __init__(self, input_size, hidden_size):
        """ Init function for LSTM class
            Args:
                input_size (int): the number of features in the inputs.
            Returns: 
                None
        """
        super(StackedAutoEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # layers
        # ...

        self.init_hidden()

    def init_hidden(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x: torch.Tensor):
        out = torch.zeros(x.shape[0])
        return out

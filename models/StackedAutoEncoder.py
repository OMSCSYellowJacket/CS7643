import torch
import torch.nn as nn


class AutoEncoder(nn.Module):

    def __init__(self, input_size=25, hidden_size=10, rho=0.2, gamma=0.1, beta=0.1):
        """
            input_size (int): the number of features in the inputs.
            hidden_size (int): the size of the hidden layer
            rho (float): sparsity parameter
            gamma (float): weight decay term
            beta (float): sparse penalty term
        """
        super(AutoEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rho = rho
        self.gamma = gamma
        self.beta = beta
        # layers
        self.encoder = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.decoder = nn.linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor):
        """Assumes x is of shape (sequence, feature)"""
        out = self.encoder(x)
        out = self.sigmoid(out)
        rho_hat = torch.mean(out, dim=0)
        out = self.decoder(out)
        return out, rho_hat

    def kullback_leibler_divergence(self, rho_hat):
        """Calculates the KL divergence given the mean of the hidden state."""
        return torch.sum(self.rho * torch.log(self.rho / rho_hat) + (1 - self.rho) * torch.log((1 - self.rho) / (1 - rho_hat)))

    def loss(self, output, input, rho_hat):
        """Loss function from paper"""
        loss = nn.L1Loss(reduction='sum')
        loss = 0.5 * loss(output, input)
        loss += 0.5 * self.gamma * (torch.norm(self.encoder.weight) ** 2 + torch.norm(self.decoder.weight) ** 2)
        loss += self.beta * self.kullback_leibler_divergence(rho_hat)
        return loss

    def fit(self, x, input_size=25, hidden_size=10, rho=0.2, gamma=0.1, beta=0.1, epochs=10, lr=1e-4, VERBOSE=False):
        """Simple training script for """
        model = AutoEncoder(x.size, hidden_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for e in range(epochs):
            pred, rho_hat = model.forward(x)
            loss = model.loss(pred, x, rho_hat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if VERBOSE:
                print("Epoch", str(e), "Loss:", loss)


class StackedAutoEncoder(nn.Module):

    def __init__(self, input_size=25, hidden_size=10, rho=0.2, gamma=0.1, beta=0.1):
        """
            input_size (int): the number of features in the inputs.
            hidden_size (int): the size of the hidden layer
            rho (float): sparsity parameter
            gamma (float): weight decay term
            beta (float): sparse penalty term
        """
        super(StackedAutoEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rho = rho
        self.gamma = gamma
        self.beta = beta
        # layers
        self.layers = [AutoEncoder(input_size=25, hidden_size=10, rho=0.2, gamma=0.1, beta=0.1)]

    def forward(self, x: torch.Tensor, training: bool = False):
        """Assumes x is of shape (sequence, feature)"""
        out = x
        for l in range(len(self.layers)):
            out = self.layers[l].encoder.forward(out)
            if l == len(self.layers) - 1 and training:
                out = self.layers[l].decoder.forward(out)
        return out

    def fit(self, x, input_size=25, hidden_size=10, rho=0.2, gamma=0.1, beta=0.1, epochs=10, lr=1e-4, VERBOSE=False):
        """Simple training script for """
        pass

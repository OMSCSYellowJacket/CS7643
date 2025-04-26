import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class SimpleAutoEncoder(nn.Module):

    def __init__(self, input_size=25, hidden_size=10):
        """
            input_size (int): the number of features in the inputs.
            hidden_size (int): the size of the hidden layer
            rho (float): sparsity parameter
            gamma (float): weight decay term
            beta (float): sparse penalty term
        """
        super(SimpleAutoEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # layers
        self.encoder = nn.Linear(self.input_size, self.hidden_size)
        self.hidden_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.hidden_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.hidden_3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.decoder = nn.Linear(self.hidden_size, self.input_size)
        self.sigmoid = nn.Sigmoid()
        self.batchnorm = nn.BatchNorm1d(self.hidden_size)

    def forward(self, x: torch.Tensor):
        """Assumes x is of shape (sequence, feature)"""
        self.hidden = self.batchnorm(self.sigmoid(self.encoder(x)))
        self.hidden = self.batchnorm(self.sigmoid(self.hidden_1(self.hidden)))
        self.hidden = self.batchnorm(self.sigmoid(self.hidden_2(self.hidden)))
        self.hidden = self.batchnorm(self.sigmoid(self.hidden_3(self.hidden)))
        out = self.decoder(self.hidden)
        return out

    def fit(self, x, val=None, epochs=10, batch_size=250, lr=1e-4, VERBOSE=False):
        """Simple training script for """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        dataset = TensorDataset(x)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for e in range(epochs):
            total_loss = 0
            for batch in dataloader:
                inputs = batch[0]
                loss_fxn = nn.MSELoss()
                pred = self.forward(inputs)
                loss = loss_fxn(pred, inputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if val is not None:
                val_loss = nn.functional.mse_loss(self.forward(val), val)
                if VERBOSE:
                    print("Epoch", str(e), "Loss:", total_loss, "Val Loss:", val_loss)
            elif VERBOSE:
                print("Epoch", str(e), "Loss:", total_loss)


class AutoEncoder(nn.Module):

    def __init__(self, input_size=25, hidden_size=10, output_size=None, rho=0.2, gamma=0.1, beta=0.1):
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
        if output_size:
            self.decoder = nn.Linear(hidden_size, output_size)
        else:
            self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor):
        """Assumes x is of shape (sequence, feature)"""
        self.hidden = self.encoder(x)
        self.hidden = self.sigmoid(self.hidden)
        rho_hat = torch.mean(self.hidden, dim=0)
        out = self.decoder(self.hidden)
        return out, rho_hat

    def kullback_leibler_divergence(self, rho_hat):
        """Calculates the KL divergence given the mean of the hidden state."""
        return torch.sum(self.rho * torch.log(self.rho / rho_hat) + (1 - self.rho) * torch.log((1 - self.rho) / (1 - rho_hat)))

    def loss(self, output, input, rho_hat):
        """Loss function from paper"""
        # rec_loss = nn.L1Loss(reduction='mean')
        rec_loss = nn.MSELoss(reduction='mean')
        rec_loss = 0.5 * rec_loss(output, input)
        f_loss = 0.5 * self.gamma * (torch.norm(self.encoder.weight) ** 2 + torch.norm(self.decoder.weight) ** 2)
        kl_loss = self.beta * self.kullback_leibler_divergence(rho_hat)
        loss = rec_loss + f_loss + kl_loss
        return loss, rec_loss, f_loss, kl_loss

    def fit(self, x, target=None, epochs=10, lr=1e-4, VERBOSE=False):
        """Simple training script for """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for e in range(epochs):
            pred, rho_hat = self.forward(x)
            if target is not None:
                loss, rec_loss, f_loss, kl_loss = self.loss(pred, target, rho_hat)
            else:
                loss, rec_loss, f_loss, kl_loss = self.loss(pred, x, rho_hat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if VERBOSE:
                print("Epoch", str(e), "Loss:", loss, "Rec_Loss:", rec_loss, "f_loss:", f_loss, "kl_loss:", kl_loss)


class StackedAutoEncoder(nn.Module):

    def __init__(self, number_layers=5, input_size=25, hidden_size=10, rho=0.2, gamma=0.1, beta=0.1):
        """
            input_size (int): the number of features in the inputs.
            hidden_size (int): the size of the hidden layer
            rho (float): sparsity parameter
            gamma (float): weight decay term
            beta (float): sparse penalty term
        """
        super(StackedAutoEncoder, self).__init__()
        self.number_layers = number_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rho = rho
        self.gamma = gamma
        self.beta = beta
        # layers
        self.layers = list()
        self.init_layers()

    def init_layers(self):
        self.layers.append(
            AutoEncoder(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                output_size=self.input_size,
                rho=self.rho,
                gamma=self.gamma,
                beta=self.beta
            )
        )
        for l in range(1, self.number_layers):
            self.layers.append(
                AutoEncoder(
                    input_size=self.hidden_size,
                    hidden_size=self.hidden_size,
                    output_size=self.input_size,
                    rho=self.rho,
                    gamma=self.gamma,
                    beta=self.beta
                )
            )

    def forward(self, x: torch.Tensor, training: bool = False):
        """Assumes x is of shape (sequence, feature)"""
        out = x
        for l in range(len(self.layers)):
            self.layers[l].encoder.forward(out)
            out = self.layers[l].hidden
        out = self.layers[-1].decoder.forward(out)
        return out

    def fit(self, x, epochs=10, lr=1e-4, VERBOSE=False):
        """Simple training script for """
        if VERBOSE:
            print("Fitting Layer 1")
        self.layers[0].fit(x=x, epochs=epochs, lr=lr, VERBOSE=VERBOSE)
        self.layers[0].forward(x)
        x_prime = self.layers[0].hidden.detach()
        for n in range(1, self.number_layers):
            if VERBOSE:
                print("\nFitting Layer", str(n + 1))
            self.layers[n].fit(x=x_prime, target=x, epochs=epochs, lr=lr, VERBOSE=VERBOSE)
            self.layers[n].forward(x_prime)
            x_prime = self.layers[n].hidden.detach()

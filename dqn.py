import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Deep Q-Network (DQN) model.
    Takes the state (features + position) as input and outputs Q-values for each action.
    """

    def __init__(
        self, state_size, action_size, hidden_dim1=128, hidden_dim2=64, seed=0
    ):
        """
        Initialize parameters and build model.
        Params:
        state_size (int): Dimension of each state (e.g., 96 features + 1 position)
        action_size (int): Dimension of each action (e.g., 3 for Hold, Buy, Sell)
        hidden_dim1 (int): Number of nodes in the first hidden layer
        hidden_dim2 (int): Number of nodes in the second hidden layer
        seed (int): Random seed
        """
        super(DQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size

        # Define the layers
        self.fc1 = nn.Linear(state_size, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, action_size)

        print(
            f"DQN Initialized. Input={state_size}, Hidden1={hidden_dim1}, Hidden2={hidden_dim2}, Output={action_size}"
        )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

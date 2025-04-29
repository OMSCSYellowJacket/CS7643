import random
import numpy as np
from collections import deque
import torch
import torch.optim as optim
import torch.nn.functional as F
from dqn import DQN

# Use GPU if available, otherwise CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Sort of based on:
# https://github.com/xkiwilabs/DQN-using-PyTorch-and-ML-Agents/blob/master/dqn_agent.py
# https://github.com/udacity/deep-reinforcement-learning/blob/master/dqn/exercise/dqn_agent.py
class DQNAgent:
    """Interacts with and learns in environment."""

    def __init__(
        self,
        state_size,
        action_size,
        seed=0,
        buffer_size=int(100000),  # replay buffer size
        batch_size=64,
        gamma=0.99,  # discount factor
        tau=0.001,  # for soft update of target parameters
        lr=0.0005,  # learning rate
        update_every=4,  # how often to update the local network (steps)
        target_update_every=100,  # how often to update the target network (learning steps)
    ):
        """Initialize an Agent object.

        Params:
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            buffer_size (int): replay buffer size
            batch_size (int): minibatch size
            gamma (float): discount factor
            tau (float): interpolation parameter for soft target updates
            lr (float): learning rate
            update_every (int): frequency to update the local network
            target_update_every (int): frequency to update the target network
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Hyperparameters
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.update_every = update_every
        self.target_update_every = target_update_every

        # Q-Network
        self.q_network_local = DQN(state_size, action_size, seed=seed).to(device)
        self.q_network_target = DQN(state_size, action_size, seed=seed).to(device)
        self.optimizer = optim.Adam(self.q_network_local.parameters(), lr=self.lr)
        self.update_target_network(hard_update=True)

        # Replay memory
        self.memory = deque(maxlen=buffer_size)

        # Initialize time step (update every "update_every" steps)
        self.timestep = 0
        # Initialize learning step counter (update target network)
        self.learn_step_counter = 0

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and randomly sample from buffer for learning."""
        # Save experience / reward
        self.remember(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.timestep = (self.timestep + 1) % self.update_every
        if self.timestep == 0:
            # If enough samples are available, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.sample_memory()
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0.0):
        """Returns actions for given state for current policy.

        Params:
        state (array_like): current state (features + position)
        eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.q_network_local.eval()
        with torch.no_grad():
            action_values = self.q_network_local(state)
        self.q_network_local.train()

        # Epsilon greedy action selection
        if random.random() > eps:
            # Exploit (Choose the action with the highest Q value)
            return np.argmax(action_values.cpu().data.numpy()).item()
        else:
            # Explore (choose a random action)
            return random.choice(np.arange(self.action_size))

    def remember(self, state, action, reward, next_state, done):
        """Add experience to memory."""
        # Make sure all components are NP arrays or scalars
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        action = int(action)
        reward = float(reward)
        done = bool(done)
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample_memory(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        return experiences

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params:
        experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
        gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = zip(*experiences)

        # Convert to PyTorch tensors
        states = torch.from_numpy(np.vstack(states)).float().to(device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(device)
        # Convert boolean to float
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(device)

        # Get maximum predicted Q values for next states from target
        # We do not want gradients flowing back to target network during this calculation
        Q_targets_next = (
            self.q_network_target(next_states).detach().max(1)[0].unsqueeze(1)
        )

        # Compute Q targets for current states: Q_target = r + gamma * max_a Q_target(s', a)
        # If the episode is done, future reward is 0.
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model: Q_local(s, a)
        # Select Q-value corresponding to action taken in experience tuple
        Q_expected = self.q_network_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network_local.parameters(), 1)
        self.optimizer.step()

        self.learn_step_counter = (
            self.learn_step_counter + 1
        ) % self.target_update_every
        if self.learn_step_counter == 0:
            self.update_target_network()

    def update_target_network(self, hard_update=False):
        """Soft update model parameters or perform a hard copy."""
        if hard_update:
            # Copy weights from local to target
            self.q_network_target.load_state_dict(self.q_network_local.state_dict())
        else:
            # https://arxiv.org/pdf/1509.02971
            # Use tau hyperparam to blend the two networks. This is the "soft update"
            for target_param, local_param in zip(
                self.q_network_target.parameters(), self.q_network_local.parameters()
            ):
                target_param.data.copy_(
                    self.tau * local_param.data + (1.0 - self.tau) * target_param.data
                )

    def save(self, filename_base="dqn_agent"):
        """Saves the local and target network weights."""
        torch.save(self.q_network_local.state_dict(), f"{filename_base}_qlocal.pth")
        torch.save(self.q_network_target.state_dict(), f"{filename_base}_qtarget.pth")
        print(f"Agent weights saved with base name: {filename_base}")

    def load(self, filename_base="dqn_agent"):
        """Loads the local and target network weights."""
        try:
            self.q_network_local.load_state_dict(
                torch.load(f"{filename_base}_qlocal.pth", map_location=device)
            )
            self.q_network_target.load_state_dict(
                torch.load(f"{filename_base}_qtarget.pth", map_location=device)
            )
            self.q_network_local.eval()
            self.q_network_target.eval()
            print(f"Agent weights loaded from base name: {filename_base}")
        except FileNotFoundError:
            print(
                f"Error: Could not find agent weight files with name: {filename_base}"
            )
            raise
        except Exception as e:
            print(f"Error loading agent weights: {e}")
            raise

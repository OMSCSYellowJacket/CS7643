import numpy as np
import gym
from gym import spaces


class TradingEnvironment(gym.Env):
    """
    Gym trading environment.
    Assumes input features for step 't' already contain necessary lookback history.
    """

    def __init__(
        self, features, prices, initial_capital=10000, transaction_cost_pct=0.001
    ):
        super(TradingEnvironment, self).__init__()

        # Assumes features are shape (N_days, N_features_with_lag)
        self.features = features
        # Assumes prices are shape (N_days,)
        self.prices = prices
        self.num_features = self.features.shape[1]
        self.total_steps = len(self.prices)

        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct

        # State components
        self.current_step = 0
        self.position = 0  # -1: short, 0: flat, 1: long
        self.portfolio_value = self.initial_capital
        self.history = []

        # Actions: 0 is Hold, 1 is Buy, 2 is Sell
        self.action_space = spaces.Discrete(3)

        # State size is now num_features + 1 (for position)
        self.state_size = self.num_features + 1
        # Observation space: Features for the current step + current position
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_size,), dtype=np.float32
        )

        print(
            f"Environment Initialized. Steps={self.total_steps}, Features={self.num_features}, StateSize={self.state_size}"
        )

    def get_state(self):
        """
        Returns the state: features for the current step + current position.
        """
        if self.current_step < 0 or self.current_step >= self.total_steps:
            # Handle edge case if step goes out of bounds, though 'done' should prevent this
            print(
                f"Error: get_state called with invalid current_step: {self.current_step}"
            )
            # Return zeros
            return np.zeros(self.state_size, dtype=np.float32)

        current_features = self.features[self.current_step]
        state = np.append(current_features, self.position)
        return state.astype(np.float32)

    def reset(self):
        """
        Resets the environment to the initial state
        """
        # Start at the first available step
        self.current_step = 0
        self.position = 0
        self.portfolio_value = self.initial_capital
        self.history = []
        if self.total_steps == 0:
            raise ValueError("Cannot reset environment with no data.")
        return self.get_state()

    def step(self, action):
        """
        Execute one time step within the environment
        """
        if self.current_step >= self.total_steps - 1:
            print(f"Reached end of data. Returning zero reward and done = True.")
            # Return current state as next state? Or zeros?
            next_state = self.get_state()
            return next_state, 0.0, True, {}

        current_price = self.prices[self.current_step]
        # Price for calculating reward
        next_price = self.prices[self.current_step + 1]

        prev_portfolio_value = self.portfolio_value
        prev_position = self.position

        # Execute action
        transaction_cost = 0
        if action == 1:
            # Buy
            if self.position == 0:
                self.position = 1
                transaction_cost = self.transaction_cost_pct
            elif self.position == -1:
                self.position = 1
                # Close short, open long
                transaction_cost = 2 * self.transaction_cost_pct
        elif action == 2:
            # Sell
            if self.position == 0:
                self.position = -1
                transaction_cost = self.transaction_cost_pct
            elif self.position == 1:
                self.position = -1
                # Close long, open short
                transaction_cost = 2 * self.transaction_cost_pct

        # Action 0 (Hold) does nothing to position or cost

        # Calculate reward
        step_return = (next_price / current_price) - 1 if current_price != 0 else 0
        reward = 0
        if prev_position == 1:
            # Was long before action decision
            reward = step_return
        elif prev_position == -1:
            # Was short before action decision
            reward = -step_return

        # Apply transaction cost penalty
        reward -= transaction_cost

        # Update portfolio value
        self.portfolio_value *= 1 + reward

        # Move to next step
        self.current_step += 1

        # Check if done
        # Episode ends when we cannot calculate  next step's reward/state
        done = self.current_step >= self.total_steps - 1

        # Get next state
        # If done, next_state might not matter much, but let's return the final state
        next_state = (
            self.get_state()
            if not done
            else np.zeros(self.state_size, dtype=np.float32)
        )

        # Store history (not strictly necessary)
        self.history.append(
            {
                "step": self.current_step,
                "action": action,
                "reward": reward,
                "portfolio_value": self.portfolio_value,
                "position": self.position,
                "price": next_price,
            }
        )

        info = {}

        return next_state, reward, done, info

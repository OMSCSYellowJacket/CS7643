import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import os
import random
import matplotlib.pyplot as plt
from collections import deque

from dqn import DQN
from dqn_agent import DQNAgent
from dqn_environment import TradingEnvironment

# Sort of based on https://github.com/mlpanda/DeepLearning_Financial/blob/master/run_training.py

# DQN Hyperparameters

# Replay buffer size
BUFFER_SIZE = int(100000)
# Minibatch size
BATCH_SIZE = 64
# Discount factor
GAMMA = 0.995
# For soft update of target parameters
TAU = 0.001
# Learning rate
LR = 0.005
# How often to update the local network
UPDATE_EVERY = 4
# How often to update the target network
TARGET_UPDATE_EVERY = 75
# Starting value of epsilon
EPSILON_START = 1.0
# Minimum value of epsilon
EPSILON_END = 0.1  # 0.01
# Multiplicative factor (per episode) for decreasing epsilon
EPSILON_DECAY = 0.995
# Total training episodes (adjust based on convergence)
N_EPISODES = 500

INITIAL_CAPITAL = 10000
TRANSACTION_COST_PCT = 0.001  # Example cost (0.1%)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# STEP 1: LOAD PRE-PROCESSED DATA
print("Loading pre-processed data from 'DQN/' directory...")

try:
    train_ary = np.load("DQN/train_ary.npy")
    val_ary = np.load("DQN/val_ary.npy")
    test_ary = np.load("DQN/test_ary.npy")

    # Load corresponding original closing prices
    prices_train = np.load("DQN/prices_train.npy")
    prices_val = np.load("DQN/prices_val.npy")
    prices_test = np.load("DQN/prices_test.npy")

    print("Data loaded successfully.")
    print(f"  Train array shape: {train_ary.shape}")
    print(f"  Val array shape:   {val_ary.shape}")
    print(f"  Test array shape:  {test_ary.shape}")
    print(f"  Train prices shape: {prices_train.shape}")
    print(f"  Val prices shape:   {prices_val.shape}")
    print(f"  Test prices shape:  {prices_test.shape}")

    # Basic shape validation
    assert (
        train_ary.shape[:-1] == prices_train.shape
    ), "Train features and prices shape mismatch"
    assert (
        val_ary.shape[:-1] == prices_val.shape
    ), "Validation features and prices shape mismatch"
    assert (
        test_ary.shape[:-1] == prices_test.shape
    ), "Test features and prices shape mismatch"
    assert (
        train_ary.shape[2] == 97
    ), "Train array has unexpected feature+label dimension"

except FileNotFoundError as e:
    print(f"Error: Could not load data file: {e}")
    exit()
except AssertionError as e:
    print(f"Error: Data shape validation failed: {e}")
    exit()


# Separate features (X) and labels (y) - Label is the last column
X_train = train_ary[:, :, :-1].astype(np.float32)
y_train = train_ary[:, :, -1].astype(np.float32)
X_val = val_ary[:, :, :-1].astype(np.float32)
y_val = val_ary[:, :, -1].astype(np.float32)
X_test = test_ary[:, :, :-1].astype(np.float32)
y_test = test_ary[:, :, -1].astype(np.float32)

# Aggregate data across tickers (Simple Averaging)
print("Aggregating ticker data by averaging...")
# Shape: (N_days, N_features_lagged=96)
X_train_agg = np.mean(X_train, axis=0)
X_val_agg = np.mean(X_val, axis=0)
X_test_agg = np.mean(X_test, axis=0)

# Shape: (N_days,)
prices_train_agg = np.mean(prices_train, axis=0)
prices_val_agg = np.mean(prices_val, axis=0)
prices_test_agg = np.mean(prices_test, axis=0)

print("Aggregated data shapes:")
print(f"  X_train_agg: {X_train_agg.shape}, prices_train_agg: {prices_train_agg.shape}")
print(f"  X_val_agg:   {X_val_agg.shape}, prices_val_agg:   {prices_val_agg.shape}")
print(f"  X_test_agg:  {X_test_agg.shape}, prices_test_agg:  {prices_test_agg.shape}")

# Determine state size AFTER aggregation
# Should be 96
N_features = X_train_agg.shape[1]
# Features + position
STATE_SIZE = N_features + 1
# 0: Hold, 1: Buy, 2: Sell
ACTION_SIZE = 3
print(f"State size: {STATE_SIZE}, Action size: {ACTION_SIZE}")

# STEP 2: INITIALIZE AGENT & ENV
agent = DQNAgent(
    state_size=STATE_SIZE,
    action_size=ACTION_SIZE,
    seed=0,
    buffer_size=BUFFER_SIZE,
    batch_size=BATCH_SIZE,
    gamma=GAMMA,
    tau=TAU,
    lr=LR,
    update_every=UPDATE_EVERY,
    target_update_every=TARGET_UPDATE_EVERY,
)

# Environments use the aggregated data
train_env = TradingEnvironment(
    X_train_agg,
    prices_train_agg,
    initial_capital=INITIAL_CAPITAL,
    transaction_cost_pct=TRANSACTION_COST_PCT,
)
val_env = TradingEnvironment(
    X_val_agg,
    prices_val_agg,
    initial_capital=INITIAL_CAPITAL,
    transaction_cost_pct=TRANSACTION_COST_PCT,
)
test_env = TradingEnvironment(
    X_test_agg,
    prices_test_agg,
    initial_capital=INITIAL_CAPITAL,
    transaction_cost_pct=TRANSACTION_COST_PCT,
)

# STEP 3: TRAIN THE AGENT
print("Starting DQN Training")
# list containing scores from each episode
scores = []
# last 100 scores
scores_window = deque(maxlen=100)
eps = EPSILON_START
# Track best validation score during training
best_val_score = -np.inf

for i_episode in range(1, N_EPISODES + 1):
    state = train_env.reset()
    score = 0
    episode_reward_sum = 0
    done = False

    while not done:
        action = agent.act(state, eps)
        next_state, reward, done, info = train_env.step(action)
        if state is not None and next_state is not None:
            agent.step(state, action, reward, next_state, done)
            state = next_state
            episode_reward_sum += reward
        else:
            print(f"Invalid state in episode {i_episode}. Ending early.")
            done = True

        if done:
            break

    # Final portfolio value for this episode
    score = train_env.portfolio_value
    scores_window.append(score)
    scores.append(score)
    # decrease epsilon
    eps = max(EPSILON_END, EPSILON_DECAY * eps)

    print(
        f"\rEpisode {i_episode}/{N_EPISODES} | Average Score: {np.mean(scores_window):.2f} | Epsilon: {eps:.4f}",
        end="",
    )
    if i_episode % 100 == 0:
        print(
            f"\rEpisode {i_episode}/{N_EPISODES} | Average Score: {np.mean(scores_window):.2f} | Epsilon: {eps:.4f}"
        )

        # Intermediate Validation
        state_val = val_env.reset()
        val_done = False
        val_portfolio_values = [val_env.portfolio_value]
        while not val_done:
            action_val = agent.act(state_val, eps=0.0)  # Greedy
            next_state_val, _, val_done, _ = val_env.step(action_val)
            if state_val is not None and next_state_val is not None:
                state_val = next_state_val
                val_portfolio_values.append(val_env.portfolio_value)
            else:
                print(
                    f"Error: Invalid state during validation. Ending validation episode."
                )
                val_done = True

        current_val_score = val_env.portfolio_value
        print(f"  Validation Score after Episode {i_episode}: {current_val_score:.2f}")

        # Save agent if validation score improved
        if current_val_score > best_val_score:
            print(
                f"  *** Validation score improved! ({current_val_score:.2f} > {best_val_score:.2f}). Saving agent... ***"
            )
            best_val_score = current_val_score
            agent.save("dqn_agent_checkpoint")

print("\nDQN Training Finished")

# STEP 4: TESTING & BENCHMARK
print("\nFinal Testing Phase")
try:
    # Re-initialize agent structure
    final_agent = DQNAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE, seed=999)
    # Load the best checkpoint saved during training
    final_agent.load("dqn_agent_checkpoint")
    print("Loaded best agent checkpoint for final testing.")

    print("Evaluating Best Agent on Test Set")
    state = test_env.reset()
    done = False
    final_rewards_agent = []
    final_portfolio_values_agent = [test_env.portfolio_value]

    while not done:
        # Use greedy action. Don't explore
        action = final_agent.act(state, eps=0.0)
        next_state, reward, done, _ = test_env.step(action)
        if state is not None and next_state is not None:
            state = next_state
            final_rewards_agent.append(reward)
            final_portfolio_values_agent.append(test_env.portfolio_value)
        else:
            print(f"Error: Invalid state during final testing. Ending test episode.")
            done = True

    print(f"Final Test Performance (DQN Agent):")
    print(f"  - Final Portfolio Value: {final_portfolio_values_agent[-1]:.2f}")

    # Calculate Buy and Hold Benchmark on Test Set
    print("Calculating Buy and Hold Benchmark")
    if len(prices_test_agg) > 1:
        initial_price_bh = prices_test_agg[0]
        if initial_price_bh <= 0:
            print(
                "Error: Initial price for Buy & Hold is zero or negative. Skipping benchmark."
            )
        else:
            final_portfolio_values_bh = [INITIAL_CAPITAL]
            for i in range(1, len(prices_test_agg)):
                current_price = prices_test_agg[i]
                bh_value = INITIAL_CAPITAL * (current_price / initial_price_bh)
                final_portfolio_values_bh.append(bh_value)

            print(f"Final Test Performance (Buy & Hold):")
            print(f"  - Final Portfolio Value: {final_portfolio_values_bh[-1]:.2f}")

            # Plot final test performance
            plt.figure(figsize=(14, 7))
            steps_agent = range(len(final_portfolio_values_agent))
            plt.plot(
                steps_agent,
                final_portfolio_values_agent,
                label=f"DQN Agent (Final portfolio value: ${final_portfolio_values_agent[-1]:.2f})",
                linewidth=2,
            )

            steps_bh = range(len(final_portfolio_values_bh))
            plt.plot(
                steps_bh,
                final_portfolio_values_bh,
                label=f"Buy and Hold (Final portfolio value: ${final_portfolio_values_bh[-1]:.2f})",
                linestyle="--",
                color="red",
            )

            plt.title("DQN Agent vs. Buy & Hold Performance (Test Set)")
            plt.xlabel("Time Steps in Test Period", fontsize=24)
            plt.ylabel("Portfolio Value ($)", fontsize=24)
            plt.legend(fontsize=24)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("dqn_vs_buyhold_performance_final.pdf")
            plt.close()
            print("Saved performance plot to dqn_vs_buyhold_performance_final.pdf")
    else:
        print(
            "Could not calculate Buy & Hold benchmark - aggregated test price data too short."
        )


except FileNotFoundError:
    print("Could not load 'dqn_agent_checkpoint'.")
except Exception as e:
    print(f"An error occurred during final testing: {e}")

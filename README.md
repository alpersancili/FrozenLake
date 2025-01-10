# FrozenLake
This project implements a Q-Learning agent to solve the FrozenLake-v1 environment from the OpenAI Gymnasium library. The agent navigates a slippery 8x8 grid world to reach the goal while avoiding holes.

Key features of the project include:

Reinforcement Learning: Implements the Q-learning algorithm with an epsilon-greedy exploration strategy.
Training and Testing: Allows toggling between training the agent (updating Q-table) and testing it with a pre-trained Q-table.
Visualization: Plots cumulative rewards over episodes to show learning progress and saves the plot as a PNG file.
Persistence: Saves and loads the Q-table using Python's pickle library for continued training or evaluation.
Configurable Parameters: Adjust hyperparameters like learning rate, discount factor, and exploration rate for experimentation.

How It Works:
The agent interacts with the FrozenLake environment, taking actions and receiving rewards.
During training, the Q-values are updated using the Bellman equation.
The agent uses an epsilon-greedy strategy to balance exploration and exploitation.
Training progress is visualized through cumulative rewards over the last 100 episodes.
Files:
frozen_lake_q.py: The primary script containing the Q-learning implementation.
frozen_lake8x8.png: Visualization of cumulative rewards during training.
frozen_lake8x8.pkl: Serialized Q-table for testing or resuming training.


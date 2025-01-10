# Frozen Lake Q-Learning Agent

This project implements a Q-Learning agent to solve the FrozenLake-v1 environment from OpenAI Gymnasium. The agent learns to navigate a slippery 8x8 grid world, aiming to reach the goal while avoiding holes, using reinforcement learning principles.

Features

Q-Learning Implementation: Trains the agent using the Bellman equation for Q-value updates.
Epsilon-Greedy Exploration: Balances exploration and exploitation, with a decaying epsilon for adaptive learning.
Training & Testing Modes:
Training: Initializes a new Q-table and updates it based on the agent’s interactions with the environment.
Testing: Loads a pre-trained Q-table to evaluate the agent’s performance.
Persistence: Q-table is serialized and saved using Python’s pickle library for reusability.
Visualization: Generates a plot of cumulative rewards to track learning progress and saves it as a PNG file.


How It Works

Agent Setup: The agent interacts with the FrozenLake environment, which includes 64 discrete states and 4 possible actions.
Training Phase:
Uses Q-Learning to update the Q-table based on the agent’s actions, rewards, and the environment's feedback.
Adapts exploration with epsilon decay and optimizes long-term rewards with a discount factor (gamma).
Testing Phase:
Evaluates the agent's performance using a pre-trained Q-table.
Performance Monitoring:
Plots the agent’s cumulative rewards over episodes to visualize learning progress.


Technical Details

Algorithm: Q-Learning
Libraries Used: Python, NumPy, Matplotlib, Gymnasium, Pickle
Key Parameters:
Learning Rate (Alpha): 0.9
Discount Factor (Gamma): 0.9
Exploration Rate (Epsilon): Starts at 1.0 and decays over time


Project Files

frozen_lake_q.py: Contains the full implementation of the Q-Learning agent.
frozen_lake8x8.png: Visualization of cumulative rewards during training.
frozen_lake8x8.pkl: Serialized Q-table for testing or resuming training.

https://github.com/user-attachments/assets/0ae6018a-90e0-4ddb-899d-a79a2655edfa


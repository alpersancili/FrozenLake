import gymnasium as gym  # Importing the gymnasium library for reinforcement learning environments.
import numpy as np       # Importing numpy for handling arrays and numerical operations.
import matplotlib.pyplot as plt  # Importing matplotlib for plotting results.
import pickle            # Importing pickle for saving/loading the Q-table.

def run(episodes, is_training=True, render=False):
    # Initialize the FrozenLake environment
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True, render_mode="human" if render else None)

    if is_training:
        # If training, initialize the Q-table with zeros (64 states x 4 actions)
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        # If not training, load the pre-trained Q-table from a file
        f = open("frozen_lake8x8.pkl", "rb")
        q = pickle.load(f)
        f.close()

    # Define hyperparameters
    learning_rate_a = 0.9  # Learning rate (alpha): controls the update weight for new information.
    discount_factor_g = 0.9  # Discount factor (gamma): determines the importance of future rewards.

    epsilon = 1.0  # Epsilon: initial probability of choosing a random action (exploration).
    epsilon_decay_rate = 0.0001  # Rate at which epsilon decreases over time.
    rng = np.random.default_rng()  # Random number generator for consistent randomness.

    # Array to store rewards for each episode
    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        # Reset the environment at the start of each episode
        state = env.reset()[0]  # Initial state (0 = top-left corner of the grid).
        terminated = False      # Set to True when the agent reaches the goal or falls in a hole.
        truncated = False       # Set to True if the agent takes too many steps (limit exceeded).

        while not terminated and not truncated:
            # Select an action using epsilon-greedy strategy
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()  # Random action (exploration).
            else:
                action = np.argmax(q[state, :])  # Action with the highest Q-value (exploitation).

            # Perform the action and observe the result
            new_state, reward, terminated, truncated, _ = env.step(action)

            if is_training:
                # Update the Q-value using the Bellman equation
                q[state, action] = q[state, action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
                )

            # Move to the new state
            state = new_state

        # Decrease epsilon to reduce exploration over time
        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if epsilon == 0:
            # Stabilize the Q-values when epsilon reaches zero (no exploration)
            learning_rate_a = 0.0001

        # Store reward for this episode
        if reward == 1:  # Reward = 1 means the agent reached the goal.
            rewards_per_episode[i] = 1

    # Close the environment after training/testing
    env.close()

    # Calculate cumulative rewards over the last 100 episodes
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])  # Sliding window of 100 episodes.

    # Plot cumulative rewards
    plt.plot(sum_rewards)  # Plot the cumulative rewards over episodes.
    plt.savefig("frozen_lake8x8.png")  # Save the plot as a PNG file.

    if is_training:
        # Save the Q-table to a file for future use
        f = open("frozen_lake8x8.pkl", "wb")
        pickle.dump(q, f)
        f.close()

if __name__ == "__main__": 
    # Uncomment the following line to train the agent
    # run(15000)

    # Run the agent with rendering enabled (visualize the environment)
    run(1000, is_training=True, render=True)

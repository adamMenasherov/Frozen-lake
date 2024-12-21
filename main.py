from QAgent import *
import gym

# Generate the environment
env = gym.make("FrozenLake-v1", render_mode=None)
num_of_actions = env.action_space.n
num_of_states = env.observation_space.n


# Hyper-parameters
# TODO: choose parameters
learning_episodes = 1000   # Number of learning episodes
learning_rate = 0.1       # Learning rate
max_steps = 5000             # Maximum steps per episode
epsilon = 0.5               # Exploration probability
gamma = 0.9

# TODO: Create the agent
agent = QAgent(num_of_states, num_of_actions, learning_rate, epsilon, gamma)
agent.train(env, learning_episodes, max_steps)
print(agent.q_table)
print("success rate:", agent.test(env, max_steps, learning_episodes))

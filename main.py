from QAgent import *
from constants import *
import gym

# Generate environment
env = gym.make("FrozenLake-v1")
num_of_actions = env.action_space.n
num_of_states = env.observation_space.n


# Create agent and train
agent = QAgent(num_of_states, num_of_actions, learning_rate, epsilon, gamma)
agent.train(env, learning_episodes, max_steps)
print(agent.q_table)
env = gym.make("FrozenLake-v1", render_mode='human')
print("Success rate:", agent.test(env, 100, max_steps))
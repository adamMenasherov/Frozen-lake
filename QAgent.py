import numpy as np
import gym
import random
from typing import List


def get_avlb_actions(state, n) -> List:
    avlb_actions = list(range(n))

    if state % n == 0:
        avlb_actions.remove(0)
    if state < n:
        avlb_actions.remove(3)
    if state % (n - 1) == 0:
        avlb_actions.remove(2)
    if n ** 2 - n < state < n ** 2:
        avlb_actions.remove(1)

    return avlb_actions


class QAgent:
    def __init__(self, num_of_states, num_of_actions, learning_rate, epsilon, gamma):
        self.q_table = np.zeros((num_of_states, num_of_actions))
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def train(self, env: gym.Env, n_trials, n_steps):
        for trial in range(n_trials):
            state, _ = env.reset()
            for _ in range(n_steps):
                # Choose action
                avlb_actions = get_avlb_actions(state, env.action_space.n)
                if random.uniform(0, 1) > self.epsilon:
                    psb_action = np.argmax(self.q_table[state, :])
                    action = psb_action if psb_action in avlb_actions else avlb_actions[0]
                else:
                    rand_action = env.action_space.sample()
                    action = rand_action if rand_action in avlb_actions else avlb_actions[0]

                # Perform action
                next_state, reward, done, _, _ = env.step(action)

                # Q-value update
                curr_value = self.q_table[state, action]
                new_val = (1 - self.learning_rate) * curr_value + self.learning_rate * (
                        reward + self.gamma * np.max(self.q_table[next_state, :])
                )
                self.q_table[state, action] = new_val

                state = next_state

                if done:
                    break


    def test(self, env: gym.Env, n_steps, n_test_trials):
        successes = 0
        for _ in range(n_test_trials):
            state, _ = env.reset()
            trial_reward = 0
            for _ in range(n_steps):
                action = np.argmax(self.q_table[state, :])
                next_state, reward, done, _, _ = env.step(action)
                trial_reward += reward

                if done:
                    break
                state = next_state

            if trial_reward > 0:
                successes += 1

        return (successes / n_test_trials) * 100


# Generate environment
env = gym.make("FrozenLake-v1", render_mode=None)
num_of_actions = env.action_space.n
num_of_states = env.observation_space.n

# Hyperparameters
learning_episodes = 100000
learning_rate = 0.2
max_steps = 5000
epsilon = 0.3
gamma = 0.8

# Create agent and train
agent = QAgent(num_of_states, num_of_actions, learning_rate, epsilon, gamma)
agent.train(env, learning_episodes, max_steps)

print(agent.q_table)
print("Success rate:", agent.test(env, max_steps, 500))

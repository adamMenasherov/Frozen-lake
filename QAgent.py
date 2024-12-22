import numpy as np
import gym
import random
from constants import *


class QAgent:
    def __init__(self, num_of_states, num_of_actions, learning_rate, epsilon, gamma):
        self.q_table = np.zeros((num_of_states, num_of_actions))
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def train(self, env: gym.Env, n_trials, n_steps):
        for trial in range(n_trials):
            finished_trial = False
            state, _ = env.reset()
            for _ in range(n_steps):
                # Choose action
                action = self.choose_action(env, state)
                # Perform action
                next_state, reward, done, _, _ = env.step(action)

                if done and reward == 0:  # Fell in a hole
                    reward = -1
                    finished_trial = True
                elif done and reward == 1:  # Reached goal
                    reward = 10
                    finished_trial = True

                # Q-value update
                old_value = self.q_table[state, action]
                next_max = np.max(self.q_table[next_state, :])
                new_val = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.gamma * next_max)
                self.q_table[state, action] = new_val

                state = next_state
                self.epsilon = max(min_epsilon, self.epsilon * epsilon_decay)

                if finished_trial:
                    break

    def test(self, env: gym.Env, n_test_trials, n_steps):
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

    def choose_action(self, env: gym.Env, state):
        if random.uniform(0, 1) > self.epsilon:
            action = np.argmax(self.q_table[state, :])
        else:
            action = env.action_space.sample()

        return action







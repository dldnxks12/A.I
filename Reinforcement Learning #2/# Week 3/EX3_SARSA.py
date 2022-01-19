# SARSA (On Policy Temporal Difference Control)

"""
Action space:
0 = south
1 = north
2 = east
3 = west
4 = pickup
5 = dropoff

state space : (taxi_row, taxi_col, passenger_location, destination)
"""

import gym
import sys
import random
import numpy as np
from time import sleep


def epsilon_greedy_policy(Q, state, epsilon):
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])

    return action

def SARSA(env, epsilon, n_epsiodes):
    Q = np.zeros([env.nS, env.nA])

    for _ in range(n_epsiodes):
        state = env.reset()  # Initial State
        action = epsilon_greedy_policy(Q, state, epsilon)  # initial action
        gamma = 0.9
        alpha = 0.1

        done = False
        while True:
            next_state, reward, done, info = env.step(action)
            next_action = epsilon_greedy_policy(Q, next_state, epsilon)

            Q[state][action] = ( (1-alpha) * Q[state][action] ) + (alpha*( reward + gamma*(Q[next_state][next_action]) ))

            state = next_state
            action = next_action

            if done == True:
                break
    return Q


env = gym.make('Taxi-v3').env
env = gym.wrappers.TimeLimit(env, max_episode_steps = 30)

Q = SARSA(env, epsilon = 0.3, n_epsiodes = 50000)

state = env.reset() # start state is random ...
env.render()
done = False
total_reward = 0
while not done:
    action = np.argmax(Q[state])
    next_state, reward, done, info = env.step(action)
    env.render()
    total_reward += reward
    state = next_state

print("Total Reward : ", total_reward)




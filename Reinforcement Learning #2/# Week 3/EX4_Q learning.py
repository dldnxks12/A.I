# Q learning (Off policy Learning for Temporal difference)

import gym
import sys
import numpy as np
import random
from time import sleep


def behavior_policy(random_policy, state, epsilon):
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(random_policy[state, :])

    return action

def epsilon_greedy_policy(Q, state, epsilon):
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])

    return action

def Q_Learning(env, random_policy, epsilon, n_epsiodes):
    Q = np.zeros([env.nS, env.nA])

    for e in range(n_epsiodes):
        state = env.reset()  # Initial State

        gamma = 0.9
        alpha = 0.1

        done = False
        while True:
            #action = epsilon_greedy_policy(Q, state, epsilon)
            action = behavior_policy(random_policy, state, epsilon)
            next_state, reward, done, info = env.step(action)

            Q[state][action] = ((1-alpha) * Q[state][action]) + (alpha*(reward + gamma * max(Q[next_state])))

            state = next_state

            if done == True:
                break

        print(f"Episode : {e}")
    return Q

env = gym.make('Taxi-v3').env
env = gym.wrappers.TimeLimit(env, max_episode_steps = 30)

random_policy = np.ones([env.nS, env.nA]) / env.nA

Q = Q_Learning(env, random_policy, epsilon = 0.1, n_epsiodes = 500000)

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




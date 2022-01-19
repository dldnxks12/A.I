# First Visit Monte Carlo

import gym
import sys
import numpy as np

from collections import defaultdict

env = gym.make('FrozenLake-v1')
env = gym.wrappers.TimeLimit(env, max_episode_steps = 20)

def generate_episode(policy, env):
    states, actions, rewards = [], [], []
    observation = env.reset()

    while True:
        states.append(observation)

        action_probability = policy[observation]
        action = np.random.choice(np.arange(len(action_probability)), p = action_probability)

        actions.append(action)

        observation, reward, done, info = env.step(action)

        rewards.append(reward)

        if done:
            break

    return states, actions, rewards


# Recursive Version
def first_visit_mc_prediction(env, policy, n_episodes):

    print("Version 1 : Start")
    V = np.zeros(env.nS)
    m = np.zeros(env.nS)
    gamma = 0.9 # discount factor

    for _ in range(n_episodes):
        G = 0
        states, _, rewards = generate_episode(policy, env)
        states = np.flip(states)
        rewards = np.flip(rewards)

        for i, item in enumerate( zip(states, rewards) ):
            state = item[0]
            reward = item[1]
            G = reward + gamma*G
            if state not in states[i+1 : -1]: # 해당 state가 없으면
                m[state] = m[state] + 1
                V[state] = V[state] + ((G - V[state]) / m[state])
            else:
                continue

    print("Version 1 : Done")
    return V


# Batch Version
def first_visit_mc_prediction_v2(env, policy, n_episodes):

    print("Version 2 : Start")
    V = np.zeros(env.nS)
    R = [ [] for _ in range(env.nS)]
    gamma = 0.9

    for _ in range(n_episodes):
        G = 0
        states, actions, rewards = generate_episode(policy, env)
        states = np.flip(states)
        rewards = np.flip(rewards)

        for i, item in enumerate(zip(states, rewards)):
            state = item[0]
            reward = item[1]

            G = reward + G*gamma

            if state not in states[i+1 : -1]:
                R[state].append(G)
                V[state] = np.mean(R[state])
            else:
                continue

    print("Version 2 : Done")
    return V

random_policy = np.ones([env.nS, env.nA]) / env.nA
V = first_visit_mc_prediction(env, random_policy, n_episodes = 50000)
V2 = first_visit_mc_prediction_v2(env, random_policy, n_episodes= 50000)

print("Value Function 1 ")
print(V)

print("Value Function 2 ")
print(V2)


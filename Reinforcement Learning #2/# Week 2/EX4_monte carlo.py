# First visit monte carlo prediction for Q-function

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

def first_visit_mc_prediction(env, policy, n_episodes):
    Q = np.zeros([env.nS, env.nA])
    m = np.zeros([env.nS, env.nA])
    gamma = 0.9

    for _ in range(n_episodes):
        states, actions, rewards = generate_episode(policy, env)

        states = np.flip(states)
        actions = np.flip(actions)
        rewards = np.flip(rewards)

        pairs = list(zip(states, actions))
        G = 0
        for i, items in enumerate(zip(states, actions, rewards)):
            state = items[0]
            action = items[1]
            reward = items[2]

            pair = (state, action)
            G = reward + gamma * G

            if pair in pairs[i + 1: -1]:
                m[state][action] += 1
                Q[state][action] = Q[state][action] + ((G - Q[state][action]) / m[state][action])
            else:
                continue
    return Q

random_policy = np.ones([env.nS, env.nA]) / env.nA
Q  = first_visit_mc_prediction(env, random_policy, n_episodes = 50000)

V = np.zeros(env.nS)


# Value evaluation
for s in range(env.nS):
    for a in range(env.nA):
        V[s] = V[s] + random_policy[s][a]*Q[s][a]

print("Value function 1: ", V)



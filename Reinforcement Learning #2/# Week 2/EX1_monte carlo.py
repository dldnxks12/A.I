# Every Visit Monte Carlo : Batch Version
# Every Visit Monte Carlo : Recursive Version

import gym
import sys
import numpy as np
from collections import defaultdict # 딕셔너리를 만드는 dict 클래스


env = gym.make('FrozenLake-v1')
env = gym.wrappers.TimeLimit(env, max_episode_steps = 20) # 최대 Epsiode 크기 20으로 제한

# Generating Episodes ...
def generate_episode(policy, env): # use uniformly random policy
    states, actions, rewards = [], [], []

    observation = env.reset() # observation = state_0

    while True:

        states.append(observation) # states list ...

        probs = policy[observation] # EX .... [0.25 0.25 0.25 0.25]

        action = np.random.choice(np.arange(len(probs)), p = probs)

        actions.append(action) # Action list ...

        observation, reward, done, info = env.step(action)

        rewards.append(reward) # reward list ...

        if done:
            break

    return states, actions, rewards

# Recursive Version
def every_visit_mc_prediction(env, policy, n_episodes):
    print("Version 1 : Start")
    V = np.zeros(env.nS)
    m = np.zeros(env.nS)
    gamma = 0.9 # discount factor

    for _ in range(n_episodes):
        states, actions, rewards = generate_episode(policy, env)
        G = 0
        states = np.flip(states)
        rewards = np.flip(rewards)
        for state, reward in zip(states, rewards): # states의 개수만큼 episode 진행....
            G = reward + gamma*G
            m[state] = m[state] + 1
            V[state] = V[state] + ((G - V[state]) / m[state])

    print("Version 1 : Done")
    return V

# Batch Version
def every_visit_mc_prediction_v2(env, policy, n_episodes):
    print("Version 2 : Start")
    V = np.zeros(env.nS)
    R = [ [] for _ in range(env.nS)] # 16개의 빈 list

    gamma = 0.9

    for _ in range(n_episodes):
        G = 0
        states, actions, rewards = generate_episode(policy, env)
        states = np.flip(states)
        rewards = np.flip(rewards)

        for state, reward in zip(states, rewards):
            G = reward + gamma*G
            R[state].append(G)
            V[state] = np.mean(R[state])

    print("Version 2 : Done")
    return V


random_policy = np.ones([env.nS, env.nA]) / env.nA

V = every_visit_mc_prediction(env, random_policy, n_episodes = 50000)
V2 = every_visit_mc_prediction_v2(env, random_policy, n_episodes = 50000)

print("Value Function 1 ")
print(V)

print("Value Function 2 ")
print(V2)

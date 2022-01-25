# First visit Monte Carlo Control for Q function

import gym
import numpy as np
import sys
import random

env = gym.make('FrozenLake-v1')
env = gym.wrappers.TimeLimit(env, max_episode_steps = 50)

def epsilon_greedy(state, Q, epsilon):

    if random.random() < epsilon:
        action = env.action_space.sample()

    else:
        action = np.argmax(Q[state, :])

    return action

def generate_episode(Q, epsilon, env):
    states, actions, rewards = [], [], []
    state = env.reset()

    while True:
        states.append(state)

        action = epsilon_greedy(state, Q, epsilon)
        actions.append(action)

        state, reward, done, info = env.step(action)

        rewards.append(reward)

        if done:
            break

    return states, actions, rewards

# Batch Version
def first_visit_mc_prediction(env, epsilon, n_episodes):
    Q = np.zeros([env.nS, env.nA])
    R = [[[] for _ in range(env.nA)] for _ in range(env.nS)]
    gamma = 0.9

    for _ in range(n_episodes):
        G = 0
        states, actions, rewards = generate_episode(Q, epsilon, env)

        states = np.flip(states)
        actions = np.flip(actions)
        rewards = np.flip(rewards)

        Pairs = list(zip(states, actions))

        for i, items in enumerate(zip(states, actions, rewards)):
            state  = items[0]
            action = items[1]
            reward = items[2]

            pair = (state, action)

            G = reward + gamma*G
            if pair not in Pairs[i+1 : -1]:
                R[state][action].append(G)
                Q[state][action] = np.mean(R[state][action])
            else:
                continue

    return Q

Q = first_visit_mc_prediction(env, 0.1, n_episodes=50000)

done = False
state = env.reset()
env.render()

while not done:
    action = np.argmax(Q[state]) # Optimal Policy (deterministic)
    new_state, reward, done, info = env.step(action)
    env.render()
    state = new_state
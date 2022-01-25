# Temporal Difference(0) Prediction for Value function

import gym
import sys
import numpy as np

env = gym.make('FrozenLake-v1')
env = gym.wrappers.TimeLimit(env, max_episode_steps = 20)

def TD_prediction(env, policy, n_episodes):
    V = np.zeros(env.nS)
    alpha = 0.1
    gamma = 0.9
    for _ in range(n_episodes):

        state = env.reset() # initial State ...
        done = False

        while done == False:

            action = np.random.choice(np.arange(len(policy[state])), p = policy[state])
            next_state, reward, done, info = env.step(action)

            if done == True:
                V[state] = V[state] + alpha*(reward - V[state])
            else:
                V[state] = V[state] + alpha * (reward + (gamma * V[next_state]) - V[state])

            state = next_state

    return V


random_policy = np.ones([env.nS, env.nA]) / env.nA

V = TD_prediction(env, random_policy, n_episodes=50000)


print("Value Function : ", V)
# Temporal Difference(0) Prediction for Q function

import gym
import sys
import numpy as np

env = gym.make('FrozenLake-v1')
env = gym.wrappers.TimeLimit(env, max_episode_steps = 20)

def TD_prediction(env, policy, n_episodes):
    Q = np.zeros([env.nS, env.nA])
    alpha = 0.1
    gamma = 0.9
    for _ in range(n_episodes):

        state = env.reset() # initial State ...
        done = False

        while done == False:

            action = np.random.choice(np.arange(len(policy[state])), p = policy[state])
            next_state, reward, done, info = env.step(action)
            next_action = np.random.choice(np.arange(len(policy[next_state])), p = policy[next_state])

            if done == True:
                Q[state][action] = Q[state][action] + alpha*(reward - Q[state][action])
            else:
                Q[state][action] = Q[state][action] + alpha * (reward + (gamma * Q[next_state][next_action]) - Q[state][action])

            state = next_state

    return Q


random_policy = np.ones([env.nS, env.nA]) / env.nA

Q = TD_prediction(env, random_policy, n_episodes=50000)

print("Q Value Function : ", Q)

V = np.zeros(env.nS)

for s in range(env.nS):
    for a in range(env.nA):
        V[s] = V[s] + random_policy[s][a]*Q[s][a]

print("")

print("Value Function : ", V)
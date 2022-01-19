# Double Q Learning for eliminating maximization bias problem

import gym
import sys
import random
import numpy as np
from time import sleep

def epsilon_greedy_policy(Q, state, epsilon):
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state , :])

    return action


def Double_Q_Learning(env, epsilon, n_episodes):
    Q1 = np.zeros([env.nS, env.nA])
    Q2 = np.zeros([env.nS, env.nA])

    for e in range(n_episodes):

        state = env.reset() # initial State in Episode
        gamma = 0.9
        alpha = 0.1

        done = False
        while True:
            Q = random.choice([Q1, Q2])

            action = epsilon_greedy_policy(Q, state, epsilon)

            next_state, reward, done, info = env.step(action)

            if np.all(Q == Q1):
                Q1[state][action] = ((1 - alpha) * Q1[state][action]) + (alpha * (reward + gamma * max(Q2[next_state])))

            else:
                Q2[state][action] = ((1 - alpha) * Q2[state][action]) + (alpha * (reward + gamma * max(Q1[next_state])))

            state = next_state

            if done == True:
                break

        print(f"Episode : {e}")

    if np.abs(np.mean(Q1 - Q2)) < 1:
        print("Q Value if Identical .. !")
    else:
        print("Q Value if Not Identical .. !")

    return Q1, Q2


env = gym.make('Taxi-v3').env
env = gym.wrappers.TimeLimit(env, max_episode_steps = 30)

Q1, Q2 =  Double_Q_Learning(env, epsilon=0.1, n_episodes=100000)

state = env.reset() # start state is random ...
env.render()
done = False
total_reward = 0
total_reward2 = 0

while not done:
    action = np.argmax(Q1[state])
    next_state, reward, done, info = env.step(action)
    env.render()
    total_reward += reward
    state = next_state

print("Total Reward : ", total_reward)





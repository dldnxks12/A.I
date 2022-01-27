
import gym
import sys
import math
import torch
import random
import numpy as np
from time import sleep

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Actor Network
class PolicyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fcA1 = torch.nn.Linear(8, 30)
        self.fcA2 = torch.nn.Linear(30, 10)
        self.fcA3 = torch.nn.Linear(10, 4)

    def forward(self, x): # x : state
        x = self.fcA1(x)
        x = torch.nn.functional.relu(x)
        x = self.fcA2(x)
        x = torch.nn.functional.relu(x)
        x = self.fcA3(x)
        x = torch.nn.functional.softmax(x, dim = -1)

        return x


# Critic Network
class VNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fcV1 = torch.nn.Linear(8, 126)
        self.fcV2 = torch.nn.Linear(126, 64)
        self.fcV3 = torch.nn.Linear(64, 1)

    def forward(self, x):  # x : state
        x = self.fcV1(x)
        x = torch.nn.functional.relu(x)
        x = self.fcV2(x)
        x = torch.nn.functional.relu(x)
        x = self.fcV3(x)

        return x


def generate_episode(ep):
    states, actions, rewards = [], [], []

    state = env.reset()
    done = False
    score = 0

    while not done:
        if ep % 1000 == 0:
            env.render()

        state = torch.FloatTensor(state).to(device)
        policy = pi(state)
        action = torch.multinomial(policy, 1).item()
        next_state, reward, done, info = env.step(action)

        score += reward

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = next_state

    return states, actions, rewards, score

def G_(rewards, t):
    G = 0
    for i, _ in enumerate(rewards, t):
        G += math.pow(gamma, i - t) * rewards[i]
        if i == len(rewards) - 1:
            break
    return G

alpha = 0.001
gamma = 0.99

pi = PolicyNetwork().to(device)
pi_optimizer = torch.optim.Adam(pi.parameters(), lr = alpha)

V  = VNetwork().to(device)
V_optimizer = torch.optim.Adam(V.parameters(), lr = alpha)

env = gym.make('LunarLander-v2')

scores = []
episode = 0
MAX_EPISODE = 5000

while episode < MAX_EPISODE:

    states, actions, rewards, score = generate_episode(episode)
    scores.append(score)

    target = 0
    G = 0
    loss1 = 0
    loss2 = 0

    for time_step, item in enumerate(zip(states, actions, rewards)):
        state = item[0]
        action = item[1]
        reward = item[2]

        G = G_(rewards, time_step)

        loss1 += (G - V(state)) ** 2
        loss2 += time_step*(G - V(state)) * ((pi(state)[action] + 1e-5).log())

    loss2 = -loss2
    V_optimizer.zero_grad()
    pi_optimizer.zero_grad()

    loss1.backward()
    loss2.backward()

    V_optimizer.step()
    pi_optimizer.step()

    Loss = loss1 + loss2
    print(f"Episode : {episode} || Rewards : {score}")
    episode += 1
env.close()
import gym
import sys
import copy
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Categorical
from IPython.display import clear_output
from time import sleep

from collections import deque

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
lmbda = 0.95
eps_clip = 0.1
K_epoch = 3
T_horizon = 200


class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()

        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_v = nn.Linear(256, 1)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v


def make_batch():
    s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
    for transition in data:
        s, a, r, s_prime, prob_a, done = transition

        s_lst.append(s)
        a_lst.append([a])
        r_lst.append([r])
        s_prime_lst.append(s_prime)
        prob_a_lst.append([prob_a])
        done_mask = 0 if done else 1
        done_lst.append([done_mask])

    s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                          torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst,
                                                                                               dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)

    return s, a, r, s_prime, done_mask, prob_a


def train(model, optimizer):
    s, a, r, s_prime, done_mask, prob_a = make_batch()

    for i in range(K_epoch):
        td_target = r + gamma * model.v(s_prime) * done_mask
        delta = td_target - model.v(s)
        delta = delta.detach().numpy()

        advantage_lst = []
        advantage = 0.0
        for delta_t in delta[::-1]:
            # Implemented Generalized Advantage Estimation (GAE) with lambda=0.95
            advantage = gamma * lmbda * advantage + delta_t[0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        advantage = torch.tensor(advantage_lst, dtype=torch.float)

        pi = model.pi(s, softmax_dim=1)
        pi_a = pi.gather(1, a)
        ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
        loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(model.v(s), td_target.detach())

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()


# env = gym.make('LunarLander-v2')
env = gym.make('CartPole-v0')
model = PPO()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

MAX_EPISODES = 3000
score = 0.0
print_interval = 20
reward_history = []
reward_history_100 = deque(maxlen=100)

for episode in range(MAX_EPISODES):
    s = env.reset()
    done = False
    data = []
    while not done:
        for t in range(T_horizon):

            prob = model.pi(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample().item()
            s_prime, r, done, info = env.step(a)
            data.append((s, a, r / 100.0, s_prime, prob[a].item(), done))
            s = s_prime

            score = score + r
            if done:
                break

        train(model, optimizer)
        data = []

    reward_history.append(score)
    reward_history_100.append(score)
    avg = sum(reward_history_100) / len(reward_history_100)
    episode = episode + 1
    if episode % 100 == 0:
        print('episode: {}, reward: {:.1f}, avg: {:.1f}'.format(episode, score, avg))

    score = 0.0

env.close()

import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import sleep
from collections import deque
import matplotlib.pyplot as plt
import gym
import sys
import random

batch_size = 16       # Mini Batch Size for Sampling from Replay Memory
buffer_limit = 50000  # Replay Memory Size

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition) # transition : (state, action, reward, next_state, done)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n) # buffer에서 n개 뽑기
        states, actions, rewards, next_states, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            state, action, reward, next_state, done = transition
            states.append(state) # s = [COS SIN 각속도]
            actions.append([action])
            rewards.append([reward])
            next_states.append(next_state)
            done_mask = 0.0 if done else 1.0
            done_mask_lst.append([done_mask])

        s_lst = np.array(states)
        #a_lst = np.array(actions)
        r_lst = np.array(rewards)
        s_prime_lst = np.array(next_states)
        done_mask_lst = np.array(done_mask_lst)

        return torch.tensor(s_lst, device = device, dtype=torch.float), torch.tensor(actions, device = device,dtype=torch.float), \
               torch.tensor(r_lst, device = device,dtype=torch.float), torch.tensor(s_prime_lst,device = device, dtype=torch.float), \
               torch.tensor(done_mask_lst, device = device, dtype=torch.float)

    def size(self):
        return len(self.buffer)

class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

# Model 1
class MuNet1(nn.Module):  # Mu = Torque -> Action
    def __init__(self):
        super(MuNet1, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, 1)

    def forward(self, x): # Input : state (COS, SIN, 각속도)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x)) * 2
        return mu # Return Deterministic Policy

class QNet1(nn.Module):
    def __init__(self):
        super(QNet1, self).__init__()
        self.fc_s = nn.Linear(3, 64)   # State = (COS, SIN, 각속도)
        self.fc_a = nn.Linear(1, 64)   # Action = Torque
        self.fc_q = nn.Linear(128, 32) # State , Action 이어붙이기
        self.fc_out = nn.Linear(32, 1) # Output : Q value

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x)) # 64
        h2 = F.relu(self.fc_a(a)) # 64
        cat = torch.cat([h1, h2], dim = 1)  # 128

        q = F.relu(self.fc_q(cat)) # 32
        q = self.fc_out(q)  # 1
        return q

# Model 2
class MuNet2(nn.Module):  # Mu = Torque -> Action
    def __init__(self):
        super(MuNet2, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_mu = nn.Linear(32, 1)

    def forward(self, x): # Input : state (COS, SIN, 각속도)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x)) * 2
        return mu # Return Deterministic Policy

class QNet2(nn.Module):
    def __init__(self):
        super(QNet2, self).__init__()
        self.fc_s = nn.Linear(3, 32)   # State = (COS, SIN, 각속도)
        self.fc_a = nn.Linear(1, 32)   # Action = Torque
        self.fc_q = nn.Linear(64, 32) # State , Action 이어붙이기
        self.fc_out = nn.Linear(32, 1) # Output : Q value

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x)) # 64
        h2 = F.relu(self.fc_a(a)) # 64
        cat = torch.cat([h1, h2], dim = 1)  # 128

        q = F.relu(self.fc_q(cat)) # 32
        q = self.fc_out(q)  # 1
        return q


# Model 3
class MuNet3(nn.Module):  # Mu = Torque -> Action
    def __init__(self):
        super(MuNet3, self).__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc_mu = nn.Linear(32, 1)

    def forward(self, x): # Input : state (COS, SIN, 각속도)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x)) * 2
        return mu # Return Deterministic Policy

class QNet3(nn.Module):
    def __init__(self):
        super(QNet3, self).__init__()
        self.fc_s = nn.Linear(3, 16)   # State = (COS, SIN, 각속도)
        self.fc_a = nn.Linear(1, 16)   # Action = Torque
        self.fc_q = nn.Linear(32, 32) # State , Action 이어붙이기
        self.fc_out = nn.Linear(32, 1) # Output : Q value

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x)) # 64
        h2 = F.relu(self.fc_a(a)) # 64
        cat = torch.cat([h1, h2], dim = 1)  # 128

        q = F.relu(self.fc_q(cat)) # 32
        q = self.fc_out(q)  # 1
        return q
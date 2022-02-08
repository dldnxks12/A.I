# Gradient 충돌 문제 해결해야함

import gym
import sys
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import sleep
from collections import deque

#GPU Setting

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameters
lr_mu = 0.0005       # Learning Rate for Torque (Action)
lr_q = 0.001         # Learning Rate for Q
gamma = 0.99         # discount factor
batch_size = 32      # Mini Batch Size for Sampling from Replay Memory
buffer_limit = 50000 # Replay Memory Size
tau = 0.005          # for target network soft update

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition) # transition : (state, action, reward, next_state, done)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n) # buffer에서 n개 뽑기
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s) # s = [COS SIN 각속도]
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.float), \
               torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst, dtype=torch.float)

    def size(self):
        return len(self.buffer)


class MuNet(nn.Module):  # Mu = Torque -> Action
    def __init__(self):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, 1)

    def forward(self, x): # Input : state (COS, SIN, 각속도)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x)) * 2  # Multipled by 2 because the action space of the Pendulum-v0 is [-2,2]
        return mu

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(3, 64)   # State = (COS, SIN, 각속도)
        self.fc_a = nn.Linear(1, 64)   # Action = Torque
        self.fc_q = nn.Linear(128, 32) # State , Action 이어붙이기
        self.fc_out = nn.Linear(32, 1) # Output : Q value

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x)) # 64
        h2 = F.relu(self.fc_a(a)) # 64
        # cat = torch.cat([h1, h2], dim = -1) # 128
        # cat = torch.cat([h1, h2], dim = 1) # 128
        cat = torch.cat([h1, h2], dim = -1)  # 128

        q = F.relu(self.fc_q(cat)) # 32
        q = self.fc_out(q)  # 1
        return q


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


def train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer):
    states, actions, rewards, next_states, dones = memory.sample(batch_size)

    loss1 = 0
    loss2 = 0
    for (state, action, reward, next_state, done) in zip(states, actions, rewards, next_states, dones):
        if done == 0: # done == True일 때,
            y = reward
        else:
            y = reward + gamma*q_target(next_state, mu_target(next_state))

        loss1 += (y - q(state, action))**2

    loss1 = loss1.mean()

    q_optimizer.zero_grad()
    loss1.backward()
    q_optimizer.step()

    for (state, action, reward, next_state, done) in zip(states, actions, rewards, next_states, dones):
        if done == 0: # done == True일 때,
            y = reward
        else:
            y = reward + gamma*q_target(next_state, mu_target(next_state))

        loss2 += q(state, mu(state))

    loss2 = -loss2.mean()  # Gradient Ascent -> Gradient Descent

    mu_optimizer.zero_grad()
    loss2.backward()
    mu_optimizer.step()


def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

env = gym.make('Pendulum-v0')
memory = ReplayBuffer()

# 2개의 동일한 네트워크 생성 ...
q =  QNet()
q_target = QNet()
mu = MuNet()
mu_target = MuNet()

q_target.load_state_dict(q.state_dict())   # 파라미터 동기화
mu_target.load_state_dict(mu.state_dict()) # 파라미터 동기화

score = 0.0
print_interval = 20
reward_history = []
reward_history_100 = deque(maxlen=100)

mu_optimizer = optim.Adam(mu.parameters(), lr=lr_mu)
q_optimizer = optim.Adam(q.parameters(), lr=lr_q)
ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))
MAX_EPISODES = 10000
for episode in range(MAX_EPISODES):
    s = env.reset()

    done = False

    while not done: # Stacking Experiences

        if episode % 10 == 0:
            env.render()

        a = mu(torch.from_numpy(s).float()) # Return action (-2 ~ 2 사이의 torque  ... )
        a = a.item() + ou_noise()[0] # Action에 Noise를 추가해서 Exploration 기능 추가 ...
        s_prime, r, done, info = env.step([a])

        memory.put((s, a, r / 100.0, s_prime, done))
        score = score + r
        s = s_prime

    if memory.size() > 2000:
        for i in range(10):
            train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer)
            soft_update(mu, mu_target)
            soft_update(q, q_target)

    reward_history.append(score)
    reward_history_100.append(score)
    avg = sum(reward_history_100) / len(reward_history_100)
    episode = episode + 1
    if episode % 100 == 0:
        print('episode: {}, reward: {:.1f}, avg: {:.1f}'.format(episode, score, avg))
    score = 0.0

env.close()


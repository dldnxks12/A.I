# TD3 = DDPG + Remove Maximization Bias
# 학습 OK

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
#device =  'cpu'

print("")
print(f"On {device}")
print("")

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

        s_lst = np.array(s_lst)
        a_lst = np.array(a_lst)
        r_lst = np.array(r_lst)
        s_prime_lst = np.array(s_prime_lst)
        done_mask_lst = np.array(done_mask_lst)

        return torch.tensor(s_lst, device = device, dtype=torch.float), torch.tensor(a_lst, device = device, dtype=torch.float), \
               torch.tensor(r_lst, device = device,dtype=torch.float), torch.tensor(s_prime_lst, device = device, dtype=torch.float), \
               torch.tensor(done_mask_lst, device = device, dtype=torch.float)

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
        cat = torch.cat([h1, h2], dim = 1)  # 128

        q = F.relu(self.fc_q(cat)) # 32
        q = self.fc_out(q)  # 1
        return q


# Add Noise to deterministic action for improving exploration property
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


# Update Network Priodically ...
def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

def train(episode, mu, mu_target, q1, q2, q1_target, q2_target, memory, q1_optimizer, q2_optimizer, mu_optimizer):
    states, actions, rewards, next_states, dones = memory.sample(batch_size)

    Q_loss, mu_loss = 0, 0

    noise_bar = torch.clamp(torch.tensor(ou_noise()[0]), -1, 1)

    # print("Point 1")
    # print(noise_bar) # 0.1425

    action_bar = mu_target(next_states) + noise_bar

    # Shape Check 필요
    q1_value = q1_target(next_states, action_bar).mean()
    q2_value = q2_target(next_states, action_bar).mean()

    selected_Q = torch.min(q1_value, q2_value)
    selected_Q_index = torch.argmin(torch.tensor([q1_value, q2_value]), axis = 0)

    # Q1
    if selected_Q_index == 0:
        # q1 Network Update
        y = rewards + (gamma * q1_target(next_states, action_bar)) * dones
        Q_loss = torch.nn.functional.smooth_l1_loss(q1(states, actions), y.detach())
        q2_optimizer.zero_grad()
        Q_loss.backward()
        q2_optimizer.step()

    # Q2
    else:
        # q2 Network Update
        y = rewards + (gamma * q2_target(next_states, action_bar)) * dones
        Q_loss = torch.nn.functional.smooth_l1_loss(q2(states, actions), y.detach())
        q1_optimizer.zero_grad()
        Q_loss.backward()
        q1_optimizer.step()


    # Update Policy Network periodically ...
    if episode % 5 == 0:
        if selected_Q_index == 0:
            mu_loss = -q1(states, mu(states)).mean()
        else:
            mu_loss = -q2(states, mu(states)).mean()
        mu_optimizer.zero_grad()
        mu_loss.backward()
        mu_optimizer.step()


# Hyperparameters
lr_mu = 0.0005       # Learning Rate for Torque (Action)
lr_q = 0.001         # Learning Rate for Q
gamma = 0.99         # discount factor
batch_size = 16      # Mini Batch Size for Sampling from Replay Memory
buffer_limit = 50000 # Replay Memory Size
tau = 0.005          # for target network soft update

# Import Gym Environment
env = gym.make('Pendulum-v1')

# Replay Buffer
memory = ReplayBuffer()

# Networks
q1 =  QNet().to(device) # Twin Network for avoiding maximization bias
q2 =  QNet().to(device) # Twin Network for avoiding maximization bias

q1_target = QNet().to(device)
q2_target = QNet().to(device)

mu = MuNet().to(device)
mu_target = MuNet().to(device)

# Parameter Synchronize
q1_target.load_state_dict(q1.state_dict())
q2_target.load_state_dict(q2.state_dict())
mu_target.load_state_dict(mu.state_dict())


# Optimizer
mu_optimizer = optim.Adam(mu.parameters(), lr=lr_mu)
q1_optimizer = optim.Adam(q1.parameters(), lr=lr_q)
q2_optimizer = optim.Adam(q2.parameters(), lr=lr_q)

# Noise
ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))

score = 0.0
print_interval = 20
reward_history = []
reward_history_100 = deque(maxlen=100)
MAX_EPISODES = 10000

for episode in range(MAX_EPISODES):
    state = env.reset()
    done = False

    while not done: # Stacking Experiences

        #if episode % 100 == 0:
        #    env.render()

        action = mu(torch.from_numpy(state).float().to(device)) # Return action (-2 ~ 2 사이의 torque  ... )
        action = action.item() + ou_noise()[0] # Action에 Noise를 추가해서 Exploration 기능 추가 ...
        next_state, reward, done, info = env.step([action])
        memory.put((state, action, reward / 100.0, next_state, done))
        score = score + reward
        state = next_state

        if memory.size() > 2000:
            for i in range(10):
                train(episode, mu, mu_target, q1, q2, q1_target, q2_target, memory, q1_optimizer,q2_optimizer, mu_optimizer)
                soft_update(q1, q1_target)
                soft_update(q2, q2_target)
                soft_update(mu, mu_target)

    reward_history.append(score)
    reward_history_100.append(score)
    avg = sum(reward_history_100) / len(reward_history_100)
    if episode % 10 == 0:
        print('episode: {}, reward: {:.1f}, avg: {:.1f}'.format(episode, score, avg))
    score = 0.0
    episode = episode + 1

env.close()


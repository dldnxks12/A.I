# MU NET 동일하게 4개 복제
# Q NET 1개

###########################################################################
# To Avoid Library Collision
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
###########################################################################

import gym
import sys
import random
import collections
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from time import sleep
from collections import deque
import matplotlib.pyplot as plt

###########################################################################
# Env Setting
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("")
print(f"On {device}")
print("")

# Hyperparameters
lr_mu = 0.05         # Learning Rate for Torque (Action)
lr_q  = 0.05          # Learning Rate for Q
gamma = 0.99         # discount factor
batch_size = 64      # Mini Batch Size for Sampling from Replay Memory
buffer_limit = 50000 # Replay Memory Size
tau = 0.05          # for target network soft update


###########################################################################
# Model and ReplayBuffer
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
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_mask_lst.append([done_mask])

        s_lst = np.array(s_lst)
        a_lst = np.array(a_lst)
        r_lst = np.array(r_lst)
        s_prime_lst = np.array(s_prime_lst)
        done_mask_lst = np.array(done_mask_lst)

        return torch.tensor(s_lst, device = device , dtype=torch.float), torch.tensor(a_lst, device = device , dtype=torch.float), \
               torch.tensor(r_lst, device = device , dtype=torch.float), torch.tensor(s_prime_lst, device = device , dtype=torch.float), \
               torch.tensor(done_mask_lst, device = device , dtype=torch.float)

    def size(self):
        return len(self.buffer)

# Deterministic Policy ...
'''
    ### Action Space
    Actions are motor speed values in the [-1, 1] range for each of the
    4 joints at both hips and knees.
'''
class MuNet1(nn.Module):  # Output : Deterministic Action !
    def __init__(self):
        super(MuNet1, self).__init__()
        self.fc1 = nn.Linear(24, 64)  # Input  : 24 continuous states
        self.fc2 = nn.Linear(64, 32)
        self.fc_mu = nn.Linear(32, 4)  # Output : 4 continuous actions

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x))
        return mu


class MuNet2(nn.Module):  # Output : Deterministic Action !
    def __init__(self):
        super(MuNet2, self).__init__()
        self.fc1 = nn.Linear(24, 32)  # Input  : 24 continuous states
        self.fc2 = nn.Linear(32, 16)
        self.fc_mu = nn.Linear(16, 4)  # Output : 4 continuous actions

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x))
        return mu


class MuNet3(nn.Module):  # Output : Deterministic Action !
    def __init__(self):
        super(MuNet3, self).__init__()
        self.fc1 = nn.Linear(24, 128)  # Input  : 24 continuous states
        self.fc2 = nn.Linear(128, 32)
        self.fc_mu = nn.Linear(32, 4)  # Output : 4 continuous actions

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x))
        return mu


class MuNet4(nn.Module):  # Output : Deterministic Action !
    def __init__(self):
        super(MuNet4, self).__init__()
        self.fc1 = nn.Linear(24, 64)  # Input  : 24 continuous states
        self.fc2 = nn.Linear(64, 64)
        self.fc_mu = nn.Linear(64, 4)  # Output : 4 continuous actions

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x))
        return mu

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc_s   = nn.Linear(24, 64)    # State  24 개
        self.fc_a   = nn.Linear(4, 64)     # Action 4  개
        self.fc_q   = nn.Linear(128, 64)  # State , Action 이어붙이기
        self.fc_out = nn.Linear(64, 32)   # Output : Q value
        self.fc_out2 = nn.Linear(32, 1)    # Output : Q value

    def forward(self, x, a):

        h1 = F.relu(self.fc_s(x)) # 128
        h2 = F.relu(self.fc_a(a)) # 128
        cat = torch.cat([h1, h2], dim = 1)  # 256
        q = F.relu(self.fc_q(cat))   # 128
        q = self.fc_out(q)   # 64
        q = self.fc_out2(q)  # 1 - Q Value
        return q


# Action에 Noise를 추가하여 Exploration ---- 이후 앙상블로 바꿔볼 것
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

###########################################################################
# Train ...
def train(mu1,mu2,mu3,mu4,mu_target1,mu_target2,mu_target3,mu_target4, q, q_target, memory, q_optimizer, mu_optimizer1,mu_optimizer2,mu_optimizer3,mu_optimizer4):
    states, actions, rewards, next_states, dones = memory.sample(batch_size)
    Critic = 0.0
    Actor1 = 0.0
    Actor2 = 0.0
    Actor3 = 0.0
    Actor4 = 0.0

    y1 = rewards + (gamma * q_target(next_states, mu_target1(next_states)) * dones)
    y2 = rewards + (gamma * q_target(next_states, mu_target2(next_states)) * dones)
    y3 = rewards + (gamma * q_target(next_states, mu_target3(next_states)) * dones)
    y4 = rewards + (gamma * q_target(next_states, mu_target4(next_states)) * dones)

    Critic1 = torch.nn.functional.smooth_l1_loss(q(states, actions), y1.detach())
    Critic2 = torch.nn.functional.smooth_l1_loss(q(states, actions), y2.detach())
    Critic3 = torch.nn.functional.smooth_l1_loss(q(states, actions), y3.detach())
    Critic4 = torch.nn.functional.smooth_l1_loss(q(states, actions), y4.detach())

    Critic = (Critic1 + Critic2 + Critic3 + Critic4).mean()
    q_optimizer.zero_grad()
    Critic.backward()
    q_optimizer.step()

    Actor1 = -q(states, mu1(states)).mean()
    Actor2 = -q(states, mu2(states)).mean()
    Actor3 = -q(states, mu3(states)).mean()
    Actor4 = -q(states, mu4(states)).mean()

    mu_optimizer1.zero_grad()
    Actor1.backward()
    mu_optimizer1.step()

    mu_optimizer2.zero_grad()
    Actor2.backward()
    mu_optimizer2.step()

    mu_optimizer3.zero_grad()
    Actor3.backward()
    mu_optimizer3.step()

    mu_optimizer4.zero_grad()
    Actor4.backward()
    mu_optimizer4.step()

# Soft Update
def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


# state  : continuous 24 state
# action : continuous 4 action
env = gym.make('BipedalWalker-v3')
env.reset()

memory = ReplayBuffer()

# 1개의 Q Net
q =  QNet().to(device)
q_target = QNet().to(device)

q_target.load_state_dict(q.state_dict())   # 파라미터 동기화
q_optimizer  = optim.Adam(q.parameters(), lr=lr_q)

# 4 개의 동일한 Mu Network
mu1        = MuNet1().to(device)
mu2        = MuNet2().to(device)
mu3        = MuNet3().to(device)
mu4        = MuNet4().to(device)
mu_target1 = MuNet1().to(device)
mu_target2 = MuNet2().to(device)
mu_target3 = MuNet3().to(device)
mu_target4 = MuNet4().to(device)

mu_target1.load_state_dict(mu1.state_dict()) # 파라미터 동기화
mu_target2.load_state_dict(mu2.state_dict()) # 파라미터 동기화
mu_target3.load_state_dict(mu3.state_dict()) # 파라미터 동기화
mu_target4.load_state_dict(mu4.state_dict()) # 파라미터 동기화

mu_optimizer1 = optim.Adam(mu1.parameters(), lr=lr_mu)
mu_optimizer2 = optim.Adam(mu2.parameters(), lr=lr_mu)
mu_optimizer3 = optim.Adam(mu3.parameters(), lr=lr_mu)
mu_optimizer4 = optim.Adam(mu4.parameters(), lr=lr_mu)

ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(4))
MAX_EPISODES = 1000

reward_history_20 = []

episode = 0
while episode < MAX_EPISODES:
    state = env.reset()
    done = False
    score = 0.0
    while not done:

        action1 = mu1(torch.from_numpy(state).to(device))
        action2 = mu2(torch.from_numpy(state).to(device))
        action3 = mu3(torch.from_numpy(state).to(device))
        action4 = mu4(torch.from_numpy(state).to(device))

        # 모두 다른 Noise 생성

        '''
        noise1 = torch.tensor(ou_noise(), device=device)
        noise2 = torch.tensor(ou_noise(), device=device)
        noise3 = torch.tensor(ou_noise(), device=device)
        noise4 = torch.tensor(ou_noise(), device=device)
        '''

        q_value_for_softmax1 = q_target(torch.from_numpy(state).unsqueeze(0).to(device), action1.unsqueeze(0))
        q_value_for_softmax2 = q_target(torch.from_numpy(state).unsqueeze(0).to(device), action2.unsqueeze(0))
        q_value_for_softmax3 = q_target(torch.from_numpy(state).unsqueeze(0).to(device), action3.unsqueeze(0))
        q_value_for_softmax4 = q_target(torch.from_numpy(state).unsqueeze(0).to(device), action4.unsqueeze(0))

        actions = torch.stack([q_value_for_softmax1,q_value_for_softmax2,q_value_for_softmax3,q_value_for_softmax4])
        action_softmax = torch.nn.functional.softmax(actions, dim = 0).squeeze(1).squeeze(1).cpu().detach().numpy()

        action_list = [action1, action2, action3, action4]
        action_index = [0, 1, 2, 3]

        choice_action = np.random.choice(action_index, 1, p = action_softmax)
        action = action_list[choice_action[0]].cpu().detach().numpy()

        # Add Exploration property
        #action = (action + noise).cpu().detach().numpy()

        next_state, reward, done, _ = env.step(action)

        reward = reward * 10

        # Type Check
        # print(type(state), type(action), type(next_state), type(reward), type(done))
        memory.put((state, action, reward, next_state, done))
        score += reward
        state = next_state

    if memory.size() > 500:
        train(mu1,mu2,mu3,mu4,mu_target1,mu_target2,mu_target3,mu_target4, q, q_target, memory, q_optimizer, mu_optimizer1,mu_optimizer2,mu_optimizer3,mu_optimizer4)
        soft_update(mu1, mu_target1)
        soft_update(mu2, mu_target2)
        soft_update(mu3, mu_target3)
        soft_update(mu4, mu_target4)
        soft_update(q, q_target)

    reward_history_20.append(score)
    avg = sum(reward_history_20) / len(reward_history_20)
    if episode % 10 == 0:
        print('episode: {}, reward: {:.1f}, avg: {:.1f}'.format(episode, score, avg))
    episode += 1

env.close()

#######################################################################
# Record Hyperparamters & Result Graph

with open('exploration-v3.txt', 'w', encoding = 'UTF-8') as f:
    f.write("# ----------------------- # " + '\n')
    f.write("DDPG_Parameter 2022-2-14" + '\n')
    f.write('\n')
    f.write('\n')
    f.write("# - Category 1 - #" + '\n')
    f.write('\n')
    f.write("Reward        : reward x 10" + '\n')
    f.write("lr_mu         : " + str(lr_mu) + '\n')
    f.write("lr_q          : " + str(lr_q) + '\n')
    f.write("tau           : " + str(tau) + '\n')
    f.write('\n')
    f.write("# - Category 2 - #" + '\n')
    f.write('\n')
    f.write("batch_size    : " + str(batch_size)   + '\n')
    f.write("buffer_limit  : " + str(buffer_limit) + '\n')
    f.write("memory.size() : 500" + '\n')
    f.write("# ----------------------- # " + '\n')

length = np.arange(len(reward_history_20))*20
plt.figure()
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("exploration-v3")
plt.plot(length, reward_history_20)
plt.savefig('exploration-v3.png')
# Batch Normalization 사용
# Bolzmann
# Bagging x 5

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
lr_mu = 0.001            # Learning Rate for Torque (Action)
lr_q = 0.001             # Learning Rate for Q
gamma = 0.99             # discount factor
batch_size = 100         # Mini Batch Size for Sampling from Replay Memory
buffer_limit = 2000000   # Replay Memory Size
tau = 0.01               # for target network soft update


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

class MuNet1(nn.Module):  # Output : Deterministic Action
    def __init__(self):
        super(MuNet1, self).__init__()
        self.fc1 = nn.Linear(24, 256)  # Input  : 24 continuous states
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc_mu = nn.Linear(32, 4)  # Output : 4 continuous actions

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        mu = torch.tanh(self.fc_mu(x))
        return mu

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc_s   = nn.Linear(24, 64)    # State  24 개
        self.fc_a   = nn.Linear(4, 64)     # Action 4  개
        self.fc_q   = nn.Linear(128, 64)  # State , Action 이어붙이기
        self.bn1    = nn.BatchNorm1d(64)
        self.fc_out = nn.Linear(64, 32)   # Output : Q value
        self.bn2 = nn.BatchNorm1d(32)
        self.fc_out2 = nn.Linear(32, 1)    # Output : Q value

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))              # 128
        h2 = F.relu(self.fc_a(a))              # 128
        cat = torch.cat([h1, h2], dim = 1)     # 256
        q = F.relu(self.bn1(self.fc_q(cat)))   # 128
        q = F.relu(self.bn2(self.fc_out(q)))   # 64
        q = self.fc_out2(q)                    # 1 - Q Value
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
def train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer):
    states, actions, rewards, next_states, dones = memory.sample(batch_size)
    Critic = 0.0
    Actor = 0.0

    y = rewards + gamma * q_target(next_states, mu_target(next_states)) * dones
    Critic = torch.nn.functional.smooth_l1_loss(q(states, actions), y.detach())

    q_optimizer.zero_grad()
    Critic.backward()
    q_optimizer.step()

    Actor = -q(states, mu(states)).mean()

    mu_optimizer.zero_grad()
    Actor.backward()
    mu_optimizer.step()

# Soft Update
def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


env = gym.make('BipedalWalker-v3')
memory = ReplayBuffer()

# 1개의 Q Net
q =  QNet().to(device)
q_target = QNet().to(device)

q_target.load_state_dict(q.state_dict())   # 파라미터 동기화
q_optimizer  = optim.Adam(q.parameters(), lr=lr_q)

# 4 개의 동일한 Mu Network
mu1        = MuNet1().to(device)
#mu2        = MuNet1().to(device)
#mu3        = MuNet1().to(device)
mu_target1 = MuNet1().to(device)
#mu_target2 = MuNet1().to(device)
#mu_target3 = MuNet1().to(device)

mu_target1.load_state_dict(mu1.state_dict()) # 파라미터 동기화
# mu_target2.load_state_dict(mu2.state_dict()) # 파라미터 동기화
# mu_target3.load_state_dict(mu3.state_dict()) # 파라미터 동기화

mu_optimizer1 = optim.Adam(mu1.parameters(), lr=lr_mu)
# mu_optimizer2 = optim.Adam(mu2.parameters(), lr=lr_mu)
# mu_optimizer3 = optim.Adam(mu3.parameters(), lr=lr_mu)

ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(4))
MAX_EPISODES = 1000
DECAY_RATE = 1
avg_history = []
reward_history_20 = []
episode = 0
while episode < MAX_EPISODES:
    state = env.reset()
    done = False
    score = 0.0

    while not done:
        stack = [state] * 2
        stack = np.array(stack)
        stack = torch.from_numpy(stack).float().to(device).squeeze(0)

        action1 = mu1(stack)
        #action2 = mu2(stack)
        #action3 = mu3(stack)

        q_value_for_softmax1 = q_target(stack.to(device), action1)[0].unsqueeze(0)
        #q_value_for_softmax2 = q_target(stack.to(device), action2)[0].unsqueeze(0)
        #q_value_for_softmax3 = q_target(stack.to(device), action3)[0].unsqueeze(0)

        #actions = torch.stack([q_value_for_softmax1,q_value_for_softmax2,q_value_for_softmax3])
        #action_softmax = torch.nn.functional.softmax(actions, dim = 0).squeeze(1).squeeze(1).cpu().detach().numpy()

        #action_list = [action1[0], action2[0], action3[0]]
        #action_index = [0, 1, 2]

        #choice_action = np.random.choice(action_index, 1, p = action_softmax)
        #action = action_list[choice_action[0]].cpu().detach().numpy()

        # 추가
        action = action1[0].cpu().detach().numpy()

        next_state, reward, done, _ = env.step(action)
        memory.put((state, action, reward / 10.0, next_state, done))
        score += reward
        state = next_state

    if memory.size() > 2000:
        for _ in range(10):
            # Bagging 을 통해 Variance 줄이기
            train(mu1, mu_target1, q, q_target, memory, q_optimizer, mu_optimizer1)
            #train(mu2, mu_target2, q, q_target, memory, q_optimizer, mu_optimizer2)
            #train(mu3, mu_target3, q, q_target, memory, q_optimizer, mu_optimizer3)

        soft_update(mu1, mu_target1)
        #soft_update(mu2, mu_target2)
        #soft_update(mu3, mu_target3)
        soft_update(q, q_target)

    # Moving Average Count
    reward_history_20.insert(0, score)
    if len(reward_history_20) == 10:
        reward_history_20.pop()
    avg = sum(reward_history_20) / len(reward_history_20)
    avg_history.append(avg)
    if episode % 10 == 0:
        print('episode: {}, reward: {:.1f}, 10 avg: {:.1f}'.format(episode, score, avg))
    episode += 1

env.close()

#######################################################################
# Record Hyperparamters & Result Graph

with open('test 1 check.txt', 'w', encoding = 'UTF-8') as f:
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

length = np.arange(len(avg_history))
plt.figure()
plt.xlabel("Episode")
plt.ylabel("10 episode MVA")
plt.plot(length, avg_history)
plt.savefig('test 1 check.png')
# Batch Normalization 사용
# Bolzmann
# Bagging x 5

###########################################################################
# To Avoid Library Collision
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
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
lr_mu = 0.0005  # Learning Rate for Torque (Action)
lr_q = 0.005  # Learning Rate for Q
gamma = 0.99  # discount factor
#batch_size = 100  # Mini Batch Size for Sampling from Replay Memory
buffer_limit = 2000000  # Replay Memory Size
tau = 0.01  # for target network soft update



###########################################################################
# Model and ReplayBuffer
class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)  # transition : (state, action, reward, next_state, done)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)  # buffer에서 n개 뽑기
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)  # s = [COS SIN 각속도]
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


        return torch.tensor(s_lst, device=device, dtype=torch.float),         \
               torch.tensor(a_lst, device=device, dtype=torch.float),         \
               torch.tensor(r_lst, device=device, dtype=torch.float),         \
               torch.tensor(s_prime_lst, device=device,dtype=torch.float),    \
               torch.tensor(done_mask_lst, device=device, dtype=torch.float)


    def size(self):
        return len(self.buffer)


class MuNet1(nn.Module):  # Output : Deterministic Action !
    def __init__(self):
        super(MuNet1, self).__init__()
        self.fc1 = nn.Linear(24, 64)  # Input  : 24 continuous states
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc_mu = nn.Linear(32, 4)  # Output : 4 continuous actions

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        mu = torch.tanh(self.fc_mu(x))
        return mu


class MuNet2(nn.Module):  # Output : Deterministic Action !
    def __init__(self):
        super(MuNet2, self).__init__()
        self.fc1 = nn.Linear(24, 32)  # Input  : 24 continuous states
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc_mu = nn.Linear(16, 4)  # Output : 4 continuous actions

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        mu = torch.tanh(self.fc_mu(x))
        return mu


class MuNet3(nn.Module):  # Output : Deterministic Action !
    def __init__(self):
        super(MuNet3, self).__init__()
        self.fc1 = nn.Linear(24, 128)  # Input  : 24 continuous states
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc_mu = nn.Linear(32, 4)  # Output : 4 continuous actions

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        mu = torch.tanh(self.fc_mu(x))
        return mu


class MuNet4(nn.Module):  # Output : Deterministic Action !
    def __init__(self):
        super(MuNet4, self).__init__()
        self.fc1 = nn.Linear(24, 64)  # Input  : 24 continuous states
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, 8)
        self.bn3 = nn.BatchNorm1d(8)
        self.fc_mu = nn.Linear(8, 4)  # Output : 4 continuous actions

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        mu = torch.tanh(self.fc_mu(x))
        return mu


class QNet1(nn.Module):
    def __init__(self):
        super(QNet1, self).__init__()
        self.fc_s = nn.Linear(24, 32)  # State  24 개
        self.fc_a = nn.Linear(4, 32)  # Action 4  개
        self.fc_q = nn.Linear(64, 16)  # State , Action 이어붙이기
        self.bn1 = nn.BatchNorm1d(16)
        self.fc_out = nn.Linear(16, 8)  # Output : Q value
        self.bn2 = nn.BatchNorm1d(8)
        self.fc_out2 = nn.Linear(8, 1)  # Output : Q value

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))  # 128
        h2 = F.relu(self.fc_a(a))  # 128
        cat = torch.cat([h1, h2], dim=1)  # 256
        q = F.relu(self.bn1(self.fc_q(cat)))  # 128
        q = F.relu(self.bn2(self.fc_out(q)))  # 64
        q = self.fc_out2(q)  # 1 - Q Value
        return q

class QNet2(nn.Module):
    def __init__(self):
        super(QNet2, self).__init__()
        self.fc_s = nn.Linear(24, 32)  # State  24 개
        self.fc_a = nn.Linear(4, 32)  # Action 4  개
        self.fc_q = nn.Linear(64, 64)  # State , Action 이어붙이기
        self.bn1 = nn.BatchNorm1d(64)
        self.fc_out = nn.Linear(64, 32)  # Output : Q value
        self.bn2 = nn.BatchNorm1d(32)
        self.fc_out2 = nn.Linear(32, 1)  # Output : Q value

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))  # 128
        h2 = F.relu(self.fc_a(a))  # 128
        cat = torch.cat([h1, h2], dim=1)  # 256
        q = F.relu(self.bn1(self.fc_q(cat)))  # 128
        q = F.relu(self.bn2(self.fc_out(q)))  # 64
        q = self.fc_out2(q)  # 1 - Q Value
        return q

class QNet3(nn.Module):
    def __init__(self):
        super(QNet3, self).__init__()
        self.fc_s = nn.Linear(24, 32)  # State  24 개
        self.fc_a = nn.Linear(4, 32)  # Action 4  개
        self.fc_q = nn.Linear(64, 32)  # State , Action 이어붙이기
        self.bn1 = nn.BatchNorm1d(32)
        self.fc_out = nn.Linear(32, 8)  # Output : Q value
        self.bn2 = nn.BatchNorm1d(8)
        self.fc_out2 = nn.Linear(8, 1)  # Output : Q value

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))  # 128
        h2 = F.relu(self.fc_a(a))  # 128
        cat = torch.cat([h1, h2], dim=1)  # 256
        q = F.relu(self.bn1(self.fc_q(cat)))  # 128
        q = F.relu(self.bn2(self.fc_out(q)))  # 64
        q = self.fc_out2(q)  # 1 - Q Value
        return q

class QNet4(nn.Module):
    def __init__(self):
        super(QNet4, self).__init__()
        self.fc_s = nn.Linear(24, 32)  # State  24 개
        self.fc_a = nn.Linear(4, 32)  # Action 4  개
        self.fc_q = nn.Linear(64, 128)  # State , Action 이어붙이기
        self.bn1 = nn.BatchNorm1d(128)
        self.fc_out = nn.Linear(128, 16)  # Output : Q value
        self.bn2 = nn.BatchNorm1d(16)
        self.fc_out2 = nn.Linear(16, 1)  # Output : Q value

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))  # 128
        h2 = F.relu(self.fc_a(a))  # 128
        cat = torch.cat([h1, h2], dim=1)  # 256
        q = F.relu(self.bn1(self.fc_q(cat)))  # 128
        q = F.relu(self.bn2(self.fc_out(q)))  # 64
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
def train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer, batch_size):
    states, actions, rewards, next_states, dones = memory.sample(batch_size)

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
q1 = QNet1().to(device)
q2 = QNet2().to(device)
q3 = QNet3().to(device)
q4 = QNet4().to(device)
q_target1 = QNet1().to(device)
q_target2 = QNet2().to(device)
q_target3 = QNet3().to(device)
q_target4 = QNet4().to(device)

q_target1.load_state_dict(q1.state_dict())  # 파라미터 동기화
q_target2.load_state_dict(q2.state_dict())  # 파라미터 동기화
q_target3.load_state_dict(q3.state_dict())  # 파라미터 동기화
q_target4.load_state_dict(q4.state_dict())  # 파라미터 동기화
q_optimizer1 = optim.Adam(q1.parameters(), lr=0.005)
q_optimizer2 = optim.Adam(q2.parameters(), lr=0.005)
q_optimizer3 = optim.Adam(q3.parameters(), lr=0.001)
q_optimizer4 = optim.Adam(q4.parameters(), lr=0.01)

mu1 = MuNet1().to(device)
mu2 = MuNet2().to(device)
mu3 = MuNet3().to(device)
mu4 = MuNet4().to(device)
mu_target1 = MuNet1().to(device)
mu_target2 = MuNet2().to(device)
mu_target3 = MuNet3().to(device)
mu_target4 = MuNet4().to(device)

mu_target1.load_state_dict(mu1.state_dict())  # 파라미터 동기화
mu_target2.load_state_dict(mu2.state_dict())  # 파라미터 동기화
mu_target3.load_state_dict(mu3.state_dict())  # 파라미터 동기화
mu_target4.load_state_dict(mu4.state_dict())  # 파라미터 동기화
mu_optimizer1 = optim.Adam(mu1.parameters(), lr = 0.0005)
mu_optimizer2 = optim.Adam(mu2.parameters(), lr = 0.005)
mu_optimizer3 = optim.Adam(mu3.parameters(), lr = 0.05)
mu_optimizer4 = optim.Adam(mu4.parameters(), lr = 0.01)

ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(4))
MAX_EPISODES = 1000
avg_history = []
reward_history_20 = []
episode = 0

# DECAY_RATE = 3
BAD_REWARD_FLAG = False
BAD_REWARD_SIGNAL = 0

while episode < MAX_EPISODES:
    state = env.reset()
    done = False
    score = 0.0

    while not done:

        # Batch Norm 계산을 위해 ..
        stack = [state] * 2
        stack = np.array(stack)
        stack = torch.from_numpy(stack).float().to(device).squeeze(0)

        action1 = mu1(stack)
        action2 = mu2(stack)
        action3 = mu3(stack)
        action4 = mu4(stack)

        q_value_for_softmax1 = q1(stack, action1)[0].unsqueeze(0)
        q_value_for_softmax2 = q2(stack, action2)[0].unsqueeze(0)
        q_value_for_softmax3 = q3(stack, action3)[0].unsqueeze(0)
        q_value_for_softmax4 = q3(stack, action4)[0].unsqueeze(0)

        # Voting
        actions = torch.stack([q_value_for_softmax1, q_value_for_softmax2, q_value_for_softmax3, q_value_for_softmax4])
        action_softmax = torch.nn.functional.softmax(actions, dim=0).squeeze(1).squeeze(1)

        action_list = [action1[0], action2[0], action3[0], action4[0]]
        action_index = [0, 1, 2, 3]

        # Softmax Choice
        choice_action = np.random.choice(action_index, 1, p=action_softmax.cpu().detach().numpy())
        # Best Choice
        best_action = torch.argmax(actions)

        sample = random.random()
        if sample > 0.1:
            action = action_list[best_action].cpu().detach().numpy()
            if BAD_REWARD_FLAG == True:
                next_state, reward, done, _ = env.step(action * 1.2)
            else:
                next_state, reward, done, _ = env.step(action)
        else:
            action = action_list[choice_action[0]].cpu().detach().numpy()
            if BAD_REWARD_FLAG == True:
                next_state, reward, done, _ = env.step(action * 1.2)
            else:
                next_state, reward, done, _ = env.step(action)

        memory.put((state, action, reward / 10.0, next_state, done))
        score += reward
        state = next_state

    if memory.size() > 5000:
        for _ in range(10):
            # Bagging 을 통해 Variance 줄이기
            train(mu1, mu_target1, q1, q_target1, memory, q_optimizer1, mu_optimizer1, batch_size = 128)
            train(mu2, mu_target2, q2, q_target2, memory, q_optimizer2, mu_optimizer2, batch_size = 128)
            train(mu3, mu_target3, q3, q_target3, memory, q_optimizer3, mu_optimizer3, batch_size = 128)
            train(mu4, mu_target4, q4, q_target4, memory, q_optimizer4, mu_optimizer4, batch_size = 128)

            soft_update(mu1, mu_target1)
            soft_update(mu2, mu_target2)
            soft_update(mu3, mu_target3)
            soft_update(mu4, mu_target4)
            soft_update(q1, q_target1)
            soft_update(q2, q_target2)
            soft_update(q3, q_target3)
            soft_update(q4, q_target4)

    # Moving Average Count
    reward_history_20.insert(0, score)
    if len(reward_history_20) == 10:
        reward_history_20.pop()
    avg = sum(reward_history_20) / len(reward_history_20)

    # BAD REWARD 5번 연속 ... 로봇 흔들기
    if avg < -90:
        BAD_REWARD_SIGNAL += 1
        if BAD_REWARD_SIGNAL == 5:
            BAD_REWARD_FLAG = True
            BAD_REWARD_SIGNAL = 0
    else:
        BAD_REWARD_SIGNAL = 0
        BAD_REWARD_FLAG = False

    avg_history.append(avg)
    if episode % 10 == 0:
        print('episode: {} | reward: {:.1f} | 10 avg: {:.1f} | Noise : {}'.format(episode, score, avg, BAD_REWARD_FLAG))
    episode += 1

env.close()

#######################################################################
# Result Graph

length = np.arange(len(avg_history))
plt.figure()
plt.xlabel("Episode")
plt.ylabel("10 episode MVA")
plt.plot(length, avg_history)
plt.savefig('Another Model 2 check.png')
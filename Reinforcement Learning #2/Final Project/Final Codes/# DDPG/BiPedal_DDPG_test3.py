# 코드 동작 OK
# 학습 X
###########################################################################
# To Avoid Library Collision
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
###########################################################################

import gym
import sys
import copy
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
gamma = 0.99          # discount factor
buffer_limit = 100000 # Replay Memory Size
tau = 0.01            # for target network soft update


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
class MuNet1(nn.Module):
    def __init__(self):
        super(MuNet1, self).__init__()
        self.fc1 = nn.Linear(24, 256)  # Input  : 24 continuous states
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, 4) # Output : 4 continuous actions

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
        x = torch.tensor(x, device = device, dtype = torch.float)

        return x

###########################################################################
# Train ...
def train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer, batch_size):
    states, actions, rewards, next_states, dones = memory.sample(batch_size)
    Critic, Actor = 0.0 , 0.0

    with torch.no_grad():
        y = rewards + ( gamma * q_target(next_states, mu_target(next_states)) * dones )

    Critic = torch.nn.functional.smooth_l1_loss( q(states, actions), y.detach() )
    q_optimizer.zero_grad()
    Critic.backward()
    q_optimizer.step()

    Actor = -q(states, mu(states)).mean()
    mu_optimizer.zero_grad()
    Actor.backward()
    mu_optimizer.step()

    soft_update(mu, mu_target)
    soft_update(q, q_target)

# Soft Update
def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


# state  : continuous 24 state
# action : continuous 4 action
env = gym.make('BipedalWalker-v3')
env.reset()

memory = ReplayBuffer()

# 2개의 동일한 네트워크 생성 ...
q1 =  QNet().to(device)
q_target1 = copy.deepcopy(q1).eval().to(device) # Parameter Synchronize

# 8개의 동일한 Actor Network 생성
mu1 = MuNet1().to(device)
mu2 = MuNet1().to(device)
mu3 = MuNet1().to(device)
mu4 = MuNet1().to(device)
mu_target1 = copy.deepcopy(mu1).eval().to(device) # Parameter Synchronize
mu_target2 = copy.deepcopy(mu2).eval().to(device) # Parameter Synchronize
mu_target3 = copy.deepcopy(mu3).eval().to(device) # Parameter Synchronize
mu_target4 = copy.deepcopy(mu4).eval().to(device) # Parameter Synchronize


# Target Network Gradient 계산 중지
for p in q_target1.parameters():
    p.requires_grad = False

for m in mu_target1.parameters():
    m.requires_grad = False
for m in mu_target2.parameters():
    m.requires_grad = False
for m in mu_target3.parameters():
    m.requires_grad = False
for m in mu_target4.parameters():
    m.requires_grad = False

# Optimizer 생성 및 하이퍼파라미터 튜닝
mu_optimizer1 = optim.Adam(mu1.parameters(), lr=0.009)
mu_optimizer2 = optim.Adam(mu2.parameters(), lr=0.005)
mu_optimizer3 = optim.Adam(mu3.parameters(), lr=0.003)
mu_optimizer4 = optim.Adam(mu4.parameters(), lr=0.001)

q_optimizer1  = optim.Adam(q1.parameters(), lr=0.005)

ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(4))
MAX_EPISODES = 300

DECAYING_RATE = 2      # For Decaying Noise
avg_history       = [] # Average Reward List
reward_history_20 = [] # Reward List
avg = 0.0

episode = 0
while episode < MAX_EPISODES:
    state = env.reset()
    done = False
    score = 0.0
    while not done:

        # 100 ~ 150사이의 Episode에서 Noise 사라지도록
        DECAY = DECAYING_RATE - episode * 0.01
        if DECAY < 0:
            DECAY = 0
        state = torch.from_numpy(state).float().to(device)

        # 8개의 Noise 생성
        noise1 = ou_noise() * DECAY
        noise2 = ou_noise() * DECAY
        noise3 = ou_noise() * DECAY
        noise4 = ou_noise() * DECAY

        with torch.no_grad():

            # 8개의 Action + Noise
            action1 = mu1(state) + noise1
            action2 = mu2(state) + noise2
            action3 = mu3(state) + noise3
            action4 = mu4(state) + noise4

            # 8개의 Action에 대한 Q Value 계산
            q_value_for_softmax1 = q1(state.unsqueeze(0), action1.unsqueeze(0))[0][0].unsqueeze(0)
            q_value_for_softmax2 = q1(state.unsqueeze(0), action2.unsqueeze(0))[0][0].unsqueeze(0)
            q_value_for_softmax3 = q1(state.unsqueeze(0), action3.unsqueeze(0))[0][0].unsqueeze(0)
            q_value_for_softmax4 = q1(state.unsqueeze(0), action4.unsqueeze(0))[0][0].unsqueeze(0)

        # Soft Voting
        actions = torch.stack(
            [q_value_for_softmax1, q_value_for_softmax2, q_value_for_softmax3, q_value_for_softmax4])


        #print(actions.shape)
        #sys.exit()
        action_softmax = torch.nn.functional.softmax(actions, dim=0).squeeze(1)

        # Soft Voting
        action_list = [action1, action2, action3, action4]
        action_index = [0, 1, 2, 3]
        choice_action = np.random.choice(action_index, 1, p=action_softmax.cpu().detach().numpy())

        # Noise가 사라지면, Deterministic하게 또는 Uniform하게 Optimal Action을 선택
        action = action_list[choice_action[0]].cpu().detach().numpy()

        next_state, reward, done, info = env.step(action)
        memory.put((state.cpu().numpy(), action, reward, next_state, done))
        score = score + reward
        state = next_state

        if memory.size() > 2000:
            for i in range(5):
                train(mu1, mu_target1, q1, q_target1, memory, q_optimizer1, mu_optimizer1, batch_size=32)
                train(mu2, mu_target2, q1, q_target1, memory, q_optimizer1, mu_optimizer2, batch_size=32)
                train(mu3, mu_target3, q1, q_target1, memory, q_optimizer1, mu_optimizer3, batch_size=32)
                train(mu4, mu_target4, q1, q_target1, memory, q_optimizer1, mu_optimizer4, batch_size=32)

    # Moving Average Count
    reward_history_20.append(score)
    if len(reward_history_20) > 10:
        avg = sum(reward_history_20[-10:]) / 10
        avg_history.append(avg)
    if episode % 10 == 0:
        print('episode: {} | reward: {:.1f} | 10 avg: {:.1f} '.format(episode, score, avg))
    episode += 1

env.close()

# Numpy array로 list 저장
avg_history = np.array(avg_history)
np.save("./type3", avg_history)

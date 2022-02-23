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
lr_mu = 0.01  # Learning Rate for Torque (Action)
lr_q = 0.00005  # Learning Rate for Q
gamma = 0.99  # discount factor
batch_size = 100  # Mini Batch Size for Sampling from Replay Memory
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


class MuNet1(nn.Module):  # Output : Deterministic Action
    def __init__(self):
        super(MuNet1, self).__init__()
        self.fc1A = nn.Linear(24, 32)  # Input  : 24 continuous states
        self.bn1A = nn.BatchNorm1d(32)
        self.fc2A = nn.Linear(32, 16)
        self.bn2A = nn.BatchNorm1d(16)
        self.fc3A = nn.Linear(16, 16)
        self.bn3A = nn.BatchNorm1d(16)
        self.fc_muA = nn.Linear(16, 4)  # Output : 4 continuous actions

        self.fc1B = nn.Linear(24, 64)  # Input  : 24 continuous states
        self.bn1B = nn.BatchNorm1d(64)
        self.fc2B = nn.Linear(64, 32)
        self.bn2B = nn.BatchNorm1d(32)
        self.fc3B = nn.Linear(32, 8)
        self.bn3B = nn.BatchNorm1d(8)
        self.fc_muB = nn.Linear(8, 4)  # Output : 4 continuous actions

        self.fc1C = nn.Linear(24, 64)  # Input  : 24 continuous states
        self.bn1C = nn.BatchNorm1d(64)
        self.fc2C = nn.Linear(64, 16)
        self.bn2C = nn.BatchNorm1d(16)
        self.fc3C = nn.Linear(16, 8)
        self.bn3C = nn.BatchNorm1d(8)
        self.fc_muC = nn.Linear(8, 4)  # Output : 4 continuous actions

    def forward(self, x):
        # Block 1
        xA = F.relu(self.bn1A(self.fc1A(x)))
        xA = F.relu(self.bn2A(self.fc2A(xA)))
        xA = F.relu(self.bn3A(self.fc3A(xA)))

        # Block 2
        xB = F.relu(self.bn1B(self.fc1B(x)))
        xB = F.relu(self.bn2B(self.fc2B(xB)))
        xB = F.relu(self.bn3B(self.fc3B(xB)))

        # Block 3
        xC = F.relu(self.bn1C(self.fc1C(x)))
        xC = F.relu(self.bn2C(self.fc2C(xC)))
        xC = F.relu(self.bn3C(self.fc3C(xC)))

        muA = torch.tanh(self.fc_muA(xA))
        muB = torch.tanh(self.fc_muB(xB))
        muC = torch.tanh(self.fc_muC(xC))

        return [muA, muB, muC]


class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(24, 32)  # State  24 개
        self.fc_a = nn.Linear(4, 32)  # Action 4  개
        self.fc_q = nn.Linear(64, 16)  # State , Action 이어붙이기
        #self.bn1 = nn.BatchNorm1d(16)
        self.fc_out = nn.Linear(16, 8)  # Output : Q value
        #self.bn2 = nn.BatchNorm1d(8)
        self.fc_out2 = nn.Linear(8, 1)  # Output : Q value

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))  # 128
        h2 = F.relu(self.fc_a(a))  # 128
        cat = torch.cat([h1, h2], dim=1)  # 256
        q = F.relu(self.fc_q(cat))  # 128
        q = F.relu(self.fc_out(q))  # 64
        #q = F.relu(self.bn1(self.fc_q(cat)))  # 128
        #q = F.relu(self.bn2(self.fc_out(q)))  # 64
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
def train(mu1, mu_target1, q, q_target, memory, q_optimizer, mu_optimizer1):
    states, actions, rewards, next_states, dones = memory.sample(batch_size)
    Critic = 0.0

    with torch.no_grad():
        actions  = mu1(states) # (batchsize x 4) tensor 3개 return ..
        action1 = actions[0]
        action2 = actions[1]
        action3 = actions[2]

        q_value_for_softmax1 = q(states, action1).sum().unsqueeze(0)
        q_value_for_softmax2 = q(states, action2).sum().unsqueeze(0)
        q_value_for_softmax3 = q(states, action3).sum().unsqueeze(0)

        action_stack = torch.stack([q_value_for_softmax1, q_value_for_softmax2, q_value_for_softmax3])
        min_idx  = torch.argmin(action_stack)

    y1 = rewards + gamma * q_target(next_states, mu_target1(next_states)[min_idx]) * dones

    Critic = torch.nn.functional.smooth_l1_loss(q(states, actions[min_idx]), y1.detach())

    q_optimizer.zero_grad()
    Critic.backward()
    q_optimizer.step()

    Actor1 = -q(states, mu1(states)[min_idx]).mean()

    mu_optimizer1.zero_grad()
    Actor1.backward()
    mu_optimizer1.step()


# Soft Update
def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


env = gym.make('BipedalWalker-v3')
memory = ReplayBuffer()

# 1개의 Q Net
q = QNet().to(device)
q_target = QNet().to(device)

q_target.load_state_dict(q.state_dict())  # 파라미터 동기화
q_optimizer = optim.Adam(q.parameters(), lr=lr_q)

mu1 = MuNet1().to(device)
mu_target1 = MuNet1().to(device)

mu_target1.load_state_dict(mu1.state_dict())  # 파라미터 동기화
mu_optimizer1 = optim.Adam(mu1.parameters(), lr=lr_mu)

ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(4))
MAX_EPISODES = 1000
avg_history = []
reward_history_20 = []
episode = 0

DECAY_RATE = 3
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

        action_list = mu1(stack)
        action1, action2, action3 = action_list[0], action_list[1], action_list[2]

        q_value_for_softmax1 = q(stack, action1)[0].unsqueeze(0)
        q_value_for_softmax2 = q(stack, action2)[0].unsqueeze(0)
        q_value_for_softmax3 = q(stack, action3)[0].unsqueeze(0)

        # 값이 변화하는지 체크

        #print("Actions ", action1[0], action2[0], action3[0])
        #print("Q Values", q_value_for_softmax1.item(), q_value_for_softmax2.item(), q_value_for_softmax3.item())

        actions = torch.stack([q_value_for_softmax1, q_value_for_softmax2, q_value_for_softmax3])
        #print(actions)
        action_softmax = torch.nn.functional.softmax(actions, dim=0).squeeze(1).squeeze(1)

        # 값이 변화하는지 체크
        print("Action Softmax", action_softmax)


        #sys.exit()



        action_list = [action1[0], action2[0], action3[0]]
        action_index = [0, 1, 2]

        choice_action = np.random.choice(action_index, 1, p=action_softmax.cpu().detach().numpy())
        action = action_list[choice_action[0]].cpu().detach().numpy()

        '''
        # 5번 마다 Noise 추가 ...
        if BAD_REWARD_FLAG == True:
            noise = torch.tensor(ou_noise(), device=device) * (DECAY_RATE - (0.0001 * episode))
            action = action_list[choice_action[0]]
            action = (action + noise).cpu().detach().numpy()
        else:
            action = action_list[choice_action[0]].cpu().detach().numpy()

        action = action1[0].cpu().detach().numpy()
        '''

        next_state, reward, done, _ = env.step(action)
        memory.put((state, action, reward / 100.0, next_state, done))
        score += reward
        state = next_state

    if memory.size() > 2000:
        for _ in range(10):
            # Bagging 을 통해 Variance 줄이기
            train(mu1, mu_target1, q, q_target, memory, q_optimizer, mu_optimizer1)

        soft_update(mu1, mu_target1)
        soft_update(q, q_target)

    # Moving Average Count
    reward_history_20.insert(0, score)
    if len(reward_history_20) == 10:
        reward_history_20.pop()
    avg = sum(reward_history_20) / len(reward_history_20)

    '''
    # Noise Signal ...
    if avg < -90:
        BAD_REWARD_SIGNAL += 1
        if BAD_REWARD_SIGNAL == 3:
            BAD_REWARD_FLAG = True
            BAD_REWARD_SIGNAL = 0
    else:
        BAD_REWARD_SIGNAL = 0
    '''
    avg_history.append(avg)
    if episode % 10 == 0:
        print('episode: {} | reward: {:.1f} | 10 avg: {:.1f}'.format(episode, score, avg))
    episode += 1

env.close()

#######################################################################
# Record Hyperparamters & Result Graph

with open('Another Model.txt', 'w', encoding='UTF-8') as f:
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
    f.write("batch_size    : " + str(batch_size) + '\n')
    f.write("buffer_limit  : " + str(buffer_limit) + '\n')
    f.write("memory.size() : 500" + '\n')
    f.write("# ----------------------- # " + '\n')

length = np.arange(len(avg_history))
plt.figure()
plt.xlabel("Episode")
plt.ylabel("10 episode MVA")
plt.plot(length, avg_history)
plt.savefig('Another Model.png')
import sys
import gym
import random
import collections
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal # Normal distribution

from time import sleep
from IPython.display import clear_output

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        # State, next_state    : vector 형태
        # reward, action, done : scalar 형태
        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_mask_lst.append([done_mask])

        # To Tensor , To GPU
        return torch.tensor(s_lst, device = device, dtype=torch.float), torch.tensor(a_lst, device = device, dtype=torch.float), \
               torch.tensor(r_lst, device = device, dtype=torch.float), torch.tensor(s_prime_lst, device = device, dtype=torch.float), \
               torch.tensor(done_mask_lst, device = device, dtype=torch.float)

    def size(self):
        return len(self.buffer)

class QNet(nn.Module):
    def __init__(self, learning_rate):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(3, 64)     # Input 3
        self.fc_a = nn.Linear(1, 64)     # Input 1
        self.fc_cat = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32, 1)   # Output 1 : Q Value

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1) # Concatenate...
        q = F.relu(self.fc_cat(cat))
        q = self.fc_out(q)
        return q # Output : Q(s, a)

    # Estimate Q - Step 2
    def train_net(self, target, mini_batch):

        s, a, r, s_prime, done = mini_batch

        Q_output = self.forward(s, a)
        loss = F.smooth_l1_loss(Q_output, target) # Target : y = r + (gamma * V)

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    def soft_update(self, net_target):
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

class PolicyNet(nn.Module):
    def __init__(self, learning_rate):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(3, 128) # Input : State (COS , SIN, 각속도)
        self.fc_mu = nn.Linear(128, 1) # Output Action (토크)

        self.fc_std = nn.Linear(128, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.log_alpha = torch.tensor(np.log(init_alpha))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

    def forward(self, x): # x : state
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)  # mu : 토크

        # softplus : smooth approximation of relu function
        std = F.softplus(self.fc_std(x))

        # a = f_w(zeta, state) ...
        # action을 Sampling하는 function을 parameterize -> Deterministic한 Action을 Stochastic하게 ...
        dist = Normal(mu, std)  # normal distribution - 평균 : mu, 표준편차 : std(학습 ... )

        # Action 출력
        action = dist.rsample() # rsample : Backpropagation을 수행할 때 특정 변수 (파라미터) 미분 X
        real_action = torch.tanh(action)  # action : -1 ~ 1

        # Action의 확률 ...
        log_prob = dist.log_prob(action)
        real_log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)

        # real_log_porb : state x에서 real_action을 할 확률
        return real_action, real_log_prob

    def train_net(self, q1, q2, mini_batch):

        # dist를 학습해야한다
        pass
        # Fill in this part

def calc_target(pi, q1, q2, mini_batch):
    s, a, r, s_prime, done = mini_batch

    with torch.no_grad():

        # action, action의 확률
        a_prime, log_prob = pi(s_prime)
        entropy = -pi.log_alpha.exp() * log_prob

        q1_val = q1(s_prime, a_prime)
        q2_val = q2(s_prime, a_prime)

        q1_q2 = torch.cat([q1_val, q2_val], dim=1)

        # To mitigate Positive Bias ... Choose minimum Q Value
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]

        # entropy : H
        # min_q + H = V
        target = r + gamma * (min_q + entropy) * done

    return target

# Hyperparameters
tau          = 0.01       # Soft update
lr_pi        = 0.0005     # Pi network learning rate
lr_q         = 0.001      # Q Network learning rate
gamma        = 0.98       # Discount factor
batch_size   = 32         # Batch Sampling Number
buffer_limit = 50000

############################################################ - ?
init_alpha = 0.01
target_entropy = -1.0  # for automated alpha update
lr_alpha = 0.001       # for automated alpha update
############################################################ - ?

env = gym.make('Pendulum-v0')
memory = ReplayBuffer()

q1        = QNet(lr_q)
q2        = QNet(lr_q)
q1_target = QNet(lr_q)
q2_target = QNet(lr_q)
pi = PolicyNet(lr_pi)

q1_target.load_state_dict(q1.state_dict())
q2_target.load_state_dict(q2.state_dict())

score = 0.0
scores = []
print_interval = 20

for n_epi in range(2000):
    s = env.reset()
    done = False

    while not done:

        # action , ln (pi (action | state))
        a, log_prob = pi(torch.from_numpy(s).float())

        s_prime, r, done, info = env.step([2.0 * a.item()]) # Torque 가 -2 ~ 2 사이 값이므로 ..
        memory.put((s, a.item(), r / 10.0, s_prime, done))
        score += r
        s = s_prime

    if memory.size() > 1000:
        for i in range(20):

            # Sampling from Replay memory
            mini_batch = memory.sample(batch_size)

            # td_target =
            td_target = calc_target(pi, q1_target, q2_target, mini_batch)


            q1.train_net(td_target, mini_batch)
            q2.train_net(td_target, mini_batch)
            entropy = pi.train_net(q1, q2, mini_batch)

            # Soft Update
            q1.soft_update(q1_target)
            q2.soft_update(q2_target)

    if n_epi % print_interval == 0 and n_epi != 0:
        clear_output(wait=True)
        print("# of episode :{}, avg score : {:.1f} alpha:{:.4f}".format(n_epi, score / print_interval,
                                                                         pi.log_alpha.exp()))
        scores.append(score)
        score = 0.0

env.close()


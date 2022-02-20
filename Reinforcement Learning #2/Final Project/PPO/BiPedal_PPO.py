'''
# Info
# 이산화 과정 ~ ing
'''
import gym
import sys
import copy
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.distributions import Categorical
from time import sleep
from collections import deque

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("")
print(f"On {device}")
print("")

## Hyper-Parameters
lr       = 0.0005 # Learning Rate
gamma    = 0.98   # Discount Factor
LD       = 0.95   # GAE
Eps_clip = 0.1    # L_Clip 범위
K        = 3      # 모아둔 데이터 반복 학습 횟수
T        = 200    # 데이터 모을 Time Step

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()

        self.fc1 = nn.Linear(24 , 128)
        self.fc2 = nn.Linear(128, 256)

        self.fc_pi = nn.Linear(256, 4) # Action에 대한 확률
        self.fc_v  = nn.Linear(256, 1)

    # Actor
    def pi(self, x, softmax_dim = 0 ):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim = softmax_dim) # 각 Action에 대해 Softmax
        return x, prob # Action + SoftMax

    # Critic
    def v(self, x): # rV(S') - V(S) 를 통해 GAE 찾을 것
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_v(x)
        return v


def make_batch():
    states, actions, rewards, next_states, probs, dones = [],[],[],[],[],[]

    for transition in data:
        # state, action ,... 데이타 뭉치들
        state, action, reward, next_state, prob, done = transition
        states.append(state)
        actions.append(action)
        rewards.append([reward])
        next_states.append(next_state)
        probs.append(prob)

        done_mask = 0 if done else 1
        dones.append(done_mask)

    states, actions, rewards, next_states, dones, probs = torch.tensor(states, dtype=torch.float),      \
                                          torch.tensor(actions)                                  ,      \
                                          torch.tensor(rewards, dtype=torch.float)               ,      \
                                          torch.tensor(next_states,dtype=torch.float)            ,      \
                                          torch.tensor(dones, dtype=torch.float)                 ,      \
                                          torch.tensor(probs)

    return states, actions, rewards, next_states, dones, probs

def train(ppo, optimizer):
    states, actions, rewards, next_states, dones, probs = make_batch()

    for i in range(K): # 같은 배치 데이터에 대해 K번 학습
        TD_Target = rewards + gamma * ppo.v(next_states) * dones
        Delta = TD_Target - ppo.v(states)
        Delta = Delta.detach().numpy()


        GAE_list = []
        GAE = 0.0
        for Delta_t in Delta[::-1]:
            GAE = gamma * LD * GAE + Delta_t[0]
            GAE_list.append([GAE])
        GAE_list.reverse()

        GAE_Value = torch.tensor(GAE_list, dtype = torch.float())
        PI = ppo.pi(states, softmax_dim = 1)
        PI_action_probs = PI.gather(1, actions)
        Ratio = torch.exp(torch.log(PI_action_probs) - torch.log(probs))

        Surrogate1 = Ratio * GAE_Value
        Surrogate2 = torch.clamp(Ratio, 1 - Eps_clip, 1 + Eps_clip) * GAE_Value

        loss = -torch.min(Surrogate1, Surrogate2) + F.smooth_l1_loss(ppo.v(states) - TD_Target.detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

env = gym.make('BipedalWalker-v3')
ppo = PPO().to(device)
optimizer = optim.Adam(ppo.parameters(), lr = lr)

score = 0.0
episode = 0
MAX_EPISODES = 3000
reward_history = []

A = np.arange(-1, 1, 0.0001)
while episode < MAX_EPISODES:
    state = env.reset()
    done = False
    data = []
    while not done:
        # T Step 동안 데이터 수집
        for t in range(T):
            actions, prob   = ppo.pi(torch.from_numpy(state).float().to(device))
            print(prob)
            sys.exit()
            action_index    = Categorical(prob).sample().item()
            action = actions[action_index].cpu().detach().numpy()

            discrete_action = np.array([np.digitize(action, bins = A)])
            action = A[discrete_action - 1]

            # Discretized Action : action
            # Action Index       : action_index
            # Action Probability : prob
            # 해당 Action의 Softmax 확률과 Action의 Index가 필요




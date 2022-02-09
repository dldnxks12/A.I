# DQN - A2C
# 동시에 Gradient 계산 Ok

import gym
import sys
import math
import torch
import random
import collections
import numpy as np
from time import sleep
from collections import deque

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'

# Policy
class PolicyNetwork(torch.nn.Module): # torch Module Import...
  def __init__(self):
    super().__init__() # Parent Class def __init__ 친구들 상속
    self.fcA1 = torch.nn.Linear(4, 10) # Input  : state  (속도, 가속도, 각도, 각속도)
    self.fcA2 = torch.nn.Linear(10, 2) # Output : Action (왼쪽 , 오른쪽)

  def forward(self, x): # x : state
    x      = self.fcA1(x)
    x      = torch.nn.functional.relu(x)
    x      = self.fcA2(x)
    x      = torch.nn.functional.softmax(x, dim = -1)
    return x

# Q Value function
class QNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fcS1 = torch.nn.Linear(4, 16)
        self.fcA1 = torch.nn.Linear(1, 16)
        self.fcQ1 = torch.nn.Linear(32, 16)
        self.fcQ2 = torch.nn.Linear(16, 1)

    def forward(self, x, a): # Input : state / Output : Q value
        h1 = self.fcS1(x)
        h1 = torch.nn.functional.relu(h1)
        h2 = self.fcA1(a)
        h2 = torch.nn.functional.relu(h2)
        cat = torch.cat([h1, h2], dim = -1)
        q = self.fcQ1(cat)
        q = torch.nn.functional.relu(q)
        q = self.fcQ2(q)
        return q

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen = 50000)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for transition in mini_batch:
            state, action, reward, next_state, done = transition
            states.append(state)
            actions.append([action])
            rewards.append([reward])
            next_states.append(next_state)
            done_mask = 0.0 if done else 1.0
            dones.append([done_mask])

        return torch.tensor(states, device = device, dtype = torch.float), torch.tensor(actions, device = device, dtype = torch.float), \
               torch.tensor(rewards,  device = device,dtype = torch.float), torch.tensor(next_states,  device = device,dtype = torch.float),\
               torch.tensor(dones,  device = device,dtype = torch.float)

    def size(self):
        return len(self.buffer)

def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - beta) + param.data * beta)

# Replay buffer object ...
memory = ReplayBuffer()
alpha = 0.001    # Learning rate
gamma = 0.99     # Discount factor
beta = 0.005     # Soft Update rate
MAX_EPISODE = 10000
episode = 0

pi       = PolicyNetwork().to(device)
Q        = QNetwork().to(device)
Q_target = QNetwork().to(device)
Q_target.load_state_dict(Q.state_dict())   # 파라미터 동기화

pi_optimizer = torch.optim.Adam(pi.parameters(), lr = alpha)
Q_optimizer  = torch.optim.Adam(Q.parameters(), lr = alpha)
env = gym.make('CartPole-v1')


def train(memory, Q, Q_target, Q_optimizer):
    states, actions, rewards, next_states, dones = memory.sample(32)
    critic = 0
    actor = 0
    for (state, action, reward, next_state, done) in zip(states, actions, rewards, next_states, dones):
        if done == 0:
            y = reward
        else:
            with torch.no_grad():
                next_action = torch.FloatTensor([max(pi(next_state))])
                next_action = next_action.to(device)
                y = reward + gamma * Q_target(next_state, next_action)

        critic += (y - Q(state, action)) ** 2

    critic = critic / 64
    Q_optimizer.zero_grad()
    critic.backward()
    Q_optimizer.step()

    # Soft Update  ...
    soft_update(Q, Q_target)

    for (state, action, reward, next_state, done) in zip(states, actions, rewards, next_states, dones):
        with torch.no_grad():
            result = Q(state, action)
        action = torch.tensor(action, device = device, dtype = torch.int)
        action = action.item()
        actor += result * ((pi(state)[action] + 1e-5).log())

    actor = -actor / 32

    pi_optimizer.zero_grad()
    actor.backward()
    pi_optimizer.step()

while episode < MAX_EPISODE:

    state = env.reset() # state : numpy
    done = False
    score = 0

    while not done:

        state = np.array(state)
        policy = pi(torch.from_numpy(state).float().to(device))
        action = torch.multinomial(policy, 1).item()
        next_state, reward, done, info = env.step(action)
        memory.put((state, action, reward, next_state, done))

        score += reward
        state = next_state

        if memory.size() > 2000:
            for i in range(10):
                train(memory, Q, Q_target, Q_optimizer)


    print(f"Episode : {episode} || Reward : {score} ")
    episode += 1
env.close()






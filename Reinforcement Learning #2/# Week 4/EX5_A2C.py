# DQN - A2C
# 1.30 실패 - Tensorflow 기반으로 다시 해보기

import gym
import sys
import math
import random
import torch
import numpy as np
from time import sleep

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Policy
class PolicyNetwork(torch.nn.Module): # torch Module Import...
  def __init__(self):
    super().__init__() # Parent Class def __init__ 친구들 상속
    self.fcA1 = torch.nn.Linear(4, 10) # Input  : state  (속도, 가속도, 각도, 각속도)
    self.fcA2 = torch.nn.Linear(10, 2) # Output : Action (왼쪽 , 오른쪽)

  def forward(self, x): # x : state
    policy      = self.fcA1(x)
    policy      = torch.nn.functional.relu(policy)
    policy      = self.fcA2(policy)
    policy      = torch.nn.functional.softmax(policy, dim = -1)
    return policy

# Q Value function
class QNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fcQ1 = torch.nn.Linear(4, 256)
        self.fcQ2 = torch.nn.Linear(256, 256)
        self.fcQ3 = torch.nn.Linear(256, 2)

    def forward(self, x): # Input : state / Output : Q value
        x = self.fcQ1(x)
        x = torch.nn.functional.relu(x)
        x = self.fcQ2(x)
        x = torch.nn.functional.relu(x)
        x = self.fcQ3(x)
        return x

class QTargetNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fcQ1 = torch.nn.Linear(4, 256)
        self.fcQ2 = torch.nn.Linear(256, 256)
        self.fcQ3 = torch.nn.Linear(256, 2)

    def forward(self, x): # Input : state / Output : Q value
        x = self.fcQ1(x)
        x = torch.nn.functional.relu(x)
        x = self.fcQ2(x)
        x = torch.nn.functional.relu(x)
        x = self.fcQ3(x)
        return x

alpha = 0.001
gamma = 0.99

pi       = PolicyNetwork().to(device)
Q        = QNetwork().to(device)
Q_target        = QTargetNetwork().to(device)

pi_optimizer       = torch.optim.Adam(pi.parameters(), lr = alpha)
Q_optimizer        = torch.optim.Adam(Q.parameters(), lr = alpha)
Q_target_optimizer = torch.optim.Adam(Q_target.parameters(), lr = alpha)


replay = []
MAX_TRAIN = 1000
episode =0
env = gym.make('CartPole-v1')
while episode < MAX_TRAIN:

    state = env.reset()
    state = torch.FloatTensor(state).to(device)
    done = False

    beta = 0.3
    while not done:
        policy = pi(state)
        action = torch.multinomial(policy, 1).item()
        next_state, reward, done, info = env.step(action)

        replay.append([state, action, reward, next_state])

        if len(replay) <= 50: # Replay Memory에 experience가 쌓일 때 까지 ...
            continue

        Mini_batch = random.sample(replay, 40)
        Critic_loss = 0
        Policy_loss = 0

        for state, action, reward, next_state in Mini_batch:
            if done:
                y = reward
            else:
                y = reward + gamma*Q_target(state)[action]

            Critic_loss += (y - Q(state)[action])**2
            Q_value = (Q(state)).clone()
            Policy_loss +=  Q_value[action]*((pi(state)[action]).log())
        Critic_loss /= len(Mini_batch)

        Q_optimizer.zero_grad()
        Critic_loss.backward()
        Q_optimizer.step()

        for (w_target , w) in zip(Q_target.parameters(), Q.parameters()):
            pass

        for (w_target , w) in zip(Q_target.parameters(), Q.parameters()):
            pass

        Policy_loss = -Policy_loss
        pi_optimizer.zero_grad()
        Policy_loss.backward()
        pi_optimizer.step()

    episode += 1



state = env.reset()
done = False
score = 0
while not done:

    env.render()
    state = torch.FloatTensor(state).to(device)
    policy = pi(state)
    action = torch.multinomial(policy, 1).item()
    next_state, reward, done, info = env.step(action)

    if done:
        reward = -10

    score += reward
    state = next_state

    print(f"Rewards : {score}")




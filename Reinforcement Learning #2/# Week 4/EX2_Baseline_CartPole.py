# Value function을 Baseline function으로 하여 Variance 줄이기
# Temporal difference를 이용하여 Value function 또한 근사

import gym
import sys
import math
import torch
import random
import numpy as np
from time import sleep
from IPython.display import clear_output # 그림을 띄우고 뭘 하고 싶은데, 매번 그림을 띄우고 지우는 코드가 필요

# GPU / CPU Setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Policy
class PolicyNetwork(torch.nn.Module): # torch Module Import...
  def __init__(self):
    super().__init__() # Parent Class def __init__ 친구들 상속
    self.fcA1 = torch.nn.Linear(4, 10) # Input  : state  (속도, 가속도, 각도, 각속도)
    self.fcA2 = torch.nn.Linear(10, 2) # Output : Action (왼쪽 , 오른쪽)

  # softmax ( f(x, w) )
  def forward(self, x): # x : state
    policy      = self.fcA1(x)
    policy      = torch.nn.functional.relu(policy)
    policy      = self.fcA2(policy)
    policy      = torch.nn.functional.softmax(policy, dim = -1)
    return policy

# Value function
class VNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fcV1 = torch.nn.Linear(4, 256)
        self.fcV2 = torch.nn.Linear(256, 256)
        self.fcV3 = torch.nn.Linear(256, 1) # V = x1w1 + x2w2 + x3w2 + x4w4 -> State가 Continuous며 이런식으로 Value functon을 ..

    def forward(self, x):
        x = self.fcV1(x)
        x = torch.nn.functional.relu(x)
        x = self.fcV2(x)
        x = torch.nn.functional.relu(x)
        x = self.fcV3(x)
        return x

def gen_epsiode(ep):
  states , actions, rewards = [], [], []

  state = env.reset()
  done  = False
  score = 0

  while not done:

    if ep % 2000 == 0:
        env.render()

    state = torch.FloatTensor(state).to(device) # Tensor type and to GPU status
    policy= pi(state) # Action에 대한 Softmax (확률 분포) 반환
    action = torch.multinomial(policy, 1).item() # 반환받은 분포에서 1개를 뽑는다. (Index 반환)
    next_state, reward, done, info = env.step(action)

    if done:
      reward = -10

    score += reward

    states.append(state)
    actions.append(action)
    rewards.append(reward)

    state = next_state

  return states, actions, rewards, score

def G_(rewards, t):
    G = 0
    for i, _ in enumerate(rewards, t):
        G += math.pow(gamma, i - t) * rewards[i]
        if i == len(rewards) - 1:
            break
    return G

# Hyperparamer Setting
alpha = 0.001 # TD/MC Update Factor
gamma = 0.99  # Discount Factor

# Model 객체 + Optimizer
pi = PolicyNetwork().to(device)
pi_optimizer = torch.optim.Adam(pi.parameters(), lr=alpha)

V = VNetwork().to(device)
V_optimizer = torch.optim.Adam(V.parameters(), lr=alpha)

env = gym.make('CartPole-v1')

scores = []
episode = 0
MAX_EPISODES = 10000

while episode < MAX_EPISODES:
    states, actions, rewards, score = gen_epsiode(episode)  # 학습하는 거 아니고, 현재 조건에 따라서 생성
    scores.append(score) # for Visualization later with matplotlib

    target = 0
    G = 0
    loss1 = 0
    loss2 = 0
    for time_step, item in enumerate(zip(states, actions, rewards)):
        state = item[0]
        action = item[1]
        reward = item[2]

        G = G_(rewards, time_step)

        loss1 += (G - V(state)) ** 2
        loss2 += time_step*(G - V(state)) * ((pi(state)[action] + 1e-5).log())

    loss2 = -loss2
    V_optimizer.zero_grad()
    pi_optimizer.zero_grad()

    loss1.backward()
    loss2.backward()

    V_optimizer.step()
    pi_optimizer.step()

    Loss = loss1 + loss2
    print(f"Episode : {episode} || Rewards : {score} || Loss : {Loss}")
    episode += 1

env.close()




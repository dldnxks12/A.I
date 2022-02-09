# Gradient Ascent -> Gradient Descent를 위해 Optimizer 이전에 Cost를 음수로 ..

# Q를 G로 근사 ... Q or V의 Network 학습 필요 XX

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

# Q or V의 Linear Approximation을 학습
class PolicyNetwork(torch.nn.Module): # torch Module Import...
  def __init__(self):
    super().__init__() # Parent Class def __init__ 친구들 상속
    self.fcA1 = torch.nn.Linear(4, 10) # Input  : state  (속도, 가속도, 각도, 각속도)
    self.fcA2 = torch.nn.Linear(10, 2) # Output : Action (왼쪽 , 오른쪽)

  # f(x, w) = x1w1 + x2w2 + x3w3 + x4w4
  def forward(self, x): # x : state
    x = self.fcA1(x)
    x = torch.nn.functional.relu(x)
    x = self.fcA2(x)
    x = torch.nn.functional.softmax(x, dim = -1)
    return x  # Softmax Result ---> return 받아서 Log 씌우고 미분 때리기

def gen_epsiode(ep):
  states , actions, rewards = [], [], []

  state = env.reset()
  done  = False
  score = 0

  while not done:

    if ep % 1000 == 0:
        env.render()

    state = torch.FloatTensor(state).to(device) # Tensor type and to GPU status
    probabilities = pi(state) # Action에 대한 Softmax (확률 분포) 반환
    action = torch.multinomial(probabilities, 1).item() # 반환받은 분포에서 1개를 뽑는다. (Index 반환)
    next_state, reward, done, info = env.step(action)


    if done:
      reward = -10

    score += reward

    states.append(state)
    actions.append(action)
    rewards.append(reward)

    state = next_state

  return states, actions, rewards, score


# Version 2 - Use G_0
def G(rewards):
  G_0 = 0
  for i in range(len(rewards) - 1):
    gam = math.pow(gamma, i) # gamma의 i제곱
    G_0 += gam*rewards[i]
  return G_0


# Hyperparamer Setting
alpha = 0.001 # TD/MC Update Factor
gamma = 0.99  # Discount Factor

# Model 객체 생성
pi = PolicyNetwork().to(device)

# Method 1 : Adam Optimizer
# Method 2 : Numerical Grdient Update
pi_optimizer = torch.optim.Adam(pi.parameters(), lr=alpha)  # Gradient Descent 수행

env = gym.make('CartPole-v1')
episode = 0
MAX_EPISODES = 50000

# Vanilla Policy Gradient (High Level : Batch Version)
while episode < MAX_EPISODES:
    states, actions, rewards, score = gen_epsiode(episode)  # 학습하는 거 아니고, 현재 조건에 따라서 생성

    # Get G_0
    G_0 = G(rewards)

    Loss_temp = 0
    for state, action in zip(states, actions):

      soft_max = pi(state) # softmax value ...
      ln_soft_max = soft_max[action].log()
      Loss_temp += ln_soft_max

    loss = -Loss_temp * G_0
    pi_optimizer.zero_grad()

    loss.backward()
    pi_optimizer.step()

    print(f"Episode : {episode} || Rewards : {score} || Loss : {loss.item()}")
    episode += 1

env.close()




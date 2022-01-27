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
from torchviz import make_dot

# GPU / CPU Setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyNetwork(torch.nn.Module): # torch Module Import...
  def __init__(self):
    super().__init__() # Parent Class def __init__ 친구들 상속
    self.fcA1 = torch.nn.Linear(4, 10) # Input  : state  (속도, 가속도, 각도, 각속도)
    self.fcA2 = torch.nn.Linear(10, 2) # Output : Action (왼쪽 , 오른쪽)

  # f(x, w) = x1w1 + x2w2 + x3w3 + x4w4
  def forward(self, x): # x : state
    policy      = self.fcA1(x)
    policy      = torch.nn.functional.relu(policy)
    policy      = self.fcA2(policy)
    policy      = torch.nn.functional.softmax(policy, dim = -1)

    value      = self.fcA1(x)
    value      = self.fcA2(value)
    return policy, value

def gen_epsiode(ep):
  states , actions, rewards = [], [], []

  state = env.reset()
  done  = False
  score = 0

  # value function ...
  value = 0
  while not done:

    if ep % 1000 == 0:
        env.render()

    state = torch.FloatTensor(state).to(device) # Tensor type and to GPU status
    probabilities, value = pi(state) # Action에 대한 Softmax (확률 분포) 반환
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
MAX_EPISODES = 10000

def G_(rewards, t):
    G = 0
    for i, _ in enumerate(rewards, t):
        G += math.pow(gamma, i - t) * rewards[i]
        if i == len(rewards) - 1:
            break
    return G

V = torch.from_numpy(np.zeros(2))
while episode < MAX_EPISODES:
    states, actions, rewards, score = gen_epsiode(episode)  # 학습하는 거 아니고, 현재 조건에 따라서 생성

    Loss_temp = 0
    G = 0
    for time_step, item in enumerate(zip(states, actions, rewards)):
        if time_step == len(states) - 1:
            break
        state = item[0]
        action = item[1]
        reward = item[2]

        # get G
        G = G_(rewards, time_step)

        # get V
        next_state = states[time_step + 1]

        # TD Update
        # value approximation : x1w1 + x2w2 + x3w3 +x4w4
        policy, value = pi(state)
        _, next_value = pi(next_state)

        with torch.no_grad():
            V[action] = V[action] + alpha*(reward + (gamma*next_value[action]) - value[action])

        # Target
        target = G - V

        gamma_target = math.pow(gamma, time_step) * target
        Loss_temp += ((policy + 1e-5).log()) * gamma_target

    Loss_temp = Loss_temp.sum()
    Loss = - Loss_temp

    pi_optimizer.zero_grad()
    Loss.backward()

    pi_optimizer.step()

    print(f"Episode : {episode} || Value : {V.sum()} || Rewards : {score} || Loss : {Loss.item()}")
    episode += 1

env.close()




# 학습이 진행되지 않음

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


# We dont Need G anymore
# We gonna Use TD Advantage r + gamma*V(S') - V(S)

alpha = 0.005
gamma = 0.99

pi = PolicyNetwork().to(device)
V  = VNetwork().to(device)

pi_optimizer = torch.optim.Adam(pi.parameters(), lr = alpha)
V_optimizer  = torch.optim.Adam(V.parameters(), lr = alpha)

env = gym.make('CartPole-v1')

scores = []
episode = 0
MAX_EPISODE = 10000

while episode < MAX_EPISODE:
    states, actions, rewards, score = gen_epsiode(episode)
    scores.append(score)

    loss1 = 0
    loss2 = 0
    for time_step, item in enumerate(zip(states, actions, rewards)):
        if time_step == len(states) - 1:
            break
        state  = item[0]
        action = item[1]
        reward = item[2]

        next_state = states[time_step+1]

        # G - V[state] => reward + gamma*V[next_state] - V[state]
        loss1 += ( reward + (gamma*V(next_state)) - V(state) )**2
        loss2 += math.pow(gamma, time_step)*(reward + (gamma*V(next_state)) - V(state)) * ((pi(state)[action] + 1e-5).log())


    loss1 = loss1/len(states)
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


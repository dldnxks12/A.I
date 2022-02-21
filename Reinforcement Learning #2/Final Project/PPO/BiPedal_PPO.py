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
    def __init__(self, DiscretizedActionRange):
        super(PPO, self).__init__()

        self.fc1 = nn.Linear(24, 256)
        self.fc2 = nn.Linear(256, 256)

        self.fc_pi_a11 = nn.Linear(256, 128)
        self.fc_pi_a12 = nn.Linear(128, DiscretizedActionRange)

        self.fc_pi_a21 = nn.Linear(256, 128)
        self.fc_pi_a22 = nn.Linear(128, DiscretizedActionRange)

        self.fc_pi_a31 = nn.Linear(256, 128)
        self.fc_pi_a32 = nn.Linear(128, DiscretizedActionRange)

        self.fc_pi_a41 = nn.Linear(256, 128)
        self.fc_pi_a42 = nn.Linear(128, DiscretizedActionRange)

        self.fc_v = nn.Linear(256, 1)

    # Actors (For 4 Joint)
    def pi_a1(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi_a11(x)
        x = self.fc_pi_a12(x)

        prob_a1 = F.softmax(x, dim=softmax_dim)
        return [x, prob_a1]

    def pi_a2(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi_a21(x)
        x = self.fc_pi_a22(x)

        prob_a2 = F.softmax(x, dim=softmax_dim)
        return [x, prob_a2]

    def pi_a3(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi_a31(x)
        x = self.fc_pi_a32(x)

        prob_a3 = F.softmax(x, dim=softmax_dim)
        return [x, prob_a3]

    def pi_a4(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi_a41(x)
        x = self.fc_pi_a42(x)

        prob_a4 = F.softmax(x, dim=softmax_dim)
        return [x, prob_a4]

    def PI(self, x, softmax_dim = 0):
      action1_prob = self.pi_a1(x, softmax_dim)
      action2_prob = self.pi_a2(x, softmax_dim)
      action3_prob = self.pi_a3(x, softmax_dim)
      action4_prob = self.pi_a4(x, softmax_dim)

      return action1_prob, action2_prob, action3_prob, action4_prob

    # Critic
    def v(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.fc_v(x)
        return v


def make_batch():
    states, actions, rewards, next_states, probs, dones = [],[],[],[],[],[]

    for transition in data:
        state, action, reward, next_state, prob, done = transition
        states.append(state)
        actions.append(action)
        rewards.append(torch.tensor(reward))
        next_states.append(next_state)
        probs.append(prob)
        done_mask = 0 if done else 1
        dones.append(done_mask)

    states      = torch.tensor(states, dtype=torch.float)
    actions     = torch.tensor(actions)
    rewards     = torch.tensor(rewards, dtype=torch.float)
    next_states = torch.tensor(next_states, dtype=torch.float)
    dones       = torch.tensor(dones, dtype=torch.float)
    probs       = torch.tensor(probs, dtype=torch.float)

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

        GAE_Value = torch.tensor(GAE_list, dtype = torch.float)

        ############################################# Ok ##################################
        ############################################# Restart From here ... ##################################
        old_a1,old_a2,old_a3,old_a4  = ppo.PI(states, softmax_dim = 1) # 현재 Action에 대한 Policy ...

        actions  = Discretization(old_a1,old_a2,old_a3,old_a4, Probs = False)

        print("Ok")

        sys.exit()
        PI_action_probs = PI_.gather(1, actions) # Action에 대한 Old 확률 가져와야한다.
        Ratio = torch.exp(torch.log(PI_action_probs) - torch.log(probs))

        Surrogate1 = Ratio * GAE_Value
        Surrogate2 = torch.clamp(Ratio, 1 - Eps_clip, 1 + Eps_clip) * GAE_Value

        loss = -torch.min(Surrogate1, Surrogate2) + F.smooth_l1_loss(ppo.v(states) - TD_Target.detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def Discretization(a1, a2, a3, a4, Probs = True):
  actions1, prob_a1 = a1
  actions2, prob_a2 = a2
  actions3, prob_a3 = a3
  actions4, prob_a4 = a4

  # SoftMax 분포에 따라 Action Sample (Return Index)
  action_index1 = Categorical(prob_a1).sample().item()
  action_index2 = Categorical(prob_a2).sample().item()
  action_index3 = Categorical(prob_a3).sample().item()
  action_index4 = Categorical(prob_a4).sample().item()

  # Continuous Action
  action1 = actions1[action_index1].cpu().detach().numpy()
  action2 = actions2[action_index2].cpu().detach().numpy()
  action3 = actions3[action_index3].cpu().detach().numpy()
  action4 = actions4[action_index4].cpu().detach().numpy()

  # Discretize (Return Index)
  discrete_action1 = np.array([np.digitize(action1, bins=A)])
  discrete_action2 = np.array([np.digitize(action2, bins=A)])
  discrete_action3 = np.array([np.digitize(action3, bins=A)])
  discrete_action4 = np.array([np.digitize(action4, bins=A)])

  # Discretized Action
  discrete_action1 = A[discrete_action1]
  discrete_action2 = A[discrete_action2]
  discrete_action3 = A[discrete_action3]
  discrete_action4 = A[discrete_action4]

  # Probabilities of Above Actions
  action1_prob = np.array([prob_a1[action_index1].cpu().detach().numpy()])
  action2_prob = np.array([prob_a2[action_index2].cpu().detach().numpy()])
  action3_prob = np.array([prob_a3[action_index3].cpu().detach().numpy()])
  action4_prob = np.array([prob_a4[action_index4].cpu().detach().numpy()])

  action = [discrete_action1, discrete_action2, discrete_action3, discrete_action4]
  action_prob = [action1_prob, action2_prob, action3_prob, action4_prob]

  if Probs == True:
    return action, action_prob

  else:
    return action


A = np.arange(-1, 1, 0.005)
env = gym.make('BipedalWalker-v3')
ppo = PPO(DiscretizedActionRange=len(A)).to(device)
optimizer = optim.Adam(ppo.parameters(), lr=lr)

score = 0.0
episode = 0
MAX_EPISODES = 3000
reward_history_10 = []
avg_history = []

while episode < MAX_EPISODES:
    state = env.reset()
    done = False
    data = []
    while not done:
        # T Step 동안 데이터 수집
        for t in range(T):
            a1, a2, a3, a4 = ppo.PI(torch.from_numpy(state).float().to(device))
            action, action_prob = Discretization(a1, a2, a3, a4)
            next_state, reward, done, _ = env.step(action)
            data.append((state, action, reward, next_state, action_prob, done))

            state = next_state
            score += reward

            if done:
                break

        train(ppo, optimizer)
        data = []

    # Moving Average Count
    reward_history_10.append(score)
    avg = reward_history_10[-10:].mean()
    avg_history.append(avg)
    if episode % 10 == 0:
        print('episode: {} | reward: {:.1f} | 10 avg: {:.1f} '.format(episode, score, avg))
    episode += 1

env.close()
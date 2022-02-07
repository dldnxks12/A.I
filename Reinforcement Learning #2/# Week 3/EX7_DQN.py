import random
import gym
import math
import random
import numpy as np
import torch
import collections
from collections import deque
from time import sleep
from IPython.display import clear_output

# 그림을 띄우고 뭘 하고 싶은데, 매번 그림을 띄우고 지우는 코드가 필요
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    return torch.tensor(states, dtype = torch.float), torch.tensor(actions, dtype = torch.float) , torch.tensor(rewards, dtype = torch.float), torch.tensor(next_states, dtype = torch.float), torch.tensor(dones, dtype = torch.float)

  def size(self):
    return len(self.buffer)

def update(net, net_target):
  for param_target, param in zip(net_target.parameters(), net.parameters()):
    param_target.data.copy_(param.data)
    
class PolicyNetwork(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.fcA1 = torch.nn.Linear(4, 10)
    self.fcA2 = torch.nn.Linear(10, 2)

  def forward(self, x):
    x = self.fcA1(x)
    x = torch.nn.functional.relu(x)
    x = self.fcA2(x)
    x = torch.nn.functional.softmax(x, dim = -1)
    return x # Return Policy  

class QNetwork(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.fcQ1 = torch.nn.Linear(4, 256)
    self.fcQ2 = torch.nn.Linear(256, 32)
    self.fcQ3 = torch.nn.Linear(32, 1)

  def forward(self, x):
    x = self.fcQ1(x)
    x = torch.nn.functional.relu(x)
    x = self.fcQ2(x)
    x = torch.nn.functional.relu(x)
    x = self.fcQ3(x)
    return x # Return Q value
  
 
memory = ReplayBuffer()
alpha = 0.001
gamma = 0.99

MAX_EPISODE = 10000
episode = 0 

pi = PolicyNetwork().to(device)
Q = QNetwork().to(device)
Q_target = QNetwork().to(device)

Q_target.load_state_dict(Q.state_dict()) # Synchronize Parameters

pi_optimizer = torch.optim.Adam(pi.parameters(), lr = alpha)
Q_optimizer = torch.optim.Adam(Q.parameters(), lr = alpha)

def train(memory, Q, Q_target, Q_optimizer):
  states, actions, rewards, next_states, dones = memory.sample(32)

  loss = 0 
  for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
    if done == 0:
      y = reward
    else:
      y = reward + gamma*(max(Q_target(next_state)))

    loss += (y - Q(state)[action]) ** 2

  loss = loss / 64

  Q_optimizer.zero_grad()
  loss.backward()
  Q_optimizer.step()
  
env = gym.make('CartPole-v1')
  
flag = False
while episode < MAX_EPISODE:

  state = env.reset()
  done = False
  score = 0 

  while not done:

    if episode % 10 == 0:
      flag == True
    else:
      flag == False

    #if episode % 100 == 0:
    #  env.render()

    state = np.array(state)
    policy = pi(torch.from_numpy(state).float().to(device))
    
    action = torch.multinomial(policy, 1).item()
    next_state, reward, done, info = env.step(action)

    memory.put((state, action, reward, next_state, done)) # Stack ... 

    score += reward
    state = next_state

    if memory.size() > 2000:
      train(memory, Q, Q_target, Q_optimizer)
      
      if flag == True:
        update(Q, Q_target)
        flag = False

    print(f"Epidoe : {episode} || Reward : {score}")
    episode += 1

env.close()

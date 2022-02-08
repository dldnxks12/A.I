'''

# 1)
plt.ion()  # 그때 그때 그림을 갱신하고 싶다.

# 2)
plt.ioff() # 그림에 관련된 모든 명령을 아래에서 다 실행한 후
 ~~
plt.show() # 최종적으로 그림을 한 번만 그린다.

'''
import gym
import sys
import math
import random
import numpy as np
import collections
from collections import deque
from time import sleep
from IPython.display import clear_output

from PIL import Image

import matplotlib.pyplot as plt
import matplotlib

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F

# 그림을 띄우고 뭘 하고 싶은데, 매번 그림을 띄우고 지우는 코드가 필요
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#is_ipython = 'inline' in matplotlib.get_backend()
#if is_ipython:
#  from IPython import display

#plt.ion()

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

class DQN(nn.Module):
  def __init__(self, w, h, outputs):
    super().__init__()
    # In Channel = 3 | Out Channel = 16
    self.conv1 = nn.Conv2d(3, 16, kernel_size = 5, stride = 2)
    self.bn1 = nn.BatchNorm2d(16)
    self.conv2 = nn.Conv2d(16, 32, kernel_size = 5, stride = 2)
    self.bn2 = nn.BatchNorm2d(32)
    self.conv3 = nn.Conv2d(32, 32, kernel_size = 5, stride = 2)
    self.bn3 = nn.BatchNorm2d(32)


    def conv2d_size_out(size, kernel_size = 5, stride = 2):
      return (size - (kernel_size - 1) - 1) // stride + 1 # Convolution Output Size

    # Convolution을 3번 거친 결과의 Width & Height - fc layer input 크기 계산하는 쉬운 방법 get
    convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
    convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
    linear_input_size = convw*convh*32
    self.head = nn.Linear(linear_input_size , outputs)

  def forward(self, x):
    x = x.to(device)
    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = F.relu(self.bn3(self.conv3(x)))
    x = self.head(x.view(x.size(0), -1)) # data flatten 후 fc layer에 넣어주기


resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

# Image Preprocessing ...
def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)

# Image Preprocessing ...
def get_screen():
    # Rendering Image 가져오기
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))

    # Channel 제외 height, width Get
    _, screen_height, screen_width = screen.shape
    # Channel 제외 Height, Width 조정
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]

    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)

    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)

    screen = screen[:, :, slice_range]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    return resize(screen).unsqueeze(0)

env = gym.make('CartPole-v1').unwrapped
env.reset()

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# gym 행동 공간에서 행동의 숫자를 얻습니다.
n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayBuffer()

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max (1)은 각 행의 가장 큰 열 값을 반환합니다.
            # 최대 결과의 두번째 열은 최대 요소의 주소값이므로,
            # 기대 보상이 더 큰 행동을 선택할 수 있습니다.
            print(policy_net(state))
            print(policy_net(state).max(1))
            print(policy_net(state).max(1)[1])
            print(policy_net(state).max(1)[1].view(1, 1))
            sys.exit()
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)





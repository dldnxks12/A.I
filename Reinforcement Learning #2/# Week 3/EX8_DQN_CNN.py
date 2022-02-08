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

env = gym.make('CartPole-v1').unwrapped

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
  from IPython import display

plt.ion()




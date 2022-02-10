import os
import gym
import sys
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from collections import deque
from IPython.display import clear_output
from skimage.color import rgb2gray
from skimage.transform import rescale

os.environ['KMP_DUPLICATE_LIB_OK'] ='True'

# CNN Network ...
class QNetwork(torch.nn.Module):
    # Input : 4 장의 사진 || Output : 4개의 Action
    def __init__(self, num_frames=4, num_actions=4):
        super(QNetwork, self).__init__()
        self.num_frames = num_frames

        # Layers
        self.conv1 = torch.nn.Conv2d(
            in_channels=self.num_frames,
            out_channels=16,
            kernel_size=8,
            stride=4,
            padding=2
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.fc1 = torch.nn.Linear(
            in_features=3200, # Atarin Game의 Input Size에 맞게 계산된 결과  ...
            out_features=256,
        )
        self.fc2 = torch.nn.Linear(
            in_features=256,
            out_features=num_actions,
        )

        # Activation Functions
        self.relu = torch.nn.ReLU()

    def flatten(self, x):
        batch_size = x.size()[0]
        x = x.view(batch_size, -1)
        return x

    def forward(self, x):
        # Forward pass
        x = self.relu(self.conv1(x))  # In: (80, 80, 4)  Out: (20, 20, 16)
        x = self.relu(self.conv2(x))  # In: (20, 20, 16) Out: (10, 10, 32)
        x = self.flatten(x)           # In: (10, 10, 32) Out: (3200,)
        x = self.relu(self.fc1(x))    # In: (3200,)      Out: (256,)
        x = self.fc2(x)               # In: (256,)       Out: (4,)

        return x

#######################################################################
# Env Setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("")
print(f"On {device}")
print("")

# env = gym.make("PongNoFrameskip-v4")
env = gym.make("Breakout-v0")

# network and optimizer
n_actions = env.action_space.n
Q = QNetwork().to(device)
optimizer = torch.optim.Adam(Q.parameters(), lr=0.0005)

# target network
Q_target = QNetwork().to(device)
Q_target.load_state_dict(Q.state_dict())

history = deque(maxlen=100000)  # replay buffer
discount = 0.99  # discount factor gamma
BATCH_SIZE = 16

# Replay Memory 에서 Batch Size만큼 transition뽑아와서 학습 시키기 --- Pixel 학습 ..!
def update_Q(Q, Q_target, optimizer):

    # Q : 4개 Frame 넣어주고, 4개 Action 받아오기

    batch = random.sample(history, BATCH_SIZE)
    print(batch.shape)
    sys.exit()
    loss = 0

    # Fill in this part


def process(state):
    state = rgb2gray(state[35:195, :, :])
    state = rescale(state, scale=0.5)
    state = state[np.newaxis, np.newaxis, :, :]
    state = state[0][0]
    return state


max_time_steps = 1000

# for computing average reward over 100 episodes
reward_history = deque(maxlen=100)

# for updating target network
target_interval = 1000
target_counter = 0

# training
for episode in range(2500):
    # sum of accumulated rewards
    rewards = 0

    # get initial observation
    observation = env.reset()
    state = process(observation)
    stack = [state] * 4     # state를 4개로 복제
    state = np.array(stack) # numpy type으로

    # On GPU + float + tensor type
    state = torch.from_numpy(state).float().to(device).unsqueeze(0)

    # loop until an episode ends
    for t in range(1, max_time_steps + 1):
        # env.render()

        # Epsilon greedy policy
        with torch.no_grad():
            if random.random() < 0.05:
                action = env.action_space.sample()
            else:
                # GET ACTION : Method 1
                # action = Q(state.float().to(device)).max(1)[1].view(1, 1).item()

                # GET ACTION : Method 2
                q_values = Q(state.to(device)).detach()
                action = torch.argmax(q_values)

        observation_next, reward, done, info = env.step(action)
        state_next = process(observation_next)

        # Frame 4개 들어있는 Stack에 Frame 하나 빼고, 새 걸로 하나 넣고!
        stack.pop(0)
        stack.append(state_next)

        # Type 변환  ...
        state_next = np.array(stack)
        state_next = torch.from_numpy(state_next).float().to(device).unsqueeze(0)

        rewards = rewards + reward

        # collect a transition - Replay 메모리
        history.append([state, action, state_next, reward, done])

        observation = observation_next
        state = state_next

        if len(history) > 2000:
            update_Q(Q, Q_target, optimizer)

            # Periodically Update target network
            target_counter = target_counter + 1
            if target_counter % target_interval == 0:
                Q_target.load_state_dict(Q.state_dict())

        if done:
            env.close()
            break

    # compute average reward
    reward_history.append(rewards)
    avg = sum(reward_history) / len(reward_history)
    print('episode: {}, reward: {:.1f}, avg: {:.1f}'.format(episode, rewards, avg))

env.close()

# Save Model
if not os.path.exists("./param"):
    os.makedirs("./param")
torch.save(Q.state_dict(), 'param/Q_net_params.pkl')





import numpy as np
import matplotlib.pyplot as plt
  
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F # for softmax 
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as dsets # MNIST dataset

train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

batch_size = 100

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# GoogleNet에서 가장 중요한 Incpetion Block 
class InceptionA(nn.Module):

  def __init__(self, in_channels):
    super(InceptionA, self).__init__()

    # 1x1 Conv - Dense to Sparse && Dimension reduction 

    # Block 1
    self.branch1_1 = nn.Conv2d(in_channels, 16, kernel_size = 1) # 1x1 Conv - Dense to Sparse , Dimension reduction 

    # Block 2
    self.branch3_1 = nn.Conv2d(in_channels, 16, kernel_size = 1) # 1x1 Conv 
    self.branch3_2 = nn.Conv2d(16, 24, kernel_size = 3, padding = 1)
    self.branch3_3 = nn.Conv2d(24, 24, kernel_size = 3, padding = 1)

    # Block 3
    self.branch5_1 = nn.Conv2d(in_channels, 16, kernel_size = 1) # 1x1 Conv 
    self.branch5_2 = nn.Conv2d(16, 24, kernel_size = 5, padding = 2)

    # Block 4
    self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size = 1)

  def forward(self, x):
    branch1x1 = self.branch1_1(x)

    branch3x3 = self.branch3_1(x)
    branch3x3 = self.branch3_2(branch3x3)
    branch3x3 = self.branch3_3(branch3x3)

    branch5x5 = self.branch5_1(x)
    branch5x5 = self.branch5_2(branch5x5)

    branch_pool = F.avg_pool2d(x, kernel_size = 3, stride = 1, padding = 1)
    branch_pool = self.branch_pool(branch_pool)

    # 4개의 output들을 1개의 list로
    outputs = [branch1x1, branch3x3, branch5x5, branch_pool]

    # torch.cat (concatenate)

    cat = torch.cat(outputs,  1) # outputs list의 tensor들을 dim = 1로 이어준다.

    return cat

# model
class GoogleNet(nn.Module):
  def __init__(self):
    super(GoogleNet, self).__init__()
    # Model의 앞 단은 그냥 Basic Conv layer로 구성

    self.conv1 = nn.Conv2d(1, 10, kernel_size = 5)
    self.conv2 = nn.Conv2d(88, 20, kernel_size = 5)

    self.incept1 = InceptionA(in_channels = 10)
    self.incept2 = InceptionA(in_channels = 20)

    self.mp = nn.MaxPool2d(kernel_size = 2)
    self.fc = nn.Linear(1408, 10)


  def forward(self, x):

    in_size = x.size(0) # 0 차원 크기 : batch size 

    x = self.conv1(x)
    x = F.relu(self.mp(x))
    x = self.incept1(x)

    x = self.conv2(x)
    x = F.relu(self.mp(x))
    x = self.incept2(x)

    x = x.view(in_size, -1) # flatten data
    
    x = self.fc(x) # batch x 10 

    return x
    
# model create    
model = GoogleNet()

# define optimizer
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5)

# train
for epoch in range(10):

  avg_cost = 0
  for x, y, in train_loader:

    optimizer.zero_grad()
    hypothesis = model(x)
    cost = F.cross_entropy(hypothesis, y)
    cost.backward()
    optimizer.step()

    avg_cost += cost
    print("Cost :", cost)

  print("Avg_cost :", avg_cost)

    

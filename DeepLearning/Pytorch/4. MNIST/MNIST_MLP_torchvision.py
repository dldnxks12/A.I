# MLP를 이용한 4. MNIST
# torchvision의 dataset을 이용해서 구현
# 앞선 MNIST_Classification은 Single Layer Perceptron으로 이해할 수 있다.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

# dataset for 4. MNIST dataset, transforms for data preprocessing
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import random # random seed for data shuffling
from torch.utils.data import DataLoader

# 1. GPU 설정 + seed 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("학습 장비 : ", device)

torch.manual_seed(777)
random.seed(777)
if device == 'cuda':
  torch.cuda.manual_seed_all(777)

# 2. 4. MNIST dataset 불러오기

mnist_train = dsets.MNIST(root = 'MNIST_data/', train = True, transform = transforms.ToTensor(), download = True )
mnist_test  = dsets.MNIST(root = 'MNIST_data/', train = False, transform = transforms.ToTensor(), download = True )

training_epochs = 15
batch_size = 100

dataloader = DataLoader(dataset = mnist_train, batch_size = batch_size, shuffle = True, drop_last = True)

# 3. Model 구성  
model = nn.Sequential(
      nn.Linear(28*28, 100, bias = True),
      nn.ReLU(),
      nn.Linear(100,50, bias = True),
      nn.ReLU(),
      nn.Linear(50,10, bias = True),
      nn.ReLU()
).to(device)

optimizer = optim.SGD(model.parameters(), lr = 0.1)
total_batch = len(dataloader)

for epoch in range(training_epochs):
  avg_cost = 0
  for x, y in dataloader: # dataloader 에서 batch 크기 만큼 데이터 뽑아옴 
    
    x_data = x.view(-1, 28*28).to(device)
    y_data = y.to(device)

    prediction = model(x_data).to(device)
    cost = F.cross_entropy(prediction, y_data) # prediction은 class 수만큼 나와야 함 --- 확률로 출력될것 

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    avg_cost += cost / total_batch

  print(epoch, cost, avg_cost)    

# ---- test ---- #

with torch.no_grad():

  X_test = mnist_test.test_data.view(-1,28*28).float().to(device)
  Y_test = mnist_test.test_labels.to(device)

  prediction = model(X_test)
  correction_prediction = torch.argmax(prediction, 1) == Y_test

  accuracy = correction_prediction.float().mean()

  r = random.randint(0, len(mnist_test)-1)

  x_single_test = mnist_test.test_data[r: r+1].view(-1, 28*28).float().to(device)
  y_single_test = mnist_test.test_labels[r: r+1].float().to(device)

  single_prediction = model(x_single_test)

  print("ACC", accuracy)
  print(torch.argmax(single_prediction, 1))

  plt.imshow(mnist_test.test_data[r:r+1].view(28,28), cmap = 'Greys', interpolation = 'nearest')
  plt.show()


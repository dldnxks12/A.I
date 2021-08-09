# 분석 필요

# 1. predicted.eq(targets/data/view_as(predicted)).sum()
# 2. torch.max(outputs.data, 1)
# 3. data_num = len(test_loader)    and    data_num = len(test_loader.dataset) 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# 데이터 가져오기

mnist = fetch_openml('mnist_784', version = 1, cache = True)

mnist.target = mnist.target.astype(np.int8)
x = mnist.data/255 # 0 ~ 1 사이로 정규화
y = mnist.target

# 훈련 데이터와 테스트 데이터 분리

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/7, random_state = 0)

# Pytorch 학습을 위해 data들 Tensor type으로 casting
x_train = torch.Tensor(x_train)
x_test = torch.Tensor(x_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# input , label dataset 뭉치로 만들기  ---- with TensorDataset 
ds_train = TensorDataset(x_train, y_train) 
ds_test = TensorDataset(x_test, y_test)

train_loader = DataLoader(ds_train, batch_size=64, shuffle = True)
test_loader = DataLoader(ds_test, batch_size=64, shuffle = True)


# MLP 구성
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
model = nn.Sequential(
    
    nn.Linear(28*28, 100 , bias = True),
    nn.ReLU(),
    nn.Linear(100, 100 , bias = True),
    nn.ReLU(),
    nn.Linear(100, 10 , bias = True),
)

optimizer = optim.Adam(model.parameters(), lr = 0.01)

# training 

def train(epoch):
  model.train() # model을 학습 모드로 

  for data, targets in train_loader:

    prediction = model(data)
    cost = F.cross_entropy(prediction, targets)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

  print(epoch)    

print("Finished")  
    
def test():

  model.eval()
  correct = 0


  with torch.no_grad():
    for data, targets in test_loader:

      prediction = model(data)
      
      _, predicted = torch.max(prediction.data , 1)  # 확률이 가장 높은 레이블 계산
      correct += predicted.eq(targets.data.view_as(predicted)).sum() # 정답과 일치한 경우 정답 카운트 증가 

  data_num = len(test_loader.dataset)
  print(correct, data_num, 100.0*correct/data_num)


for epoch in range(3):
  train(epoch)

test()

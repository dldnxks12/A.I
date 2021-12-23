import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml 

# 데이터 가져오기
mnist = fetch_openml('mnist_784', version = 1, cache = True)

# 데이터 전처리
x = mnist.data/255 # x data -- 0 ~ 1 사이로 정규화
mnist.target = mnist.target.astype(np.int8) # y data -- type casting
y = mnist.target

# 훈련 데이터와 테스트 데이터 분리
import torch
from torch.utils.data import DataLoader, TensorDataset # DataLoader for batch training, TensorDataset for aseemble x_data, y_data to Tensor dataset 
from sklearn.model_selection import train_test_split   # train 데이터와 test 데이터를 쪼개주는 sklearn module 

# train data = 60000 개 , test data = 10000 개로 split !
# 다음과 같이 x와 y의 데이터를 각각 훈련용, 테스트용 데이터로 노나준다.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/7, random_state = 0)

# Pytorch 학습을 위해 data들을 Tensor로 감싼다 or Tensor형으로 type casting
x_train = torch.Tensor(x_train)
x_test = torch.Tensor(x_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# 각 tensor data들을 DataLoader에 넣어주기 위해 dataset 뭉치로 만들기  (dataset = ~ 이 부분에 넣어줬는데, 사실 전에는 class 객체를 생성해서 넣어주었다. 아마 객체로 만들어서 넣어주는 과정인 것 같다.)
ds_train = TensorDataset(x_train, y_train) 
ds_test = TensorDataset(x_test, y_test)

# dataset으로 만든 Tensor를 이제 DataLoader에 넣어주자 --- Batch 학습을 위한 과정 
train_loader = DataLoader(ds_train, batch_size = 64, shuffle = True)
test_loader = DataLoader(ds_test, batch_size = 64, shuffle = True)

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

# Optimizer와 Cost function의 정의.
# Cost function은 따로 객체 생성안하고, loop에서 바로 호출해서 사용해도 상관없다.
optimizer = optim.Adagrad(model.parameters(), lr = 0.01)

# training function 
def train(epoch):
  model.train() # model을 학습 모드로 
  for data, targets in train_loader: # train_loader에서 Batch 크기만큼의 데이터 가져와서 학습 

    prediction = model(data)
    cost = F.cross_entropy(prediction, targets)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

  print("Epoch : {} Finished".format(epoch))  
    
def test():

  model.eval() # 모델을 평가 모드로 전환
  correct = 0 # 맞은 개수 

  with torch.no_grad(): # gradient 업데이트 수행 x
    
    for data, targets in test_loader: 
      prediction = model(data)      
      _, predicted = torch.max(prediction.data , 1)  # torch.max(input, dim)  -> (maxvalue, maxindex) type? (value => tensor, index => longtensor)
      correct += predicted.eq(targets.data.view_as(predicted)).sum() # 정답과 일치한 경우 정답 카운트 증가 
      # torch.ed -> 맞으면 True, 틀리면 False
      # torch.view_as -> size 맞춰서 

      # targets.data 를 predicted와 같은 size로 하고, 각각을 비교해서 맞으면 True, 틀리면 False -> 이거 True모두 Summation해서 correct에 +

  data_num = len(test_loader.dataset)
  print(correct/data_num, 100.0*correct/data_num)

for epoch in range(3):
  train(epoch)

test()

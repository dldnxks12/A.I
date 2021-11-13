import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Data Load
x_train = torch.FloatTensor([[1],[2],[3]]) # rank를 2로 유지학 위함
y_train = torch.FloatTensor([[1],[2],[3]]) # 1개 짜리 데이터 3개 임을 나타내기 위함  ---  [1,2,3] 으로하면 1개 짜리 3개인지, 3개짜리 1개인지 헷갈림

print(x_train.shape)
print(y_train.shape)
# Hypothesis & Cost

'''
Hypothesis H(x) = Wx + B 
Cost : 주로 MSE 사용
'''

model = nn.Linear(1,1) # linear module == Wx + B 내포
optimizer = optim.SGD(model.parameters(), lr = 0.01)


for epoch in range(1000):

    hypothesis = model(x_train)
    cost = F.mse_loss(hypothesis, y_train)

    # cost를 이용해 업데이트
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 99:
        params = list(model.parameters())
        weight = params[0]
        bias = params[1]

        print(f"Epoch : {epoch}/{1000} , Weight : {weight.item()}, Bias : {bias.item()}")

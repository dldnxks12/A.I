import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Data Load
dataset = np.loadtxt("data_linear_regression.csv", delimiter=',', dtype = np.float32)
np.random.shuffle(dataset)
print(dataset.shape) # 25, 4

x_train = torch.FloatTensor(dataset[:,:-1])
y_train = torch.FloatTensor(dataset[:,[-1]])

print(x_train.shape)
print(y_train.shape)

model = nn.Linear(3,1)
optimizer = optim.SGD(model.parameters(), lr = 0.00005)

print("Parameters of Linear model :",list(model.parameters()))

for epoch in range(10000):

    hypothesis = model(x_train)
    cost = F.mse_loss(hypothesis, y_train)

    # cost를 이용해 업데이트
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 99:

        print(f"Epoch : {epoch}/{5000} , Cost:{cost.item()}")


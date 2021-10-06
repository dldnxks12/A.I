# x의 개수가 1000개라면 ? x_train1 ~ x_train1000 그리고 w1~w1000 까지 모두 선언해야한다. 

# 이를 해결하기 위해 행렬 곱셈 (또는 벡터 내적)을 사용한다.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_train  =  torch.FloatTensor([[73,  80,  75], 
                               [93,  88,  93], 
                               [89,  91,  80], 
                               [96,  98,  100],   
                               [73,  66,  70]])  
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

print(x_train.shape)
print(y_train.shape)

W = torch.zeros((3,1), requires_grad=True)
b = torch.zeros((1), requires_grad=True) # bias는 알아서 casting 된다 5개로 복제되어 더해 질 것

optimizer = optim.SGD([W,b], lr = 1e-5)

num_epochs = 20

for epoch in range(num_epochs+1):

  hypothesis = x_train.matmul(W) + b
  cost = torch.mean((hypothesis-y_train)**2)

  optimizer.zero_grad()
  cost.backward()
  optimizer.step()

  print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
      epoch, num_epochs, hypothesis.squeeze().detach(), cost.item()
  ))


# hypothesis.squeeze() -- axis 제거 
# detach() 필요없는 부분 제거해서 출력되어 나옴 

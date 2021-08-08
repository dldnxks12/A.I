# low level로 구현한 sotmax regression 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]
y_train = [2, 2, 2, 1, 1, 1, 0, 0]
x_train = torch.FloatTensor(x_train) # size = (8, 4)
y_train = torch.LongTensor(y_train)  # size = (8,  )

# 3개의 class로 분류할 것 --- weight는   4x3이 되어야함
W = torch.zeros((4,3), requires_grad =True)
# 더하는 과정에서 알아서 multi casting 될 것  ( 1x1 -> 8x1로 !)
b = torch.zeros(1, requires_grad = True) 

# low level로 구현할 것 --- cost 함수 계산할 때 one-hot encoding된 y label 넣어주어야 함

y_onehot = torch.zeros((8,3))
y_train = y_train.unsqueeze(1)
y_onehot.scatter_(1, y_train, 1)

optimizer = optim.SGD([W,b], lr = 0.01)
num_epochs = 2000

for epoch in range(num_epochs + 1):
    
    hypothesis = F.softmax(x_train.matmul(W) + b , dim = 1)    
    cost = (y_onehot * -torch.log(hypothesis)).sum(dim = 1).mean()
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(epoch , cost)

        
'''
0 tensor(1.0986, grad_fn=<MeanBackward0>)
100 tensor(0.8508, grad_fn=<MeanBackward0>)
200 tensor(0.7849, grad_fn=<MeanBackward0>)
300 tensor(0.7446, grad_fn=<MeanBackward0>)
400 tensor(0.7146, grad_fn=<MeanBackward0>)
500 tensor(0.6907, grad_fn=<MeanBackward0>)
600 tensor(0.6708, grad_fn=<MeanBackward0>)
700 tensor(0.6538, grad_fn=<MeanBackward0>)
800 tensor(0.6391, grad_fn=<MeanBackward0>)
900 tensor(0.6262, grad_fn=<MeanBackward0>)
1000 tensor(0.6146, grad_fn=<MeanBackward0>)
1100 tensor(0.6042, grad_fn=<MeanBackward0>)
1200 tensor(0.5948, grad_fn=<MeanBackward0>)
1300 tensor(0.5861, grad_fn=<MeanBackward0>)
1400 tensor(0.5781, grad_fn=<MeanBackward0>)
1500 tensor(0.5707, grad_fn=<MeanBackward0>)
1600 tensor(0.5637, grad_fn=<MeanBackward0>)
1700 tensor(0.5572, grad_fn=<MeanBackward0>)
1800 tensor(0.5511, grad_fn=<MeanBackward0>)
1900 tensor(0.5453, grad_fn=<MeanBackward0>)
2000 tensor(0.5398, grad_fn=<MeanBackward0>)
'''

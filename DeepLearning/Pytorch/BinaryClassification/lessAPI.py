# 최대한 numpy style로 구현

# torch.sigmoid, torch.nn.binary_cross_entropy(예측 값, 실제 값) 사용 x 

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)


# pytorch에서 변수들을 학습하기 위해서는 모두 Tensor type으로 바꾸어주어야한다!

x_train = torch.FloatTensor( [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]] ) # 6 x 2 Tensor
y_train = torch.FloatTensor( [[0],[0],[0],[1],[1],[1]] )# 6 x 1 Tensor 

W = torch.zeros((2,1), requires_grad = True) # 2 x 1 Tensor  --- (6x2) matmul (2 x 1) --- 6x1 output 
b = torch.zeros(1, requires_grad = True)


# 가설식 구현 1 : hypothesis

hypothesis1 = 1 / ( 1 + torch.exp(-(x_train.matmul(W) + b)))

# 가설식 구현 2 

hypothesis2 = torch.sigmoid(x_train.matmul(W) + b)



'''

# 예측 값
print(hypothesis1) 

# 실제 값
print(y_train)

tensor([[0.5000],
        [0.5000],
        [0.5000],
        [0.5000],
        [0.5000],
        [0.5000]], grad_fn=<MulBackward0>)
tensor([[0.],
        [0.],
        [0.],
        [1.],
        [1.],
        [1.]])

'''

# 첫 번째 Sample에 대해서 현재 예측 값과 실제 값에 대한 BinaryCrossEntropy loss 구해보기

cost1 = -(y_train[0]*torch.log(hypothesis1[0]) + (1-y_train[0])*torch.log(1-hypothesis[0]))

print(cost1.item()) # tensor([0.6931], grad_fn=<NegBackward>) --- 0.6931471824645996

# 모든 Sample에 대해 BinaryCrossEntropy loss 구하기

losses = -(y_train*torch.log(hypothesis1) + (1-y_train)*torch.log(1-hypothesis))

print(losses)

# 이제 이 값들을 모두 더한 후 평균 내기

cost = torch.mean(losses)

print(cost)

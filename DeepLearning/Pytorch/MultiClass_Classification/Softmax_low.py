'''

* CrossEntropy 직접 구현 

새로 사용한 API

F.softmax()
tensor.unsqueeze() --- 차원 늘리기
tensor.scatter_(dim, y , y index 자리에 넣을 값)  --- One-hot Encoding에 사용 

'''

import torch
import torch.nn.functional as F

z = torch.rand((3,5), requires_grad=True) # 3 x 5 random value tensor define
hypothesis = F.softmax(z, dim = 1) # 각 x_data에 대해서 softmax를 진행하므로 dim = 1로 설정

# 이제 위 hypothesis와 label을 사용해서 cost를 계산한다 --- onehot 인코딩된 label이 필요!

# print(y) : tensor([4, 0, 1])
y = torch.randint(5, (3,)).long()  # long type으로 0~5까지 수 중에서 3개 3, size로 
y = y.unsqueeze(1)

# 사실 y = torch.randint(5, (3,1)).long()으로 해도 상관 x

'''

print(y)

tensor([3,4,3])  --- tensor( [[3],[4],[3]] )
shape : (3, ) -> (3 x 1) 
        
'''

# one-hot encoding 
y_onehot = torch.zeros_like(hypothesis)  # hypothesis와 크기가 같은 0으로 초기화된 tensor

'''
print(y_onehot)

tensor([[0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.]])
'''

# y_onehot

y_onehot.scatter_(1, y, 1) # (dim, 해당하는 y, 해당하는 위치에 1을 넣어라!)

'''
print(y_onehot)

y_onehot.scatter_(1, y, 1)

tensor([[0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 1.],
        [0., 1., 0., 0., 0.]])

y_onehot.scatter_(1, y, 2) 라면?

tensor([[0., 2., 0., 0., 0.],
        [0., 0., 0., 0., 2.],
        [0., 2., 0., 0., 0.]])
'''

# 이제 onehot 인코딩된 label과 softmax 결과를 이용해서 cost (loss)를 계산 -- Cross Entropy

cost = (y_onehot * -torch.log(hypothesis)).sum(dim = 1).mean()

print(cost)

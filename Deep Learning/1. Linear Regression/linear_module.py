'''

Pytorch에서 선형 회귀 모델은 nn.1. Linear Regression() 라는 함수로 구현되어있다.
또한 MSE는 nn.functional.mse_loss() 함수로 되어있다. 

model = nn.1. Linear Regression(input_dim, output_dim)
cost = F.mse_loss(prediction, y_train)

'''


import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.manual_seed(1)

# 데이터 , W는 2가 되고, Bias는 0이 되어야 알맞는 결과 
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

model = nn.Linear(1,1)  # 입력 차원 1, 출력 차원 1

# model에는 가중치 w와 편향 b가 선언되어있다. --- model.parameters() 함수로 불러올 수 있다.

'''
print(list(model.parameters()))

[Parameter containing:
tensor([[0.5153]], requires_grad=True), Parameter containing:
tensor([-0.4414], requires_grad=True)]

'''

# optimizer를 정의하는데, 학습할 변수를 model.parameters()로 넘겨줄 수 있다.

optimizer = optim.SGD(model.parameters(), lr = 0.01)

num_epochs = 1000
for epoch in range(num_epochs + 1):

  prediction = model(x_train) # hypothesis 계산  --- forward 연산 

  cost = F.mse_loss(prediction, y_train) 

  optimizer.zero_grad()
  cost.backward()
  optimizer.step()

  if epoch % 100 == 0:
  # 100번마다 로그 출력
    print('Epoch {:4d}/{} Cost: {:.6f}'.format(
        epoch, num_epochs, cost.item()
    ))

print(list(model.parameters()))

'''
Epoch    0/1000 Cost: 13.103541
Epoch  100/1000 Cost: 0.002791
Epoch  200/1000 Cost: 0.001724
Epoch  300/1000 Cost: 0.001066
Epoch  400/1000 Cost: 0.000658
Epoch  500/1000 Cost: 0.000407
Epoch  600/1000 Cost: 0.000251
Epoch  700/1000 Cost: 0.000155
Epoch  800/1000 Cost: 0.000096
Epoch  900/1000 Cost: 0.000059
Epoch 1000/1000 Cost: 0.000037
[Parameter containing:
tensor([[1.9930]], requires_grad=True), Parameter containing:
tensor([0.0159], requires_grad=True)]

'''

# 학습이 잘 되었는지 test

test_var = torch.FloatTensor([[4.0]]) # 1x1 value
pred_y = model(test_var)
print(pred_y) # weight = 2, bias = 0 이었으므로 pred_y 는 8에 가깝게 나와야한다. 

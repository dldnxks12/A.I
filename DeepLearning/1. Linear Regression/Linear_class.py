# 클래스로 구현하기 - 대부분의 pytorch 모델을 클래스로 구현해서 사용한다.


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LinearRegression(nn.Module): # torch.nn.module을 상속받아서 사용 
  def __init__(self):
    super().__init__() # nn.Module 클래스의 속성을 가지고 초기화 한다.
    self.linear = nn.Linear(1,1) # nn.module에 정의된 forward 연산 

  def forward(self, x): # model 객체를 데이터와 함께 호출하면 자동으로 실행된다. 
    return self.linear(x)

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

model = LinearRegression()    
optimizer = optim.SGD(model.parameters(), lr = 0.01)

num_epochs = 1000
for epoch in range(num_epochs + 1):
  # hypothesis (forward 연산)

  prediction = model(x_train)

  cost = F.mse_loss(prediction, y_train)

  optimizer.zero_grad()
  cost.backward()
  optimzer.step()

  if epoch % 100 == 0:
  # 100번마다 로그 출력
    print('Epoch {:4d}/{} Cost: {:.6f}'.format(
        epoch, num_epochs, cost.item()
    ))


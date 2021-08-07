import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim # optimizer 

torch.manual_seed(1) # 현재 실습한 코드를 재실행해도 같은 결과가 나오도록

# 모델을 학습 시키기 위한 데이터는 파이토치의 텐서 형태를 가지고 있어야한다. (torch.tensor)

x_train = torch.FloatTensor([ [1] ,[2], [3] ])
y_train = torch.FloatTensor([ [2] ,[4], [6] ])


# 선형 회귀의 목표는 가장 잘 맞는 직선을 정의하는 weight , bias를 찾는 것 

# 먼저 weight를 0으로 초기화 

W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# requires_grad는 이 변수가 학습을 통해 계속 값이 변경되는 변수임을 의미한다

# 현재 y = (0 * x) + 0

optimizer = optim.SGD( [W, b], lr = 0.01 ) # 경사 하강법 학습 대상 - weight, bias 

num_epochs = 999 # 원하는 만큼 경사 하강법 반복

for epoch in range(num_epochs + 1): 
  
  hypothesis = x_train*W + b # 가설 
  cost = torch.mean( (hypothesis - y_train)**2 ) # loss -- MSE 

# optimizer.zero_grad()를 통해 미분을 통해 얻은 기울기를 0으로 초기화한다.
# 이렇게 초기화 해야만 새로운 weight, bias에 대해서 또 새로운 기울기를 구할 수 있다.
# 이 다음 cost.backward() 를 호출해서 새 weight, bias가 계산된다. 
# 이 다음 optimizer.step() 을 호출해서 인수로 들어갔던 W, b 를 새 기울기 x learning rate를 빼서 업데이트 한다.

  optimizer.zero_grad()
  cost.backward()
  optimizer.step()

  if epoch % 100 == 0:
      print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
          epoch, num_epochs, W.item(), b.item(), cost.item()
      ))

print(W) # tensor([1.9708], requires_grad=True)
print(W.item()) # 1.9707825183868408


'''
Epoch    0/999 W: 0.187, b: 0.080 Cost: 18.666666
Epoch  100/999 W: 1.746, b: 0.578 Cost: 0.048171
Epoch  200/999 W: 1.800, b: 0.454 Cost: 0.029767
Epoch  300/999 W: 1.843, b: 0.357 Cost: 0.018394
Epoch  400/999 W: 1.876, b: 0.281 Cost: 0.011366
Epoch  500/999 W: 1.903, b: 0.221 Cost: 0.007024
Epoch  600/999 W: 1.924, b: 0.174 Cost: 0.004340
Epoch  700/999 W: 1.940, b: 0.136 Cost: 0.002682
Epoch  800/999 W: 1.953, b: 0.107 Cost: 0.001657
Epoch  900/999 W: 1.963, b: 0.084 Cost: 0.001024
'''

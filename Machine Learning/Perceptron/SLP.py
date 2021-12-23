import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# GPU Setting
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
    
# 변수 선언
x_data = torch.FloatTensor([[0,0],[0,1],[1,0],[1,1]]).to(device) # defualt = CPU
y_data = torch.FloatTensor([[0],[1],[1],[0]]).to(device)         # defualt = CPU

#1 개의 layer를 가진 SLP ( sigmoid( y^ = Wx+b) )
linear = nn.Linear(2, 1, bias = True) # bias = True는 default 지만 걍 선언해줌
sigmoid = nn.Sigmoid()

model = nn.Sequential(
        linear,
        sigmoid
).to(device)
    
# Optimizer 선언

optimizer = optim.SGD(model.parameters(), lr = 1)
for step in range(1000):
    
    hypothesis = model(x_data)
    cost = F.binary_cross_entropy(hypothesis, y_data)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if step % 100 == 0:
        print(step, cost)
    
'''

!!!
cost가 0.6931 아래로 줄어들지 않는 것을 확인할 수 있다. 
이는 SLP가 XOR 문제를 해결하지 못하는 것을 보여준다.

0 tensor(0.7274, grad_fn=<BinaryCrossEntropyBackward>)
100 tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)
200 tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)
300 tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)
400 tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)
500 tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)
600 tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)
700 tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)
800 tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)
900 tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)

'''    

with torch.no_grad():
    hypothesis = model(x_data)
    prediction = (hypothesis > 0.5).float() # sigmoid activation function 이기에 0.5 위 아래로 나뉨
    acc = (prediction == y_data).float().mean()
    
    print(y_data) 
    print(prediction)
    
'''

정답 0 1 1 10

tensor([[0.],
        [1.],
        [1.],
        [0.]])
        
예측 값 0 0 0 0

tensor([[0.],
        [0.],
        [0.],
        [0.]])
'''    

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


model = nn.Sequential(
    nn.Linear(2, 10, bias = True),
    nn.Sigmoid(),
    nn.Linear(10,10, bias = True),
    nn.Sigmoid(),
    nn.Linear(10,10, bias = True),
    nn.Sigmoid(),
    nn.Linear(10,1, bias = True),
    nn.Sigmoid()
).to(device)
    
# Optimizer 선언
optimizer = optim.SGD(model.parameters(), lr = 1)
for step in range(10000):
    
    hypothesis = model(x_data)
    cost = F.binary_cross_entropy(hypothesis, y_data)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if step % 100 == 0:
        print(step, cost)
    


with torch.no_grad():
    hypothesis = model(x_data)
    prediction = (hypothesis > 0.5).float() # sigmoid activation function 이기에 0.5 위 아래로 나뉨
    acc = (prediction == y_data).float().mean()
    
    print(y_data) 
    print(prediction)
    
'''
0 tensor(0.6949, grad_fn=<BinaryCrossEntropyBackward>)
100 tensor(0.6932, grad_fn=<BinaryCrossEntropyBackward>)
200 tensor(0.6932, grad_fn=<BinaryCrossEntropyBackward>)
300 tensor(0.6932, grad_fn=<BinaryCrossEntropyBackward>)
400 tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)
500 tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)
600 tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)
700 tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)
800 tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)
900 tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)
1000 tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)
1100 tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)
1200 tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)
1300 tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)
1400 tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)
1500 tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)
1600 tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)
1700 tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)
1800 tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)
1900 tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)
2000 tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)
2100 tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)
2200 tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)
2300 tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)
2400 tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)
2500 tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)
2600 tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)
2700 tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)
2800 tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)
2900 tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)
3000 tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)
3100 tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)
3200 tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)
3300 tensor(0.6931, grad_fn=<BinaryCrossEntropyBackward>)
3400 tensor(0.6930, grad_fn=<BinaryCrossEntropyBackward>)
3500 tensor(0.6930, grad_fn=<BinaryCrossEntropyBackward>)
3600 tensor(0.6930, grad_fn=<BinaryCrossEntropyBackward>)
3700 tensor(0.6930, grad_fn=<BinaryCrossEntropyBackward>)
3800 tensor(0.6930, grad_fn=<BinaryCrossEntropyBackward>)
3900 tensor(0.6929, grad_fn=<BinaryCrossEntropyBackward>)
4000 tensor(0.6929, grad_fn=<BinaryCrossEntropyBackward>)
4100 tensor(0.6929, grad_fn=<BinaryCrossEntropyBackward>)
4200 tensor(0.6928, grad_fn=<BinaryCrossEntropyBackward>)
4300 tensor(0.6927, grad_fn=<BinaryCrossEntropyBackward>)
4400 tensor(0.6926, grad_fn=<BinaryCrossEntropyBackward>)
4500 tensor(0.6924, grad_fn=<BinaryCrossEntropyBackward>)
4600 tensor(0.6921, grad_fn=<BinaryCrossEntropyBackward>)
4700 tensor(0.6917, grad_fn=<BinaryCrossEntropyBackward>)
4800 tensor(0.6907, grad_fn=<BinaryCrossEntropyBackward>)
4900 tensor(0.6886, grad_fn=<BinaryCrossEntropyBackward>)
5000 tensor(0.6821, grad_fn=<BinaryCrossEntropyBackward>)
5100 tensor(0.6473, grad_fn=<BinaryCrossEntropyBackward>)
5200 tensor(0.4499, grad_fn=<BinaryCrossEntropyBackward>)
5300 tensor(0.0414, grad_fn=<BinaryCrossEntropyBackward>)
5400 tensor(0.0097, grad_fn=<BinaryCrossEntropyBackward>)
5500 tensor(0.0050, grad_fn=<BinaryCrossEntropyBackward>)
5600 tensor(0.0033, grad_fn=<BinaryCrossEntropyBackward>)
5700 tensor(0.0024, grad_fn=<BinaryCrossEntropyBackward>)
5800 tensor(0.0019, grad_fn=<BinaryCrossEntropyBackward>)
5900 tensor(0.0015, grad_fn=<BinaryCrossEntropyBackward>)
6000 tensor(0.0013, grad_fn=<BinaryCrossEntropyBackward>)
6100 tensor(0.0011, grad_fn=<BinaryCrossEntropyBackward>)
6200 tensor(0.0010, grad_fn=<BinaryCrossEntropyBackward>)
6300 tensor(0.0009, grad_fn=<BinaryCrossEntropyBackward>)
6400 tensor(0.0008, grad_fn=<BinaryCrossEntropyBackward>)
6500 tensor(0.0007, grad_fn=<BinaryCrossEntropyBackward>)
6600 tensor(0.0007, grad_fn=<BinaryCrossEntropyBackward>)
6700 tensor(0.0006, grad_fn=<BinaryCrossEntropyBackward>)
6800 tensor(0.0006, grad_fn=<BinaryCrossEntropyBackward>)
6900 tensor(0.0005, grad_fn=<BinaryCrossEntropyBackward>)
7000 tensor(0.0005, grad_fn=<BinaryCrossEntropyBackward>)
7100 tensor(0.0005, grad_fn=<BinaryCrossEntropyBackward>)
7200 tensor(0.0004, grad_fn=<BinaryCrossEntropyBackward>)
7300 tensor(0.0004, grad_fn=<BinaryCrossEntropyBackward>)
7400 tensor(0.0004, grad_fn=<BinaryCrossEntropyBackward>)
7500 tensor(0.0004, grad_fn=<BinaryCrossEntropyBackward>)
7600 tensor(0.0003, grad_fn=<BinaryCrossEntropyBackward>)
7700 tensor(0.0003, grad_fn=<BinaryCrossEntropyBackward>)
7800 tensor(0.0003, grad_fn=<BinaryCrossEntropyBackward>)
7900 tensor(0.0003, grad_fn=<BinaryCrossEntropyBackward>)
8000 tensor(0.0003, grad_fn=<BinaryCrossEntropyBackward>)
8100 tensor(0.0003, grad_fn=<BinaryCrossEntropyBackward>)
8200 tensor(0.0003, grad_fn=<BinaryCrossEntropyBackward>)
8300 tensor(0.0003, grad_fn=<BinaryCrossEntropyBackward>)
8400 tensor(0.0002, grad_fn=<BinaryCrossEntropyBackward>)
8500 tensor(0.0002, grad_fn=<BinaryCrossEntropyBackward>)
8600 tensor(0.0002, grad_fn=<BinaryCrossEntropyBackward>)
8700 tensor(0.0002, grad_fn=<BinaryCrossEntropyBackward>)
8800 tensor(0.0002, grad_fn=<BinaryCrossEntropyBackward>)
8900 tensor(0.0002, grad_fn=<BinaryCrossEntropyBackward>)
9000 tensor(0.0002, grad_fn=<BinaryCrossEntropyBackward>)
9100 tensor(0.0002, grad_fn=<BinaryCrossEntropyBackward>)
9200 tensor(0.0002, grad_fn=<BinaryCrossEntropyBackward>)
9300 tensor(0.0002, grad_fn=<BinaryCrossEntropyBackward>)
9400 tensor(0.0002, grad_fn=<BinaryCrossEntropyBackward>)
9500 tensor(0.0002, grad_fn=<BinaryCrossEntropyBackward>)
9600 tensor(0.0002, grad_fn=<BinaryCrossEntropyBackward>)
9700 tensor(0.0002, grad_fn=<BinaryCrossEntropyBackward>)
9800 tensor(0.0002, grad_fn=<BinaryCrossEntropyBackward>)
9900 tensor(0.0002, grad_fn=<BinaryCrossEntropyBackward>)


Label

tensor([[0.],
        [1.],
        [1.],
        [0.]])
        

Prediction

tensor([[0.],
        [1.],
        [1.],
        [0.]])
'''

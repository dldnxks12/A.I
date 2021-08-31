# class로 구현 --- 일반적으로 사용되는 form이니 잘 기억해두고 사용하기

'''
class로 구현에 앞서 model은 다음과 같이 구현했었다.

model = nn.Sequential(

    nn.Linear(2,1),
    nn.Sigmoid()

)

나머지는 Binary_module.py 코드와 동일하다.
'''

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class BinaryClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2,1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        return self.sigmoid(self.linear(x))

torch.manual_seed(1)

x_train = torch.FloatTensor( [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]] ) # 6 x 2 Tensor
y_train = torch.FloatTensor( [[0],[0],[0],[1],[1],[1]] )# 6 x 1 Tensor 

model = BinaryClassification()

optimizer = optim.SGD(model.parameters(), lr = 1)

num_epochs = 1000
for epoch in range(num_epochs + 1):
    pred = model(x_train)
    cost = F.binary_cross_entropy(pred, y_train)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        prediction = pred >= torch.FloatTensor([0.5]) # 0.5 이상이면 True로 
        
        correct_prediction = prediction.float() == y_train # 실제 값과 일치하는 경우에 True
        
        acc = correct_prediction.sum().item() / len(correct_prediction) 
        
        print(epoch, cost.item(), acc*100)
    

new_var = torch.FloatTensor([[3,4]])
pred = model(new_var)

print(pred) # pred 는 hypothesis    
    
    

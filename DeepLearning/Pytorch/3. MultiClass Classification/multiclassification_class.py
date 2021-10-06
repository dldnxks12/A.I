# multi class classification을 class로 구현 

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class multiclass(nn.Module): # nn.Module 상속 -> init필요
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4,3) 
        
    def forward(self,x):
        return self.linear(x)
        
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

model = multiclass()
optimizer = optim.SGD(model.parameters(), lr = 0.1)

num_epochs = 1000
for epoch in range(num_epochs+1):
    z = model(x_train)
    cost = F.cross_entropy(z, y_train)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, num_epochs, cost.item()
        ))    

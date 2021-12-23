import numpy as np
import torch
from torch.autograd import Variable

Data = np.loadtxt("diabetes.csv", delimiter= ',')
x_data = Variable(torch.Tensor( Data[:,:-1]))
y_data = Variable(torch.Tensor( Data[:,[-1]])) 

# Model Design

class Model(torch.nn.Module):
    # Model 초기설정
    def __init__(self):
        super(Model,self).__init__()
        self.linear = torch.nn.Linear(8,1) # 8 input -> 1 output
        
    # Forward Pass
    def forward(self, x):
        y_pred = torch.nn.functional.sigmoid(self.linear(x))
        return y_pred
    
model = Model()  # Model Class 객체 생성

# Loss & Optimizer setting

criterian = torch.nn.BCELoss(reduction = 'mean') 
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

for epoch in range(1001):
    y_pred = model(x_data)
    
    loss = criterian(y_pred,y_data)
    print("epoch: ", epoch, "loss: ", loss.item())
    
    optimizer.zero_grad() 
    loss.backward()
    optimizer.step()  # Update



        
        
    

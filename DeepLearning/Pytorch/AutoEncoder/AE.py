# Stacked Auto Encoder

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init            # for weight , bias initialization
import torch.nn.functional as F
import torchvision.datasets as dsets # for MNIST dataset 
import torchvision.transforms as transforms # for tensor transforms 

from torch.utils.data import DataLoader # for batch learning

# others 
import numpy as np
import matplotlib.pyplot as plt

# MNIST DATASET
mnist_train = dsets.MNIST(root = "MNIST_DATA/", train = True, transform = transforms.ToTensor(), download = True)
mnist_test = dsets.MNIST(root = "MNIST_DATA/", train = False, transform = transforms.ToTensor(), download = True)

# Set hyperparameter

batch_size = 100
learning_rate = 0.1
training_epochs = 10

# Set Train Loader
Train_loader = DataLoader(dataset = mnist_train, batch_size = batch_size, shuffle = True, drop_last = True)
Test_loader = DataLoader(dataset = mnist_test, batch_size = batch_size, shuffle = False, drop_last = True)

# GPU & CPU Setting

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
    
print("사용 Device : ", device)    

# Model Architecture with class

class AE(nn.Module):
    def __init__(self):
        super(AE,self).__init__() # initialize super class init function
        self.encoder = nn.Linear(28*28, 20) # 28 x 28 vector -> 20 vector (latent space vector)
        self.decoder = nn.Linear(20, 28*28) # 20 -> 28 x 28 (reconstruct image feature from latent space vector)
        
    def forward(self, x):
        x = x.view(batch_size, -1) # (batch_size , channel, height , width ) -> (batch_size, channel x height x width)
        z = self.encoder(x)
        out = self.decoder(z)     # (batch_size, channel x height x width) -> (batch_size, channel, height , width )  
        out = out.view(batch_size, 1, 28 , 28) # 이미지의 형태로 reconstruct        
        return out

''' layer 2배로 
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__() # initialize super class init function
        self.encoder = nn.Linear(28*28, 100) # 28 x 28 vector -> 20 vector (latent space vector)
        self.encoder2 = nn.Linear(100, 10) # 28 x 28 vector -> 20 vector (latent space vector)        
        
        self.decoder = nn.Linear(10, 100) # 20 -> 28 x 28 (reconstruct image feature from latent space vector)
        self.decoder2 = nn.Linear(100, 28*28) # 20 -> 28 x 28 (reconstruct image feature from latent space vector)
        
    def forward(self, x): # x : 100 x 1 x 28 x 28 
        x = x.view(batch_size, -1) # (batch_size , channel, height , width ) -> (batch_size, channel x height x width)
        z = self.encoder(x)
        z = self.encoder2(z)
        out = self.decoder(z)     # (batch_size, channel x height x width) -> (batch_size, channel, height , width )  
        out = self.decoder2(out)
        out = out.view(batch_size, 1, 28 , 28) # 이미지의 형태로 reconstruct        
        return out
'''
    
# model 생성 , optimizer, cost function 

model = AE().to(device)
optimizer = optim.SGD(model.parameters(), lr = learning_rate)
loss_fun = nn.MSELoss() # Mean Squared Error 사용 (원래 이미지와 출력 이미지 간의 loss 단순 계산)

# training 
for i in range(training_epochs):  
    loss = 0
    for x, y in Train_loader:
        
        x = x.to(device)
        y = y.to(device)         
        optimizer.zero_grad() # 초기화
        
        # Inference
        out = model(x)  # out --- (batch_size, channel, height , width )  의 reconstruct된 data        
        cost = loss_fun(out,x)
        
        cost.backward() # 가중치 계산
        optimizer.step()
        
        loss = cost            
    print(loss)
                
# Test Input - Output

# out.cpu() --- GPU에서 Tensor를 가져올 떄 사용, CPU로 학습을 진행했다면 cpu() 사용안해도 상관 x

plt.figure(figsize = (8,8))
plt.subplot(1,2,1)
plt.imshow(torch.squeeze(x.data[10]))
plt.subplot(1,2,2)
plt.imshow(torch.squeeze(out.data[10]))
plt.show()

        

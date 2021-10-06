import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F # for softmax 
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as dsets # 4. MNIST dataset

class VGG_ORG(nn.Module):
    def __init__(self, in_channel): 
        super().__init__()
        
        self.layer1 = nn.Sequential(            
        nn.Conv2d(in_channel, 64, kernel_size = 3, stride = 1, padding = 1),    
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2)            
        )

        self.layer2= nn.Sequential(            
        nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),    
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2)            
        )
        
        self.layer3 = nn.Sequential(            
        nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),    
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2)            
        )
        
        self.layer4 = nn.Sequential(            
        nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1),    
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),    
        nn.ReLU(),            
        nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2)            
        )        
        
        self.layer5 = nn.Sequential(            
        nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),    
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),    
        nn.ReLU(),            
        nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2)            
        )        
        
        # batch x 512 x 7 x 7
        self.fc1 = nn.Linear(512*7*7, 4096), # 512 자리에 512 x width x heigth 넣어주기 
        self.fc2 = nn.Linear(4096, 4096),
        self.fc3 = nn.Linear(4096, 1000)

    def forward(self, x):
        
        # Conv layer
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        
        # fc layer    
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out) # output : channel x 1000 

        return out
    
class VGG_MNIST(nn.Module):
    def __init__(self, in_channel): # in_channel : 1 , size = BatchSize x 1 x 28 x 28
        super().__init__()

    
        # batchsize x 1 x 28 x 28 
        self.layer1 = nn.Sequential(            
        nn.Conv2d(in_channel, 64, kernel_size = 3, stride = 1, padding = 1),    
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2)            
        )
    
        # batchsize x 64 x 14 x 14 
        self.layer2 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1),        
        nn.BatchNorm2d(128),                        
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2),
        )

        # batchsize x 128 x 7 x 7         
        self.fc1 = nn.Linear(7*7*128, 4096)
        self.fc2 = nn.Linear(4096, 1000)
        self.fc3 = nn.Linear(1000, 10)
        
    def forward(self, x):
        
        out = self.layer1(x)
        out = self.layer2(out)
        
        # fc layer
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out) # output : batchsize x 10

        return out    
        
        
x_train = dsets.MNIST(root = "MNIST_DATA/", train = True, transform = transforms.ToTensor(), download = True)
y_train = dsets.MNIST(root = "MNIST_DATA/", train = False, transform = transforms.ToTensor(), download = True)

train_loader = DataLoader(dataset = x_train, batch_size = 100, shuffle = True, drop_last = True)

learning_rate = 0.01
training_epochs = 10

model = VGG_MNIST(1)
optimizer = optim.SGD(model.parameters(), lr = learning_rate)

for epoch in range(training_epochs+1):
    avg_cost = 0
    for x,y in train_loader:
        
        print(x.shape)
        print(y.shape)
        
        optimizer.zero_grad()
        out = model(x)
        
        cost = F.cross_entropy(out, y)
        
        cost.backward()
        optimizer.step()
            
        avg_cost += cost
        print(cost)
        
    print("Avg_cost : ", avg_costs)
        
        
        

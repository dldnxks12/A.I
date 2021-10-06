import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()        
        '''
            input size : 32 x 32 x 1 channel image            
        '''
        # Conv Layer
        self.relu = nn.ReLU()
        self.pooling = nn.AvgPool2d(kernel_size = (2,2), stride = (2,2))
        self.conv1 = nn.Conv2d(1, 6, kernel_size = (5,5), stride = (1,1), padding = (0,0))
        self.conv2 = nn.Conv2d(6,16, kernel_size = (5,5), stride = (1,1), padding = (0,0))
        
        # Fc Layer
        self.fc1 = nn.Linear(16*5*5 , 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        '''        
            output size : batch size x 10
        '''

    def forward(self, x):
        
        # Conv Layer
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pooling(x)
        
        # Fc Layer
        x = x.reshape(x.shape[0], -1) # flatten           
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x        
        
def test_LeNet():
    
    x = torch.randn(64, 1, 32, 32)
    print(x[0])
    model = LeNet()
    
    return model(x)
    
hypothesis = test_LeNet()            

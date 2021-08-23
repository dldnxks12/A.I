import torch
import torch.nn as nn
import torch.optim as optim # for optimizer 
import torch.nn.functional as F # for Softmax function

import torchvision.transforms as transforms # for tensor transforming
from torch.utils.data import TensorDataset  # for make dataset type
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

# Upsample ? Pooling으로 인해서 작아진 것을 복원 with pytorch의 upsample 연산으로 구함 
# SegNet 논문에서는 Pooling하는 과정에서 Index를 기억해두었다가 Unpooling 시 이를 이용해서 UnPooling을 진행 
# Pytorch에서 제공하는 UpSampling (with bilinear)을 이용해서 수행!

# Paper architecture 

# Encoder 
# conv - conv - pool 
# conv - conv - pool
# conv - conv - conv - pool
# conv - conv - conv - pool
# conv - conv - conv - pool

# Decoder 
# upsample - conv - conv - conv
# upsample - conv - conv - conv
# upsample - conv - conv - conv
# upsample - conv - conv
# upsample - conv - conv - softmax 

# DIY architecture 

# Encoder 
# conv - conv - pool 
# conv - conv - pool
# conv - conv - pool

# Decoder 
# upsample - conv - conv 
# upsample - conv - conv 
# upsample - conv - conv - softmax 

class SegNet(nn.Module):
    def __init__(self):
        super().__init__()
        in_channel = 3
        in_height  = 224
        in_width   = 224
        
        num_class  = 2
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, 16, kernel_size = 3,  stride = 1, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace = True),
            nn.Conv2d(16, 32, kernel_size = 3,  stride = 1, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            
            nn.Conv2d(32, 64, kernel_size = 3,  stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 128, kernel_size = 3,  stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            
            nn.Conv2d(128, 256, kernel_size = 3,  stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 512, kernel_size = 3,  stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 1024, kernel_size = 3,  stride = 1, padding = 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),                                    
        )
        
        self.decoder = nn.Sequential(        
            nn.Upsample(scale_factor=2, mode ='bilinear', align_corners = True),
            nn.Conv2d(1024, 512, kernel_size =3, stride =1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 256, kernel_size =3, stride =1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 128, kernel_size =3, stride =1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            
            nn.Upsample(scale_factor=2, mode ='bilinear', align_corners = True),            
            nn.Conv2d(128, 64, kernel_size =3, stride =1, padding = 1),            
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 32, kernel_size =3, stride =1, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
            
            nn.Upsample(scale_factor=2, mode ='bilinear', align_corners = True),            
            nn.Conv2d(32, 16, kernel_size =3, stride =1, padding = 1),          
            nn.BatchNorm2d(16),
            nn.ReLU(inplace = True),
            
            nn.Conv2d(16, num_class, kernel_size =3, stride =1, padding = 1)                      
        )
        
    def forward(self, x): # input Image : Batchsize x 3 x 224 x 224        
        out = self.encoder(x)
        out = self.decoder(out)
        
        out = F.softmax(out, dim = 1) # channel에 대해 softmax 진행  - pixel wise classification
        
        return out
    
model = SegNet()
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.7)
criterion = nn.MSELoss()

train_x = np.load("argu_x.npy")
train_y = np.load("argu_y.npy")

train_x = np.transpose(train_x, (0,3,1,2))
train_y = np.transpose(train_y, (0,3,1,2))

train_y = train_y[:, :2, :, :]

# BUSI label threshold 
thresh_np1 = np.zeros_like(train_y[:, 0, : ,:])
thresh_np2 = np.zeros_like(train_y[:, 1, : ,:])

thresh_np1[ train_y[:, 0, : ,:] < 10] = 1
thresh_np2[ train_y[:, 1, : ,:] > 10] = 1

train_y[:, 0, : ,:] = thresh_np1
train_y[:, 1, : ,:] = thresh_np2

train_x = torch.Tensor(train_x)
train_y = torch.Tensor(train_y)

print(train_x.shape)
print(train_y.shape)
print(type(train_x))

test_x = train_x[0]
test_y = train_y[0]
test_x2 = train_x[1]
test_y2 = train_y[1]

train_x = train_x[1:]
train_y = train_y[1:]

test_x = test_x.unsqueeze(0)
test_y = test_y.unsqueeze(0)
test_x2 = test_x2.unsqueeze(0)
test_y2 = test_y2.unsqueeze(0)

print(test_x.shape)
print(test_y.shape)
print(test_x2.shape)
print(test_y2.shape)

train_dataset = TensorDataset(train_x, train_y)

# DataLoader 
train_loader = DataLoader( dataset = train_dataset, batch_size = 50, shuffle = True, drop_last = True )

# train
for epoch in range(1):    
    avg_cost = 0
    batch_length = len(train_loader)
    for x, y in train_loader:
        optimizer.zero_grad()
        
        output = model(x)                
        cost = criterion(output , y)        
        cost.backward()        
        optimizer.step()
        avg_cost += cost / batch_length        
        print("epoch & Loop cost : ", epoch, cost)
    print("Avg_cost: ", avg_cost)        
    
def color_map(image, nc = 1):
    
    label_colors = np.array([(255, 255, 255), (0, 0, 0)]) 
    
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    
    for I in range(0, nc):
        idx = image == 1
        
        r[idx] = label_colors[I, 0]
        g[idx] = label_colors[I, 1]
        b[idx] = label_colors[I, 2]
    
    rgb = np.stack([r, g, b], axis = 2)
    
    return rgb    
  
  
with torch.no_grad():
    prediction = model(test_x)
    prediction = prediction.squeeze()
    
    pred = torch.argmax(prediction, dim = 0).detach().numpy()
    
rgb_pred = color_map(pred)
print(rgb_pred.shape)  

import matplotlib.pyplot as plt

plt.subplot(1,2,1)
plt.imshow(pred)
plt.subplot(1,2,2)
plt.imshow(test_y[0][0])

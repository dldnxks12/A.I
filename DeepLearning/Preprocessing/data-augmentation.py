import torchvision
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader

import numpy as np

# for visualization
import matplotlib.pyplot as plt
%matplotlib inline 

tf = transforms.Compose([        
        transforms.Resize((224,224)),
])

data1 = torchvision.datasets.ImageFolder(root='./origin/', transform = tf)

x_data = []
y_data = []

# normal image
for idx, value in enumerate(data1):
    
    data, label = value    
    data = np.array(data)
    
    if label == 0:
        x_data.append(data)
    else:
        y_data.append(data)       

# horizontal flip image        
for idx, value in enumerate(data1):
    
    data, label = value
    data = transforms.functional.hflip(data)
    data = np.array(data)
    
    if label == 0:
        x_data.append(data)
    else:
        y_data.append(data)
               
        
# vertical flip image
for idx, value in enumerate(data1):
    
    data, label = value
    data = transforms.functional.vflip(data)
    data = np.array(data)
    
    if label == 0:
        x_data.append(data)
    else:
        y_data.append(data)
  
x_data = np.array(x_data)
y_data = np.array(y_data)

print(x_data.shape) 
print(y_data.shape) 

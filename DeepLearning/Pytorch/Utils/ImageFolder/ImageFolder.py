import torchvision
import torchvision.transforms as transforms # image resize를 위한 transforms 
from torch.utils.data import DataLoader

# for visualization
import matplotlib.pyplot as plt
%matplotlib inline # 출력 값이 Jupyter Notebook 상에서 보이도록 


tf = transforms.Compose([
        transforms.Resize((64,128))
])

train_data = torchvision.datasets.ImageFolder(root = 'sample_data', transform = tf)

# Image Loading 
for idx, value in train_data:
    data, label = value
    
    print(idx, sample, label)
    
    # Image 나누어 저장 
    if(label == 0):
        data.save("custom_data/train_data/gray/%d_%d.jpeg" %(num, label))
    else:
        data.save("custom_data/train_data/red/%d_%d.jpeg" %(num, label))
    

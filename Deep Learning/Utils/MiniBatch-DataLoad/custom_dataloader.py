'''

https://wikidocs.net/57165

파이토치에서 데이터셋을 좀 더 쉽게 다룰 수 있도록 유용한 도구로서 torch.utils.data.Dataset과 torch.utils.data.DataLoader를 제공
기본 사용 법은 Dataset을 정의하고 이를 DataLoader에 전달하는 것이었다.
하지만 이 torch.utils.data.Dataset을 상속받아서 직접 데이터셋을 커스텀하는 경우가 많다.
다음은 이 Dataset을 상속받아 다음 method 들을 Override 해서 데이터셋을 만드는 방법이다.


# 커스텀 데이터셋을 만들 때 기본 뼈대가 되는 define은 총 3개이다.

1. __init__()    --- 데이터셋의 전처리를 해주는 부분 
2. __getitem__() --- 데이터셋에서 특정 1개의 샘플을 가져다 주는 함수 (dataset[i]를 했을 때, i번 째 샘플을 가져오도록 하는 함수 )
3. __len__()     --- 데이터셋의 길이. 즉, 총 샘플 개수를 알려주는 함수 


# for batch_idx, samples in enumerate(dataloader):
            ---- dataloader에서 batch_idx에 해당하는 sample을 하나씩 가져온다. 

  enumerate? --- return ( index, dataloader(index) )

'''


import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CustomDataset(torch.utils.data.Dataset):
    
    def __init__(self):
        self.x_data = [[73, 80, 75],
                       [93, 88, 93],
                       [89, 91, 90],
                       [96, 98, 100],
                       [73, 66, 70]]
        self.y_data = [[152], [185], [180], [196], [142]]
        
    def __getitem__(self, idx):
        
        x = torch.FloatTensor(self.x_data[idx])  # tensor로 바꾸어서 return 
        y = torch.FloatTensor(self.y_data[idx])  # tensor로 바꾸어서 return 
        
        return x,y
    
    def __len__(self):
        return len(self.x_data)
               

dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size = 2, shuffle = True)

model = nn.Linear(3,1)
optimizer = optim.SGD(model.parameters(), lr = 1e-5)

num_epochs = 10
for epoch in range(num_epochs+1):
    for batch_idx, samples in enumerate(dataloader):
        print("Epoch", epoch)
        print("Idx : " , batch_idx)
        print("Sample : ", samples) # enumerate ----  ( index , dataloader[index] )

# ------------------------ test ---------------------- #

new_var = torch.FloatTensor([[40,80,90]])
pred = model(new_var) # forward 수행

# return 되는 값도 tensor 일것

print(pred.item())

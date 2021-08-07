from torch.utils.data import Dataset, DataLoader 
from torch import from_numpy, tensor
import numpy as np

# from_numpy() ---- numpy type to tensor type


# DataLoader의 dataset 인자에 넣어주기 위한 객체 class는 아래와 같이 적어도 __getitem__(), __len__() 함수를 포함하여 구성해야한다.
class OurDataset(Dataset):  
  # Initialize our data, download, ...
  
  def __init__(self):
    xy = np.loadtxt("dataset_path..") # dataset 불러오기
    self.len = xy.shape[0]
    self.x_data = from_numpy(xy[: , 0:-1]) # numpy to tensor 
    self.y_data = from_numpy(xy[: , [-1]])  # [-1]로 감싸주는 이유 무엇? --- check 해보기 
    
  def _getitem__(self, index): # load한 data를 차례차례 return 해주기 
    return self.x_data[index],self.y_data[index]
  
  def __len__(self):  # 전체 data 길이 알려주기
    return self.len
  
  
dataset = OurDataset() # dataset 객체 생성 
train_loader = DataLoader(dataset = dataset, 
                         batch_size = 32, 
                         shuffle = True,
                         num_workers = 2)


# dataloader의 return 값? 

# enumerate(x) ---- (index , element) 형태로 return 

# for i , data in enumerate(train_loader):
# i    --- index
# data --- train_loader의 element 
# data? ---inputs , labels  (x_data, y_data)로 구성  


for epoch in range(2):  
  # train_loader의 return 값이 궁금하다 --- desktop에서 돌려보며 확인해보자 
  # enumerate(train_loader, 1) 과 같이 쓸 때와 어떤 차이가 있는지.. 
  for i , data in enumerate(train_loader, 0): 
    # get the inputs
    inputs, labels = data # 해당 data가 어떤 값을 담고 있는지 ..
    
    # type casting
    
    inputs, labels = tensor(inputs), tensor(labels)
    
    # print(f' ') -> {} 안에 인자 바로 넣어서 사용 가능한 기능 python 3 부터 사용 가능 
    print(f'Epoch : {i} | Inputs {inputs.data} | Labels {labels.data}') 
    




    
    

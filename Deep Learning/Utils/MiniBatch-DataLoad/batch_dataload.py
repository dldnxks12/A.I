import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# TensorDataset과 DataLoader를 가져오자
from torch.utils.data import TensorDataset # 텐서 데이터 셋
from torch.utils.data import DataLoader  # 데이터 로더

# TensorDataset은 기본적으로 Tensor를 입력으로 넣어주어야 한다.

x_train  =  torch.FloatTensor([[73,  80,  75], 
                               [93,  88,  93], 
                               [89,  91,  90], 
                               [96,  98,  100],   
                               [73,  66,  70]])  
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

dataset = TensorDataset(x_train, y_train)

# 이제 위 dattaset을 dataloader에 넣어주면 된다.
# 기본적으로 dataloader는 dataset과 미니 배치 사이즈를 인자로 받는다 .
dataloader = DataLoader(dataset, batch_size = 2, shuffle = True)

model = nn.Linear(3,1)

optimizer = optim.SGD(model.parameters(), lr = 1e-5)

num_epochs = 20

for epoch in range(num_epochs+1):
  for batch_idx, samples in enumerate(dataloader):

    print(samples) # 작게 쪼갠 Mini batch에서 데이터 하나 뽑아옴  (x 데이터 1개, y 데이터 1개 )
    x_train, y_train = samples

    # forward
    prediction = model(x_train)

    cost = F.mse_loss(prediction, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()


    print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(epoch, num_epochs, batch_idx+1, len(dataloader),cost.item()))

'''
Epoch    0/20 Batch 1/3 Cost: 17481.763672
Epoch    0/20 Batch 2/3 Cost: 6258.370117
Epoch    0/20 Batch 3/3 Cost: 2014.073730
Epoch    1/20 Batch 1/3 Cost: 447.541016
Epoch    1/20 Batch 2/3 Cost: 325.316010
Epoch    1/20 Batch 3/3 Cost: 33.595448
Epoch    2/20 Batch 1/3 Cost: 41.740875
Epoch    2/20 Batch 2/3 Cost: 16.555515
Epoch    2/20 Batch 3/3 Cost: 0.209925
Epoch    3/20 Batch 1/3 Cost: 19.399654
Epoch    3/20 Batch 2/3 Cost: 6.796311
Epoch    3/20 Batch 3/3 Cost: 22.546991
Epoch    4/20 Batch 1/3 Cost: 6.005472
Epoch    4/20 Batch 2/3 Cost: 18.306740
Epoch    4/20 Batch 3/3 Cost: 10.430309
Epoch    5/20 Batch 1/3 Cost: 9.464766
Epoch    5/20 Batch 2/3 Cost: 19.013344
Epoch    5/20 Batch 3/3 Cost: 3.184277
Epoch    6/20 Batch 1/3 Cost: 11.755163
Epoch    6/20 Batch 2/3 Cost: 16.270420
Epoch    6/20 Batch 3/3 Cost: 3.336910
Epoch    7/20 Batch 1/3 Cost: 8.335684
Epoch    7/20 Batch 2/3 Cost: 18.659006
Epoch    7/20 Batch 3/3 Cost: 6.918692
Epoch    8/20 Batch 1/3 Cost: 12.454549
Epoch    8/20 Batch 2/3 Cost: 15.810470
Epoch    8/20 Batch 3/3 Cost: 6.598351
Epoch    9/20 Batch 1/3 Cost: 27.404625
Epoch    9/20 Batch 2/3 Cost: 9.818270
Epoch    9/20 Batch 3/3 Cost: 8.571361
Epoch   10/20 Batch 1/3 Cost: 1.927659
Epoch   10/20 Batch 2/3 Cost: 23.376205
Epoch   10/20 Batch 3/3 Cost: 16.380037
Epoch   11/20 Batch 1/3 Cost: 12.742451
Epoch   11/20 Batch 2/3 Cost: 11.695382
Epoch   11/20 Batch 3/3 Cost: 10.472042
Epoch   12/20 Batch 1/3 Cost: 11.858758
Epoch   12/20 Batch 2/3 Cost: 16.292469
Epoch   12/20 Batch 3/3 Cost: 3.167581
Epoch   13/20 Batch 1/3 Cost: 19.706644
Epoch   13/20 Batch 2/3 Cost: 3.118837
Epoch   13/20 Batch 3/3 Cost: 19.046480
Epoch   14/20 Batch 1/3 Cost: 5.971957
Epoch   14/20 Batch 2/3 Cost: 15.020621
Epoch   14/20 Batch 3/3 Cost: 16.556763
Epoch   15/20 Batch 1/3 Cost: 19.784523
Epoch   15/20 Batch 2/3 Cost: 2.947240
Epoch   15/20 Batch 3/3 Cost: 19.176291
Epoch   16/20 Batch 1/3 Cost: 11.178347
Epoch   16/20 Batch 2/3 Cost: 13.027730
Epoch   16/20 Batch 3/3 Cost: 11.371270
Epoch   17/20 Batch 1/3 Cost: 19.724218
Epoch   17/20 Batch 2/3 Cost: 7.053710
Epoch   17/20 Batch 3/3 Cost: 6.649404
Epoch   18/20 Batch 1/3 Cost: 12.284131
Epoch   18/20 Batch 2/3 Cost: 2.110827
Epoch   18/20 Batch 3/3 Cost: 36.495712
Epoch   19/20 Batch 1/3 Cost: 12.266048
Epoch   19/20 Batch 2/3 Cost: 16.370943
Epoch   19/20 Batch 3/3 Cost: 9.877811
Epoch   20/20 Batch 1/3 Cost: 12.571155
Epoch   20/20 Batch 2/3 Cost: 12.011531
Epoch   20/20 Batch 3/3 Cost: 11.481649
'''


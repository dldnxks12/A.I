from tqdm import tqdm # pip install tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

x = torch.randn((1000,3,224,224))
y = torch.randint(low = 0, high = 10, size = (1000,1)) # 0 ~ 10 사이 shape = [1000 , 1]

ds = TensorDataset(x, y)
train_loader = DataLoader(dataset = ds, batch_size = 8)

model = nn.Sequential( nn.Conv2d(3, 10, 3, 1, 1),
                       nn.Flatten(),
                       nn.Linear(10*224*224, 10)
                       )

NUM_EPOCHS = 100

for epoch in range(NUM_EPOCHS):
    loop = tqdm(train_loader, leave = False) # leave = False 하면 progress bar 가 사라지고 다시 생김
    for idx, (x, y) in enumerate(loop):

        prediction = model(x)

        loop.set_description(f"Epoch [{epoch} /{NUM_EPOCHS}]")
        loop.set_postfix(loss = torch.rand(1).item(), acc = torch.rand(1).item()) # 걍 랜덤으로 값 넣었지만, 위에서 loss랑 이런거 계산한 거 넣어주면 된다.



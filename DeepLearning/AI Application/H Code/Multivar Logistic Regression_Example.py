import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

x_train = torch.FloatTensor([
    [1,2,1,1],[2,1,3,2],[3,1,3,4],[4,1,5,5],[1,7,5,5],[1,2,5,6],[1,6,6,6],[1,7,7,7]
])

y_train = torch.LongTensor([2,2,2,1,1,1,0,0]) # 이전에는 [ [1],[2],]... 이렇게 넣어줬었는데
# 이걸 그냥 벡터로 넣어버렸다. --- softmax의 결과와 비교할 것이기 때문에 One-hot Encoding을 시켜줄 것
# CE loss 함수가 y_train의 one-hot encoding 내부적으로 시켜준다.

# 목적
# Softmax( linear(X) ) == onehot(y) 가 되게끔 학습 !

# activation function : Softmax
# cost function : CE loss

# but 우리가 torch.nn.functional.cross_entropy를 사용하면 내부적으로 softmax가 수행이된다.
# 따라서 추가적으로 linear 모델을 통과시킨 후 softmax를 수행하지 않아도 된다.
model = nn.Linear(4,3) # Input dim, Output dim

optimizer = optim.SGD(model.parameters(), lr = 0.1)

for epoch in range(1000):
    hypothesis = model(x_train)
    cost = F.cross_entropy(hypothesis, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 99:
        correct_prediction = hypothesis.argmax(dim = 1) == y_train
        acc = correct_prediction.sum().item() / len(correct_prediction)
        print(f" Epoch {epoch} Cost {cost.item()} Acc {acc*100}%")





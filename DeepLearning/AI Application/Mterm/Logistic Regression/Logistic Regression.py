import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

dataset = np.loadtxt("data_multinomial_classification.csv", delimiter=',', dtype = np.float32)

print(dataset.shape) # (101, 17)

X_train = torch.FloatTensor(dataset[:, :-1])
Y_train = torch.LongTensor(dataset[:,-1]) # dataset[: , [-1]] 로 하지 않는 것은 CE에서 Onehot 인코딩할 것 이기에!

# Todo --- model 설계
# Todo --- Train loop 설계

class Logistic(nn.Module):
    def __init__(self):
        super(Logistic, self).__init__()
        self.Linear = nn.Linear(16, 7)  # 8개의 feature가 들어가서 1개의 output을 낼 것?

    def forward(self, X):
        Out = self.Linear(X) # OUt :  101 x 8
        return Out


model = Logistic()
optimizer = optim.SGD(model.parameters(), lr = 0.01)

for epoch in range(1000):

    hypothesis = model(X_train)
    cost = F.cross_entropy(hypothesis, Y_train) # Softmax + One-hot Encoding

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:

        prediction = torch.argmax(hypothesis, axis = 1)
        correction = (prediction == Y_train).sum() / len(prediction)

        print("Correct", correction)





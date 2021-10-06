import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

print("Solve Ax = B with Linear Regresion")

# A = x
# B = y
# y = wx -> B = wA
A = torch.FloatTensor([[0,1],[1,1],[2,1],[3,1]])
B = torch.FloatTensor([[-1],[0.2],[0.9],[2.1]])

print(A.shape)
print(B.shape)

model = nn.Linear(2,1, bias = False)
optimizer = optim.SGD(model.parameters(), lr = 0.001)

for epoch in range(30000):

    hypothesis = model(A)
    cost = F.mse_loss(hypothesis, B)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 99:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch+1, 1000, cost.item()))


result = list(model.parameters())

print("Result with Linear Regression : ", result)

A = torch.FloatTensor([[0,1],[1,1],[2,1],[3,1]])
B = torch.FloatTensor([[-1],[0.2],[0.9],[2.1]])

Inverse = torch.inverse(A.T.matmul(A))
A_plus = Inverse.matmul(A.T)

left = A_plus.matmul(A)
right = A_plus.matmul(B)
left_inverse = torch.inverse(left)
x = left_inverse.matmul(right)

print("Result with Pseudo Inverse Matrix :", x)





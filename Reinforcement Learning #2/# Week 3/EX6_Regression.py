import numpy as np
import matplotlib.pyplot as plt
import torch

# Make test data
num_points = 1000

x = []
y = []

for i in range(num_points):
    temp_x = np.random.normal(0.0, 1)
    temp_y = 0.1 * temp_x + 0.3 + np.random.normal(0.0, 0.03)

    x.append(temp_x)
    y.append(temp_y)

x = torch.tensor(x)
y = torch.tensor(y)

plt.plot(x, y, 'ro')
plt.show()

# Make model

W = torch.tensor(-0.6, requires_grad = True) # Gradient 변화를 기록
b = torch.tensor(0.0, requires_grad = True)
learning_rate = 0.0001

loss_list = []

# Do Gradient Descent

for step in range(1000):
    loss = ((W*x + b - y)**2).sum()

    loss.backward()

    with torch.no_grad(): # Gradient 를 더 변화시키지 않고서
        W -= learning_rate * W.grad
        b -= learning_rate * b.grad

    W.grad.zero_()
    b.grad.zero_()

    if step % 100 == 0:
        plt.plot(x, y, 'ro')
        plt.plot(x, W.detach() * x + b.detach()) # detach : tensor에서 떼어내기
        plt.show()

    loss_list.append(loss)

# check training procedure

print("W : ", W.detach())
print("b : ", b.detach())
plt.plot(range(1000), loss_list)
plt.show()
plt.plot(x, y, 'ro')
plt.plot(x, W.detach() * x + b.detach())
plt.show()
